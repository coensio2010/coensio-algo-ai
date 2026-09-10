// nr_expand: Toby Crabel NR compression -> expansion breakout.
//
// Bar k is NR when range[k] is the minimum of ranges [k+1-nr_period .. k]
// (completed bars only). Freeze high/low of that NR bar.
// Entry on a later bar i (i > k, waited >= 1, within arm_window) when
// close[i] breaks the frozen range by buffer_atr * ATR[i].
// Never enter on the arm/NR bar. Exit: ATR stop, ATR target, or time.
// Session boundary clears pending and flattens.
// Close-confirmed; engine fills next open.

use crate::data::MarketData;
use crate::recipe::{PluginSpec, Recipe};
use crate::runtime::{BatchContext, SignalBuffers};

pub fn build_signals(
    data: &MarketData,
    recipe: &Recipe,
    genome: &[f64],
    spec: &PluginSpec,
    ctx: &BatchContext,
    buffers: &mut SignalBuffers,
) {
    build_nr_expand_signals(data, recipe, genome, spec, ctx, buffers);
}

fn build_nr_expand_signals(
    data: &MarketData,
    recipe: &Recipe,
    genome: &[f64],
    spec: &PluginSpec,
    ctx: &BatchContext,
    buffers: &mut SignalBuffers,
) {
    let n = data.len();
    buffers.resize(n);
    buffers.entries.fill(false);
    buffers.exits.fill(false);
    buffers.directions.fill(1);

    let nr_period = param_i32(recipe, genome, "nr_period", 7).clamp(3, 24) as usize;
    let arm_window = param_i32(recipe, genome, "arm_window", 3).clamp(1, 16) as usize;
    let buffer_atr = param_f64(recipe, genome, "buffer_atr", 0.05).clamp(0.0, 1.0);
    let atr_period = param_i32(recipe, genome, "atr_period", 14).max(2) as usize;
    let stop_mult = param_f64(recipe, genome, "stop_mult", 1.5).clamp(0.3, 5.0);
    let target_mult = param_f64(recipe, genome, "target_mult", 2.0).clamp(0.5, 6.0);
    let allow_short = param_i32(recipe, genome, "allow_short", 1) > 0;
    let exit_after_bars = param_i32(recipe, genome, "exit_after_bars", 12).max(1) as usize;

    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    let hour = spec.session_utc_start.first().copied().unwrap_or(14);
    let minute = spec.session_utc_start.get(1).copied().unwrap_or(30);
    let session_days: std::sync::Arc<Vec<i32>> = ctx.session_days.clone().unwrap_or_else(|| {
        std::sync::Arc::new(crate::session::session_days_from_market(data, hour, minute))
    });

    let atr = wilder_atr(data, atr_period);
    let mut range = vec![f64::NAN; n];
    for i in 0..n {
        let h = data.high[i];
        let l = data.low[i];
        if h.is_finite() && l.is_finite() && h >= l {
            range[i] = h - l;
        }
    }

    let mut in_pos = false;
    let mut pos_dir: i8 = 1;
    let mut entry_i = 0usize;
    let mut stop_level = f64::NAN;
    let mut target_level = f64::NAN;

    let mut pending = false;
    let mut pending_at = 0usize;
    let mut pending_hi = f64::NAN;
    let mut pending_lo = f64::NAN;

    let mut cur_day = i32::MIN;
    let warm = nr_period.max(atr_period) + 2;

    for i in 0..n {
        let day = session_days[i];
        if day != cur_day {
            cur_day = day;
            pending = false;
            if in_pos {
                buffers.exits[i] = true;
                in_pos = false;
            }
        }

        let c = data.close[i];
        if !c.is_finite() {
            continue;
        }

        if in_pos {
            let held = i.saturating_sub(entry_i);
            let stop_hit = if pos_dir > 0 {
                stop_level.is_finite() && c <= stop_level
            } else {
                stop_level.is_finite() && c >= stop_level
            };
            let tgt_hit = if pos_dir > 0 {
                target_level.is_finite() && c >= target_level
            } else {
                target_level.is_finite() && c <= target_level
            };
            if stop_hit || tgt_hit || held >= exit_after_bars {
                buffers.exits[i] = true;
                in_pos = false;
                pending = false;
                continue;
            }
            continue;
        }

        if i < warm {
            continue;
        }

        // Identify NR on prior completed bar (exclude current close from the min-range).
        let k = i - 1;
        if is_nr_bar(&range, k, nr_period) {
            let h = data.high[k];
            let l = data.low[k];
            if h.is_finite() && l.is_finite() && h >= l {
                pending = true;
                pending_at = k;
                pending_hi = h;
                pending_lo = l;
            }
        }

        if !pending {
            continue;
        }
        let waited = i.saturating_sub(pending_at);
        if waited < 1 || waited > arm_window {
            if waited > arm_window {
                pending = false;
            }
            continue;
        }

        let a = atr[i];
        if !(a.is_finite() && a > 0.0) {
            continue;
        }
        let buf = buffer_atr * a;
        let long_brk = pending_hi.is_finite() && c > pending_hi + buf;
        let short_brk = pending_lo.is_finite() && c < pending_lo - buf;

        if long_brk {
            buffers.entries[i] = true;
            buffers.directions[i] = 1;
            in_pos = true;
            pos_dir = 1;
            entry_i = i;
            stop_level = c - stop_mult * a;
            target_level = c + target_mult * a;
            pending = false;
        } else if allow_short && short_brk {
            buffers.entries[i] = true;
            buffers.directions[i] = -1;
            in_pos = true;
            pos_dir = -1;
            entry_i = i;
            stop_level = c + stop_mult * a;
            target_level = c - target_mult * a;
            pending = false;
        }
    }
}

fn is_nr_bar(range: &[f64], k: usize, period: usize) -> bool {
    if period < 2 || k + 1 < period {
        return false;
    }
    let rk = range[k];
    if !(rk.is_finite() && rk >= 0.0) {
        return false;
    }
    let start = k + 1 - period;
    for j in start..=k {
        let r = range[j];
        if !r.is_finite() {
            return false;
        }
        if r < rk {
            return false;
        }
    }
    true
}

fn wilder_atr(data: &MarketData, period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period < 2 || n < period {
        return out;
    }
    let mut tr = vec![f64::NAN; n];
    tr[0] = data.high[0] - data.low[0];
    for i in 1..n {
        let hl = data.high[i] - data.low[i];
        let hc = (data.high[i] - data.close[i - 1]).abs();
        let lc = (data.low[i] - data.close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }
    let mut sum = 0.0;
    for i in 0..period {
        sum += tr[i];
    }
    let mut atr = sum / period as f64;
    out[period - 1] = atr;
    for i in period..n {
        atr = (atr * (period as f64 - 1.0) + tr[i]) / period as f64;
        out[i] = atr;
    }
    out
}

fn param_i32(recipe: &Recipe, genome: &[f64], name: &str, default: i32) -> i32 {
    recipe
        .try_param_value(genome, name)
        .map(|v| v.round() as i32)
        .unwrap_or(default)
}

fn param_f64(recipe: &Recipe, genome: &[f64], name: &str, default: f64) -> f64 {
    recipe.try_param_value(genome, name).unwrap_or(default)
}
