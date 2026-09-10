// ib_retrace: Session Initial Balance break + percent-into-range retrace-hold.
//
// Structure: first ib_bars of each session freeze IB high/low (classic first
// hour when tf=1h and ib_bars=1). Chase is noisy. This strategy waits:
//   1) Arm on close-confirm break of frozen IB (optional RVOL + ATR buffer)
//      after IB-width filter (min/max IB height in ATR units)
//   2) Enter only when price retraces retrace_pct of IB height INTO the box
//      within retest_window AND close holds on the break side of that level
// Never enter on the arm bar. Exit: stop at stop_pct of IB into the box from
// the broken extreme, target = target_mult x IB height beyond the extreme,
// or time. Session boundary clears pending/positions.
//
// Related: orb_retrace (N-bar opening range + ATR-tolerance boundary touch).

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
    build_ib_retrace_signals(data, recipe, genome, spec, ctx, buffers);
}

fn build_ib_retrace_signals(
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

    let ib_bars = param_i32(recipe, genome, "ib_bars", 1).clamp(1, 2) as usize;
    let rvol_period = param_i32(recipe, genome, "rvol_period", 20).max(2) as usize;
    let rvol_min = param_f64(recipe, genome, "rvol_min", 1.0).clamp(0.5, 4.0);
    let retest_window = param_i32(recipe, genome, "retest_window", 12).clamp(2, 40) as usize;
    let retrace_pct = param_f64(recipe, genome, "retrace_pct", 0.25).clamp(0.05, 0.6);
    let allow_short = param_i32(recipe, genome, "allow_short", 1) > 0;
    let target_mult = param_f64(recipe, genome, "target_mult", 0.5).clamp(0.2, 4.0);
    let stop_pct = param_f64(recipe, genome, "stop_pct", 0.6).clamp(0.2, 1.0);
    let exit_after_bars = param_i32(recipe, genome, "exit_after_bars", 24).max(1) as usize;
    let atr_period = param_i32(recipe, genome, "atr_period", 14).max(2) as usize;
    let buffer_atr = param_f64(recipe, genome, "buffer_atr", 0.05).clamp(0.0, 1.0);
    let min_ib_atr = param_f64(recipe, genome, "min_ib_atr", 0.4).clamp(0.05, 3.0);
    let max_ib_atr = param_f64(recipe, genome, "max_ib_atr", 3.0).clamp(0.5, 10.0);
    let skip_bars = param_i32(recipe, genome, "skip_bars", 0).clamp(0, 4) as usize;
    let max_ib = max_ib_atr.max(min_ib_atr);

    let hour = spec.session_utc_start.first().copied().unwrap_or(14);
    let minute = spec.session_utc_start.get(1).copied().unwrap_or(30);
    let session_days: std::sync::Arc<Vec<i32>> = ctx.session_days.clone().unwrap_or_else(|| {
        std::sync::Arc::new(crate::session::session_days_from_market(data, hour, minute))
    });

    let atr = wilder_atr(data, atr_period);
    let vol_sma = simple_sma(&data.volume, rvol_period);

    let mut in_pos = false;
    let mut pos_dir: i8 = 1;
    let mut entry_i = 0usize;
    let mut stop_level = f64::NAN;
    let mut target_level = f64::NAN;

    // Session IB build + freeze
    let mut cur_day = i32::MIN;
    let mut bars_in_session = 0usize;
    let mut ib_hi = f64::NAN;
    let mut ib_lo = f64::NAN;
    let mut ib_ready = false;
    let mut ib_ok_width = false;

    // Pending retest after IB break (levels frozen at arm)
    let mut pending_dir: i8 = 0;
    let mut pending_extreme = f64::NAN;
    let mut pending_entry = f64::NAN;
    let mut pending_at = 0usize;
    let mut pending_range = f64::NAN;

    let warm = rvol_period.max(atr_period) + 2;
    for i in 0..n {
        let day = session_days[i];
        if day != cur_day {
            cur_day = day;
            bars_in_session = 0;
            ib_hi = f64::NAN;
            ib_lo = f64::NAN;
            ib_ready = false;
            ib_ok_width = false;
            pending_dir = 0;
            if in_pos {
                buffers.exits[i] = true;
                in_pos = false;
            }
        }
        bars_in_session = bars_in_session.saturating_add(1);

        let h = data.high[i];
        let l = data.low[i];
        let c = data.close[i];
        let v = data.volume[i];
        if !(h.is_finite() && l.is_finite() && c.is_finite() && v.is_finite() && v > 0.0) {
            continue;
        }

        // Build IB from first ib_bars completed bars of the session.
        if bars_in_session <= ib_bars {
            ib_hi = if ib_hi.is_finite() { ib_hi.max(h) } else { h };
            ib_lo = if ib_lo.is_finite() { ib_lo.min(l) } else { l };
            if bars_in_session == ib_bars && ib_hi.is_finite() && ib_lo.is_finite() && ib_hi >= ib_lo
            {
                ib_ready = true;
                let a = atr[i];
                let height = ib_hi - ib_lo;
                ib_ok_width = a.is_finite()
                    && a > 0.0
                    && height.is_finite()
                    && height > 0.0
                    && (height / a) >= min_ib_atr
                    && (height / a) <= max_ib;
            }
            // No arm/enter while IB is still forming (includes freeze bar).
            continue;
        }

        if in_pos {
            let held = i.saturating_sub(entry_i);
            let stop_hit = if pos_dir > 0 {
                stop_level.is_finite() && c <= stop_level
            } else {
                stop_level.is_finite() && c >= stop_level
            };
            let target_hit = if pos_dir > 0 {
                target_level.is_finite() && c >= target_level
            } else {
                target_level.is_finite() && c <= target_level
            };
            if stop_hit || target_hit || held >= exit_after_bars {
                buffers.exits[i] = true;
                in_pos = false;
                continue;
            }
            continue;
        }

        if i < warm || !ib_ready || !ib_ok_width || bars_in_session <= ib_bars + skip_bars {
            continue;
        }

        let a = atr[i];
        if !(a.is_finite() && a > 0.0 && ib_hi.is_finite() && ib_lo.is_finite()) {
            continue;
        }
        let ib_range = ib_hi - ib_lo;
        if !(ib_range.is_finite() && ib_range > 0.0) {
            continue;
        }

        // Manage pending percent-retrace (never enter on arm bar itself).
        if pending_dir != 0 {
            let waited = i.saturating_sub(pending_at);
            if waited > retest_window {
                pending_dir = 0;
            } else if waited >= 1 {
                // Cancel if opposite IB extreme is broken (double-break filter).
                if pending_dir > 0 && c < ib_lo {
                    pending_dir = 0;
                } else if pending_dir < 0 && c > ib_hi {
                    pending_dir = 0;
                } else if pending_dir > 0 {
                    if c < pending_entry {
                        pending_dir = 0;
                    } else {
                        let touched = l.is_finite() && l <= pending_entry;
                        let held = c > pending_entry;
                        if touched && held {
                            buffers.entries[i] = true;
                            buffers.directions[i] = 1;
                            in_pos = true;
                            pos_dir = 1;
                            entry_i = i;
                            stop_level = pending_extreme - stop_pct * pending_range;
                            target_level = pending_extreme + target_mult * pending_range;
                            pending_dir = 0;
                            continue;
                        }
                    }
                } else if c > pending_entry {
                    pending_dir = 0;
                } else {
                    let touched = h.is_finite() && h >= pending_entry;
                    let held = c < pending_entry;
                    if touched && held {
                        buffers.entries[i] = true;
                        buffers.directions[i] = -1;
                        in_pos = true;
                        pos_dir = -1;
                        entry_i = i;
                        stop_level = pending_extreme + stop_pct * pending_range;
                        target_level = pending_extreme - target_mult * pending_range;
                        pending_dir = 0;
                        continue;
                    }
                }
            }
        }

        // Arm new IB break (close-confirm + RVOL). Replaces prior pending.
        let vs = vol_sma[i];
        if !(vs.is_finite() && vs > 0.0) {
            continue;
        }
        let rvol = v / vs;
        if rvol < rvol_min {
            continue;
        }

        let buf = buffer_atr * a;
        if c > ib_hi + buf {
            pending_dir = 1;
            pending_extreme = ib_hi;
            pending_entry = ib_hi - retrace_pct * ib_range;
            pending_at = i;
            pending_range = ib_range;
        } else if allow_short && c < ib_lo - buf {
            pending_dir = -1;
            pending_extreme = ib_lo;
            pending_entry = ib_lo + retrace_pct * ib_range;
            pending_at = i;
            pending_range = ib_range;
        }
    }
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

fn simple_sma(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![f64::NAN; n];
    if period < 1 || n < period {
        return out;
    }
    if period == 1 {
        out.clone_from_slice(values);
        return out;
    }
    for i in period - 1..n {
        let mut sum = 0.0;
        let mut ok = true;
        for j in i + 1 - period..=i {
            if !values[j].is_finite() {
                ok = false;
                break;
            }
            sum += values[j];
        }
        if ok {
            out[i] = sum / period as f64;
        }
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
