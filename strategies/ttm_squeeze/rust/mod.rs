// ttm_squeeze: John Carter TTM Squeeze (BB inside KC -> fire + momentum).
//
// Squeeze ON when Bollinger Bands sit entirely inside Keltner Channels.
// Fire = first OFF bar after at least `min_squeeze_bars` ON bars.
// Entry direction from smoothed momentum histogram sign.
// Exit on consecutive adverse-momentum bars or time stop.
// Close-confirmed; engine fills next open.

use crate::data::MarketData;
use crate::recipe::{PluginSpec, Recipe};
use crate::runtime::{BatchContext, SignalBuffers};

pub fn build_signals(
    data: &MarketData,
    recipe: &Recipe,
    genome: &[f64],
    _spec: &PluginSpec,
    _ctx: &BatchContext,
    buffers: &mut SignalBuffers,
) {
    build_ttm_squeeze_signals(data, recipe, genome, buffers);
}

fn build_ttm_squeeze_signals(
    data: &MarketData,
    recipe: &Recipe,
    genome: &[f64],
    buffers: &mut SignalBuffers,
) {
    let n = data.len();
    buffers.resize(n);
    buffers.entries.fill(false);
    buffers.exits.fill(false);
    buffers.directions.fill(1);

    let bb_period = param_i32(recipe, genome, "bb_period", 20).max(5) as usize;
    let bb_mult = param_f64(recipe, genome, "bb_mult", 2.0).clamp(0.5, 4.0);
    let kc_mult = param_f64(recipe, genome, "kc_mult", 1.5).clamp(0.5, 4.0);
    let atr_period = param_i32(recipe, genome, "atr_period", 14).max(2) as usize;
    let mom_period = param_i32(recipe, genome, "mom_period", 12).max(2) as usize;
    let mom_smooth = param_i32(recipe, genome, "mom_smooth", 3).max(1) as usize;
    let min_squeeze_bars = param_i32(recipe, genome, "min_squeeze_bars", 3).max(1) as usize;
    let allow_short = param_i32(recipe, genome, "allow_short", 1) > 0;
    let mom_exit_bars = param_i32(recipe, genome, "mom_exit_bars", 2).max(1) as usize;
    let exit_after_bars = param_i32(recipe, genome, "exit_after_bars", 12).max(1) as usize;

    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    let bb_mid = simple_sma(&data.close, bb_period);
    let bb_std = rolling_stdev(&data.close, bb_period);
    let kc_mid = ema(&data.close, bb_period);
    let atr = wilder_atr(data, atr_period);
    let raw_mom = donchian_mid_momentum(data, mom_period);
    let mom = simple_sma(&raw_mom, mom_smooth);

    let mut squeeze_on = vec![false; n];
    for i in 0..n {
        let mid = bb_mid[i];
        let sd = bb_std[i];
        let kmid = kc_mid[i];
        let a = atr[i];
        if !(mid.is_finite() && sd.is_finite() && kmid.is_finite() && a.is_finite() && a > 0.0) {
            continue;
        }
        let bb_u = mid + bb_mult * sd;
        let bb_l = mid - bb_mult * sd;
        let kc_u = kmid + kc_mult * a;
        let kc_l = kmid - kc_mult * a;
        squeeze_on[i] = bb_u < kc_u && bb_l > kc_l;
    }

    let mut on_run = 0usize;
    let mut in_pos = false;
    let mut pos_dir: i8 = 1;
    let mut entry_i = 0usize;
    let mut adverse_run = 0usize;

    for i in 0..n {
        if squeeze_on[i] {
            on_run = on_run.saturating_add(1);
        }

        if in_pos {
            let held = i.saturating_sub(entry_i);
            let m = mom[i];
            let adverse = if pos_dir > 0 {
                m.is_finite() && m < 0.0
            } else {
                m.is_finite() && m > 0.0
            };
            if adverse {
                adverse_run = adverse_run.saturating_add(1);
            } else {
                adverse_run = 0;
            }
            if adverse_run >= mom_exit_bars || held >= exit_after_bars {
                buffers.exits[i] = true;
                in_pos = false;
                adverse_run = 0;
                if !squeeze_on[i] {
                    on_run = 0;
                }
                continue;
            }
        }

        let fired = i > 0
            && !squeeze_on[i]
            && squeeze_on[i - 1]
            && on_run >= min_squeeze_bars;

        if !in_pos && fired {
            let m = mom[i];
            if m.is_finite() && m > 0.0 {
                buffers.entries[i] = true;
                buffers.directions[i] = 1;
                in_pos = true;
                pos_dir = 1;
                entry_i = i;
                adverse_run = 0;
            } else if allow_short && m.is_finite() && m < 0.0 {
                buffers.entries[i] = true;
                buffers.directions[i] = -1;
                in_pos = true;
                pos_dir = -1;
                entry_i = i;
                adverse_run = 0;
            }
        }

        if !squeeze_on[i] {
            on_run = 0;
        }
    }
}

fn donchian_mid_momentum(data: &MarketData, period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period < 2 || n < period {
        return out;
    }
    for i in period - 1..n {
        let start = i + 1 - period;
        let mut hh = f64::NEG_INFINITY;
        let mut ll = f64::INFINITY;
        let mut ok = true;
        for j in start..=i {
            let h = data.high[j];
            let l = data.low[j];
            if !(h.is_finite() && l.is_finite()) {
                ok = false;
                break;
            }
            if h > hh {
                hh = h;
            }
            if l < ll {
                ll = l;
            }
        }
        let c = data.close[i];
        if ok && c.is_finite() && hh.is_finite() && ll.is_finite() && hh >= ll {
            out[i] = c - 0.5 * (hh + ll);
        }
    }
    out
}

fn rolling_stdev(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![f64::NAN; n];
    if period < 2 || n < period {
        return out;
    }
    for i in period - 1..n {
        let start = i + 1 - period;
        let mut sum = 0.0;
        let mut ok = true;
        for j in start..=i {
            if !values[j].is_finite() {
                ok = false;
                break;
            }
            sum += values[j];
        }
        if !ok {
            continue;
        }
        let mean = sum / period as f64;
        let mut var = 0.0;
        for j in start..=i {
            let d = values[j] - mean;
            var += d * d;
        }
        out[i] = (var / period as f64).sqrt();
    }
    out
}

fn ema(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![f64::NAN; n];
    if period < 2 || n < period {
        return out;
    }
    let mut sum = 0.0;
    for i in 0..period {
        if !values[i].is_finite() {
            return out;
        }
        sum += values[i];
    }
    let mut prev = sum / period as f64;
    out[period - 1] = prev;
    let alpha = 2.0 / (period as f64 + 1.0);
    for i in period..n {
        let v = values[i];
        if !v.is_finite() {
            continue;
        }
        prev = alpha * v + (1.0 - alpha) * prev;
        out[i] = prev;
    }
    out
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
