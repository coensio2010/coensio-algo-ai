// donchian_atr: Donchian close-confirm + ATR expansion regime.
//
// Entry: close breaks prior entry_len Donchian high/low AND
//        ATR(fast) / ATR(slow) >= atr_ratio_min (vol expanding).
// Exit: opposite exit_len Donchian OR ATR stop OR time stop.
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
    build_donchian_atr_signals(data, recipe, genome, buffers);
}

fn build_donchian_atr_signals(
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

    let entry_len = param_i32(recipe, genome, "entry_len", 20).clamp(8, 80) as usize;
    let exit_len = param_i32(recipe, genome, "exit_len", 10).clamp(3, 40) as usize;
    let atr_fast_p = param_i32(recipe, genome, "atr_fast", 14).clamp(2, 40) as usize;
    let atr_slow_raw = param_i32(recipe, genome, "atr_slow", 50).clamp(10, 120) as usize;
    let atr_ratio_min = param_f64(recipe, genome, "atr_ratio_min", 1.0).clamp(0.5, 2.5);
    let stop_mult = param_f64(recipe, genome, "stop_mult", 2.0).clamp(0.5, 4.0);
    let allow_short = param_i32(recipe, genome, "allow_short", 1) > 0;
    let exit_after_bars = param_i32(recipe, genome, "exit_after_bars", 40).max(1) as usize;

    // Ensure slow > fast so ratio is meaningful.
    let atr_slow_p = atr_slow_raw.max(atr_fast_p + 5);

    let atr_fast = wilder_atr(data, atr_fast_p);
    let atr_slow = wilder_atr(data, atr_slow_p);
    let entry_hi = prior_donchian_high(data, entry_len);
    let entry_lo = prior_donchian_low(data, entry_len);
    let exit_hi = prior_donchian_high(data, exit_len);
    let exit_lo = prior_donchian_low(data, exit_len);

    let mut in_pos = false;
    let mut pos_dir: i8 = 1;
    let mut entry_i = 0usize;
    let mut stop_level = f64::NAN;

    let warm = entry_len.max(exit_len).max(atr_slow_p) + 2;
    for i in warm..n {
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
            let don_exit = if pos_dir > 0 {
                exit_lo[i].is_finite() && c < exit_lo[i]
            } else {
                exit_hi[i].is_finite() && c > exit_hi[i]
            };
            if stop_hit || don_exit || held >= exit_after_bars {
                buffers.exits[i] = true;
                in_pos = false;
                continue;
            }
        }

        if in_pos {
            continue;
        }

        let eh = entry_hi[i];
        let el = entry_lo[i];
        let af = atr_fast[i];
        let as_ = atr_slow[i];
        if !(af.is_finite() && af > 0.0 && as_.is_finite() && as_ > 0.0) {
            continue;
        }
        let ratio = af / as_;
        if ratio < atr_ratio_min {
            continue;
        }

        let long_break = eh.is_finite() && c > eh;
        let short_break = el.is_finite() && c < el;

        if long_break {
            buffers.entries[i] = true;
            buffers.directions[i] = 1;
            in_pos = true;
            pos_dir = 1;
            entry_i = i;
            stop_level = c - stop_mult * af;
        } else if allow_short && short_break {
            buffers.entries[i] = true;
            buffers.directions[i] = -1;
            in_pos = true;
            pos_dir = -1;
            entry_i = i;
            stop_level = c + stop_mult * af;
        }
    }
}

fn prior_donchian_high(data: &MarketData, period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period < 1 || n <= period {
        return out;
    }
    for i in period..n {
        let mut mx = f64::NEG_INFINITY;
        let mut ok = false;
        for j in (i - period)..i {
            let h = data.high[j];
            if h.is_finite() {
                mx = mx.max(h);
                ok = true;
            }
        }
        if ok {
            out[i] = mx;
        }
    }
    out
}

fn prior_donchian_low(data: &MarketData, period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period < 1 || n <= period {
        return out;
    }
    for i in period..n {
        let mut mn = f64::INFINITY;
        let mut ok = false;
        for j in (i - period)..i {
            let l = data.low[j];
            if l.is_finite() {
                mn = mn.min(l);
                ok = true;
            }
        }
        if ok {
            out[i] = mn;
        }
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

fn param_i32(recipe: &Recipe, genome: &[f64], name: &str, default: i32) -> i32 {
    recipe
        .try_param_value(genome, name)
        .map(|v| v.round() as i32)
        .unwrap_or(default)
}

fn param_f64(recipe: &Recipe, genome: &[f64], name: &str, default: f64) -> f64 {
    recipe.try_param_value(genome, name).unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trending_up_expanding() -> MarketData {
        let n = 260usize;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 * 3_600_000.0).collect();
        let mut open = vec![100.0; n];
        let mut high = vec![100.5; n];
        let mut low = vec![99.5; n];
        let mut close = vec![100.0; n];
        let volume = vec![1_000.0; n];
        for i in 0..n {
            // Quiet early, then expanding ranges + uptrend.
            let expand = if i < 80 { 0.25 } else { 0.25 + (i as f64 - 80.0) * 0.02 };
            let px = 100.0 + if i < 80 { 0.0 } else { (i as f64 - 80.0) * 0.55 };
            open[i] = px - expand * 0.3;
            close[i] = px;
            high[i] = px + expand;
            low[i] = px - expand * 0.6;
        }
        MarketData::new(ts, open, high, low, close, volume).unwrap()
    }

    #[test]
    fn donchian_atr_can_enter_long() {
        let data = trending_up_expanding();
        let recipe = crate::recipe::get_recipe("donchian_atr").unwrap();
        // entry, exit, atr_fast, atr_slow, ratio_min, stop, allow_short, exit_after
        let genome = vec![20.0, 10.0, 14.0, 50.0, 0.9, 2.0, 1.0, 60.0];
        let mut bufs = SignalBuffers::default();
        build_donchian_atr_signals(&data, &recipe, &genome, &mut bufs);
        assert!(
            bufs.entries.iter().any(|&e| e),
            "expected at least one Donchian+ATR-expansion long entry"
        );
    }

    #[test]
    fn contracting_vol_gate_rejects() {
        let n = 160usize;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 * 3_600_000.0).collect();
        let mut open = vec![100.0; n];
        let mut high = vec![100.3; n];
        let mut low = vec![99.7; n];
        let mut close = vec![100.0; n];
        let volume = vec![1_000.0; n];
        for i in 0..n {
            // Shrinking ranges: early wide, late tiny (contracting ATR ratio).
            let w = if i < 60 { 1.5 } else { 0.15 };
            let wiggle = if i % 2 == 0 { 0.05 } else { -0.05 };
            open[i] = 100.0;
            close[i] = 100.0 + wiggle;
            high[i] = 100.0 + w;
            low[i] = 100.0 - w;
        }
        let data = MarketData::new(ts, open, high, low, close, volume).unwrap();
        let recipe = crate::recipe::get_recipe("donchian_atr").unwrap();
        let genome = vec![10.0, 5.0, 14.0, 50.0, 1.2, 2.0, 0.0, 40.0];
        let mut bufs = SignalBuffers::default();
        build_donchian_atr_signals(&data, &recipe, &genome, &mut bufs);
        assert!(
            !bufs.entries.iter().any(|&e| e),
            "contracting vol with high atr_ratio_min must not enter"
        );
    }
}
