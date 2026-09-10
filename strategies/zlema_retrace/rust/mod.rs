// zlema_retrace: Ehlers/Way Zero-Lag EMA breakout + retrace-hold continuation.
//
// Structure: lag = (period-1)/2; ZLEMA = EMA(2*close - close[lag], period).
// Classic first-pierce ZLEMA chase is noisy. This strategy waits:
//   1) Arm on close-confirm break of ZLEMA (optional RVOL + min break ATR)
//   2) Enter only when price retraces to the armed ZLEMA within retest_window
//      AND close holds beyond that ZLEMA (acceptance as new support/resistance)
// Exit: ATR stop beyond armed ZLEMA, measured-move (k x ATR from ZLEMA), or time.
// Session boundary clears pending and positions.

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
    build_zlema_retrace_signals(data, recipe, genome, spec, ctx, buffers);
}

fn build_zlema_retrace_signals(
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

    let zlema_period = param_i32(recipe, genome, "zlema_period", 40).clamp(8, 200) as usize;
    let rvol_period = param_i32(recipe, genome, "rvol_period", 20).max(2) as usize;
    let rvol_min = param_f64(recipe, genome, "rvol_min", 1.0).clamp(0.5, 4.0);
    let retest_window = param_i32(recipe, genome, "retest_window", 8).clamp(2, 40) as usize;
    let retest_tol_atr = param_f64(recipe, genome, "retest_tol_atr", 0.25).clamp(0.01, 1.5);
    let allow_short = param_i32(recipe, genome, "allow_short", 1) > 0;
    let target_mult = param_f64(recipe, genome, "target_mult", 1.5).clamp(0.5, 4.0);
    let stop_mult = param_f64(recipe, genome, "stop_mult", 1.0).clamp(0.2, 4.0);
    let exit_after_bars = param_i32(recipe, genome, "exit_after_bars", 24).max(1) as usize;
    let atr_period = param_i32(recipe, genome, "atr_period", 14).max(2) as usize;
    let min_break_atr = param_f64(recipe, genome, "min_break_atr", 0.1).clamp(0.0, 2.0);
    let skip_bars = param_i32(recipe, genome, "skip_bars", 2).clamp(0, 8) as usize;

    let hour = spec.session_utc_start.first().copied().unwrap_or(14);
    let minute = spec.session_utc_start.get(1).copied().unwrap_or(30);
    let session_days: std::sync::Arc<Vec<i32>> = ctx.session_days.clone().unwrap_or_else(|| {
        std::sync::Arc::new(crate::session::session_days_from_market(data, hour, minute))
    });

    let atr = wilder_atr(data, atr_period);
    let zlema = zlema_close(&data.close, zlema_period);
    let vol_sma = simple_sma(&data.volume, rvol_period);

    let mut in_pos = false;
    let mut pos_dir: i8 = 1;
    let mut entry_i = 0usize;
    let mut stop_level = f64::NAN;
    let mut target_level = f64::NAN;

    // Pending retest after ZLEMA break (ZLEMA frozen at arm)
    let mut pending_dir: i8 = 0;
    let mut pending_zlema = f64::NAN;
    let mut pending_at = 0usize;
    let mut pending_atr = f64::NAN;

    let mut cur_day = i32::MIN;
    let mut bars_in_session = 0usize;

    // ZLEMA needs period + lag for de-lagged EMA warm
    let lag = zlema_period.saturating_sub(1) / 2;
    let warm = (zlema_period + lag).max(rvol_period).max(atr_period) + 2;
    for i in 0..n {
        let day = session_days[i];
        if day != cur_day {
            cur_day = day;
            bars_in_session = 0;
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

        let zlema_i = zlema[i];

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
        }

        if in_pos {
            continue;
        }
        if i < warm || bars_in_session <= skip_bars {
            continue;
        }

        let a = atr[i];
        if !(a.is_finite() && a > 0.0 && zlema_i.is_finite()) {
            continue;
        }
        let tol = retest_tol_atr * a;

        // Manage pending retest (never enter on arm bar itself)
        if pending_dir != 0 {
            let waited = i.saturating_sub(pending_at);
            if waited > retest_window {
                pending_dir = 0;
            } else if waited >= 1 {
                if pending_dir > 0 {
                    if c < pending_zlema {
                        pending_dir = 0;
                    } else {
                        let touched = l.is_finite() && l <= pending_zlema + tol;
                        let held = c > pending_zlema;
                        if touched && held {
                            buffers.entries[i] = true;
                            buffers.directions[i] = 1;
                            in_pos = true;
                            pos_dir = 1;
                            entry_i = i;
                            stop_level = pending_zlema - stop_mult * pending_atr;
                            target_level = pending_zlema + target_mult * pending_atr;
                            pending_dir = 0;
                            continue;
                        }
                    }
                } else if c > pending_zlema {
                    pending_dir = 0;
                } else {
                    let touched = h.is_finite() && h >= pending_zlema - tol;
                    let held = c < pending_zlema;
                    if touched && held {
                        buffers.entries[i] = true;
                        buffers.directions[i] = -1;
                        in_pos = true;
                        pos_dir = -1;
                        entry_i = i;
                        stop_level = pending_zlema + stop_mult * pending_atr;
                        target_level = pending_zlema - target_mult * pending_atr;
                        pending_dir = 0;
                        continue;
                    }
                }
            }
        }

        // Arm new ZLEMA break (close-confirm + RVOL). Replaces prior pending.
        let vs = vol_sma[i];
        if !(vs.is_finite() && vs > 0.0) {
            continue;
        }
        let rvol = v / vs;
        if rvol < rvol_min {
            continue;
        }

        let break_buf = min_break_atr * a;
        if c > zlema_i + break_buf {
            pending_dir = 1;
            pending_zlema = zlema_i;
            pending_at = i;
            pending_atr = a;
        } else if allow_short && c < zlema_i - break_buf {
            pending_dir = -1;
            pending_zlema = zlema_i;
            pending_at = i;
            pending_atr = a;
        }
    }
}

/// John Ehlers / Ric Way ZLEMA: EMA of de-lagged closes.
/// lag = (period-1)/2; ema_data = 2*close - close[lag]; ZLEMA = EMA(ema_data, period).
fn zlema_close(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    if period < 1 || n < period {
        return vec![f64::NAN; n];
    }
    let lag = period.saturating_sub(1) / 2;
    let mut ema_data = vec![f64::NAN; n];
    for i in 0..n {
        if !values[i].is_finite() {
            continue;
        }
        if lag == 0 || i < lag {
            // Before lag available: use raw close (no de-lag yet)
            ema_data[i] = values[i];
        } else if values[i - lag].is_finite() {
            ema_data[i] = 2.0 * values[i] - values[i - lag];
        }
    }
    ema_series(&ema_data, period)
}

/// EMA over a series that may contain leading NaNs.
fn ema_series(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![f64::NAN; n];
    if period < 1 || n < period {
        return out;
    }
    let alpha = 2.0 / (period as f64 + 1.0);

    let mut start = None;
    let mut run = 0usize;
    for i in 0..n {
        if values[i].is_finite() {
            run += 1;
            if run >= period {
                start = Some(i + 1 - period);
                break;
            }
        } else {
            run = 0;
        }
    }
    let Some(seed_start) = start else {
        return out;
    };
    let seed_end = seed_start + period - 1;
    let mut sum = 0.0;
    for i in seed_start..=seed_end {
        sum += values[i];
    }
    let mut ema_val = sum / period as f64;
    out[seed_end] = ema_val;
    for i in seed_end + 1..n {
        let v = values[i];
        if !v.is_finite() {
            continue;
        }
        ema_val = alpha * v + (1.0 - alpha) * ema_val;
        out[i] = ema_val;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn break_retest_data() -> MarketData {
        // Flat then break above ZLEMA, then retest hold.
        let n = 160usize;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 * 3_600_000.0).collect();
        let mut open = vec![100.0; n];
        let mut high = vec![100.5; n];
        let mut low = vec![99.5; n];
        let mut close = vec![100.0; n];
        let mut volume = vec![1_000.0; n];
        let days: Vec<i32> = vec![1; n];

        for i in 0..120 {
            open[i] = 100.0;
            high[i] = 100.4;
            low[i] = 99.6;
            close[i] = 100.0;
            volume[i] = 1_000.0;
        }
        let i_arm = 130usize;
        open[i_arm] = 100.5;
        high[i_arm] = 104.0;
        low[i_arm] = 100.4;
        close[i_arm] = 103.5;
        volume[i_arm] = 5_000.0;
        open[i_arm + 1] = 103.0;
        high[i_arm + 1] = 103.8;
        low[i_arm + 1] = 102.5;
        close[i_arm + 1] = 103.2;
        volume[i_arm + 1] = 1_500.0;
        let i_re = 132usize;
        open[i_re] = 101.5;
        high[i_re] = 101.8;
        low[i_re] = 99.8;
        close[i_re] = 101.2;
        volume[i_re] = 2_000.0;
        for i in 133..n {
            open[i] = 101.0 + (i - 133) as f64 * 0.1;
            close[i] = open[i] + 0.2;
            high[i] = close[i] + 0.2;
            low[i] = open[i] - 0.1;
            volume[i] = 1_200.0;
        }

        let mut data = MarketData::new(ts, open, high, low, close, volume).unwrap();
        data.set_session_days(days).unwrap();
        data
    }

    #[test]
    fn zlema_retrace_can_enter_long() {
        let data = break_retest_data();
        let recipe = crate::recipe::get_recipe("zlema_retrace").unwrap();
        let genome = vec![
            16.0, 5.0, 0.8, 10.0, 0.5, 1.0, 1.5, 1.0, 40.0, 10.0, 0.05, 1.0,
        ];
        let mut bufs = SignalBuffers::default();
        let spec = recipe.plugin.as_ref().unwrap();
        let ctx = BatchContext::for_recipe(&data, &recipe);
        build_zlema_retrace_signals(&data, &recipe, &genome, spec, &ctx, &mut bufs);
        assert!(
            bufs.entries.iter().any(|&e| e),
            "expected at least one ZLEMA+retrace long entry"
        );
    }

    #[test]
    fn flat_series_no_entry() {
        let n = 160usize;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 * 3_600_000.0).collect();
        let close = vec![100.0; n];
        let high = vec![100.1; n];
        let low = vec![99.9; n];
        let open = close.clone();
        let volume = vec![1_000.0; n];
        let days: Vec<i32> = (0..n).map(|i| (i / 8) as i32).collect();
        let mut data = MarketData::new(ts, open, high, low, close, volume).unwrap();
        data.set_session_days(days).unwrap();
        let recipe = crate::recipe::get_recipe("zlema_retrace").unwrap();
        let genome = vec![
            40.0, 20.0, 1.5, 8.0, 0.25, 1.0, 1.5, 1.0, 24.0, 14.0, 0.1, 2.0,
        ];
        let mut bufs = SignalBuffers::default();
        let spec = recipe.plugin.as_ref().unwrap();
        let ctx = BatchContext::for_recipe(&data, &recipe);
        build_zlema_retrace_signals(&data, &recipe, &genome, spec, &ctx, &mut bufs);
        assert!(!bufs.entries.iter().any(|&e| e));
    }
}
