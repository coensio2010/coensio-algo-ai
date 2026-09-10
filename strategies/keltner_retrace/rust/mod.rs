// keltner_retrace: Keltner Channel band breakout + retrace-hold continuation.
//
// Structure: mid = EMA(close, ema_period); band = mid +/- kc_mult * ATR.
// Classic first-pierce KC chase is noisy. This strategy waits:
//   1) Arm on close-confirm break of PRIOR-bar upper/lower band
//      (optional RVOL + min break ATR)
//   2) Enter only when price retraces to the armed band within retest_window
//      AND close holds beyond that band (acceptance as new support/resistance)
// Never enter on the arm bar. Exit: ATR stop beyond armed band, measured-move
// (k x ATR from band), or time. Session boundary clears pending/positions.
//
// Related: ttm_squeeze (BB-inside-KC fire), zlema_retrace (ZLEMA mid),
// donchian_* / dual_thrust (range channels).

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
    build_keltner_retrace_signals(data, recipe, genome, spec, ctx, buffers);
}

fn build_keltner_retrace_signals(
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

    let ema_period = param_i32(recipe, genome, "ema_period", 20).clamp(8, 200) as usize;
    let kc_mult = param_f64(recipe, genome, "kc_mult", 1.5).clamp(0.5, 5.0);
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
    let ema = ema_series(&data.close, ema_period);
    let vol_sma = simple_sma(&data.volume, rvol_period);

    let mut in_pos = false;
    let mut pos_dir: i8 = 1;
    let mut entry_i = 0usize;
    let mut stop_level = f64::NAN;
    let mut target_level = f64::NAN;

    // Pending retest after KC band break (band frozen at arm)
    let mut pending_dir: i8 = 0;
    let mut pending_level = f64::NAN;
    let mut pending_at = 0usize;
    let mut pending_atr = f64::NAN;

    let mut cur_day = i32::MIN;
    let mut bars_in_session = 0usize;

    // Prior-bar bands need i>=1; warm for EMA + ATR + RVOL
    let warm = ema_period.max(rvol_period).max(atr_period) + 2;
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
        if i < warm || i < 1 || bars_in_session <= skip_bars {
            continue;
        }

        let a = atr[i];
        if !(a.is_finite() && a > 0.0) {
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
                    if c < pending_level {
                        pending_dir = 0;
                    } else {
                        let touched = l.is_finite() && l <= pending_level + tol;
                        let held = c > pending_level;
                        if touched && held {
                            buffers.entries[i] = true;
                            buffers.directions[i] = 1;
                            in_pos = true;
                            pos_dir = 1;
                            entry_i = i;
                            stop_level = pending_level - stop_mult * pending_atr;
                            target_level = pending_level + target_mult * pending_atr;
                            pending_dir = 0;
                            continue;
                        }
                    }
                } else if c > pending_level {
                    pending_dir = 0;
                } else {
                    let touched = h.is_finite() && h >= pending_level - tol;
                    let held = c < pending_level;
                    if touched && held {
                        buffers.entries[i] = true;
                        buffers.directions[i] = -1;
                        in_pos = true;
                        pos_dir = -1;
                        entry_i = i;
                        stop_level = pending_level + stop_mult * pending_atr;
                        target_level = pending_level - target_mult * pending_atr;
                        pending_dir = 0;
                        continue;
                    }
                }
            }
        }

        // Prior-bar Keltner bands (no look-ahead into close[i])
        let ema_prev = ema[i - 1];
        let atr_prev = atr[i - 1];
        if !(ema_prev.is_finite() && atr_prev.is_finite() && atr_prev > 0.0) {
            continue;
        }
        let upper_prev = ema_prev + kc_mult * atr_prev;
        let lower_prev = ema_prev - kc_mult * atr_prev;

        // Arm new KC band break (close-confirm + RVOL). Replaces prior pending.
        let vs = vol_sma[i];
        if !(vs.is_finite() && vs > 0.0) {
            continue;
        }
        let rvol = v / vs;
        if rvol < rvol_min {
            continue;
        }

        let break_buf = min_break_atr * a;
        if c > upper_prev + break_buf {
            pending_dir = 1;
            pending_level = upper_prev;
            pending_at = i;
            pending_atr = a;
        } else if allow_short && c < lower_prev - break_buf {
            pending_dir = -1;
            pending_level = lower_prev;
            pending_at = i;
            pending_atr = a;
        }
    }
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
