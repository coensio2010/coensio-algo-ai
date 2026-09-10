// orb_retrace: Session opening-range breakout + retrace-hold continuation.
//
// Structure: first orb_bars of each session freeze OR high/low.
// Classic ORB chase is noisy. This strategy waits:
//   1) Arm on close-confirm break of frozen OR (optional RVOL + ATR buffer)
//   2) Enter only when price retraces to the armed OR level within retest_window
//      AND close holds beyond that level (acceptance as new support/resistance)
// Never enter on the arm bar. Exit: ATR stop beyond armed level, measured-move
// (k x max(OR height, ATR)), or time. Session boundary clears pending/positions.
//
// Related: ib_retrace (initial balance), nr_expand (NR bar), dual_thrust.

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
    build_orb_retrace_signals(data, recipe, genome, spec, ctx, buffers);
}

fn build_orb_retrace_signals(
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

    let orb_bars = param_i32(recipe, genome, "orb_bars", 2).clamp(1, 6) as usize;
    let rvol_period = param_i32(recipe, genome, "rvol_period", 20).max(2) as usize;
    let rvol_min = param_f64(recipe, genome, "rvol_min", 1.0).clamp(0.5, 4.0);
    let retest_window = param_i32(recipe, genome, "retest_window", 8).clamp(2, 40) as usize;
    let retest_tol_atr = param_f64(recipe, genome, "retest_tol_atr", 0.25).clamp(0.01, 1.5);
    let allow_short = param_i32(recipe, genome, "allow_short", 1) > 0;
    let target_mult = param_f64(recipe, genome, "target_mult", 1.5).clamp(0.5, 5.0);
    let stop_mult = param_f64(recipe, genome, "stop_mult", 1.0).clamp(0.2, 4.0);
    let exit_after_bars = param_i32(recipe, genome, "exit_after_bars", 24).max(1) as usize;
    let atr_period = param_i32(recipe, genome, "atr_period", 14).max(2) as usize;
    let buffer_atr = param_f64(recipe, genome, "buffer_atr", 0.05).clamp(0.0, 1.0);
    let skip_bars = param_i32(recipe, genome, "skip_bars", 0).clamp(0, 4) as usize;

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

    // Session OR build + freeze
    let mut cur_day = i32::MIN;
    let mut bars_in_session = 0usize;
    let mut or_hi = f64::NAN;
    let mut or_lo = f64::NAN;
    let mut or_ready = false;

    // Pending retest after OR break (level frozen at arm)
    let mut pending_dir: i8 = 0;
    let mut pending_level = f64::NAN;
    let mut pending_at = 0usize;
    let mut pending_atr = f64::NAN;
    let mut pending_range = f64::NAN;

    let warm = rvol_period.max(atr_period) + 2;
    for i in 0..n {
        let day = session_days[i];
        if day != cur_day {
            cur_day = day;
            bars_in_session = 0;
            or_hi = f64::NAN;
            or_lo = f64::NAN;
            or_ready = false;
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

        // Build opening range from first orb_bars completed bars of the session.
        if bars_in_session <= orb_bars {
            or_hi = if or_hi.is_finite() { or_hi.max(h) } else { h };
            or_lo = if or_lo.is_finite() { or_lo.min(l) } else { l };
            if bars_in_session == orb_bars && or_hi.is_finite() && or_lo.is_finite() && or_hi >= or_lo
            {
                or_ready = true;
            }
            // No arm/enter while OR is still forming (includes freeze bar).
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

        if i < warm || !or_ready || bars_in_session <= orb_bars + skip_bars {
            continue;
        }

        let a = atr[i];
        if !(a.is_finite() && a > 0.0 && or_hi.is_finite() && or_lo.is_finite()) {
            continue;
        }
        let tol = retest_tol_atr * a;
        let or_range = (or_hi - or_lo).max(a);

        // Manage pending retest (never enter on arm bar itself).
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
                            let risk_unit = pending_range.max(pending_atr);
                            stop_level = pending_level - stop_mult * pending_atr;
                            target_level = pending_level + target_mult * risk_unit;
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
                        let risk_unit = pending_range.max(pending_atr);
                        stop_level = pending_level + stop_mult * pending_atr;
                        target_level = pending_level - target_mult * risk_unit;
                        pending_dir = 0;
                        continue;
                    }
                }
            }
        }

        // Arm new OR break (close-confirm + RVOL). Replaces prior pending.
        let vs = vol_sma[i];
        if !(vs.is_finite() && vs > 0.0) {
            continue;
        }
        let rvol = v / vs;
        if rvol < rvol_min {
            continue;
        }

        let buf = buffer_atr * a;
        if c > or_hi + buf {
            pending_dir = 1;
            pending_level = or_hi;
            pending_at = i;
            pending_atr = a;
            pending_range = or_range;
        } else if allow_short && c < or_lo - buf {
            pending_dir = -1;
            pending_level = or_lo;
            pending_at = i;
            pending_atr = a;
            pending_range = or_range;
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
