// poc_retrace: rolling Volume Profile POC breakout + retrace-hold.
//
// Structure: rolling OHLCV volume profile -> prior POC (+ VAH/VAL for stop/target).
// Classic first-pierce POC chase is noisy. This strategy waits:
//   1) Arm on close-confirm break of POC (optional RVOL + min VA width)
//   2) Enter only when price retraces to POC within retest_window AND close
//      holds beyond the broken POC (acceptance as new support/resistance)
// Exit: opposite VA edge stop (VAL long / VAH short), measured-move
//       (k x VA width from POC), or time.
//
// This is a continuation (break + retest + acceptance) setup, not a
// mean-reversion POC fade and not a first-pierce value-area breakout.

use crate::data::MarketData;
use crate::recipe::{PluginSpec, Recipe};
use crate::runtime::{BatchContext, SignalBuffers};

const N_BINS: usize = 24;
const VA_FRAC: f64 = 0.70;

pub fn build_signals(
    data: &MarketData,
    recipe: &Recipe,
    genome: &[f64],
    _spec: &PluginSpec,
    _ctx: &BatchContext,
    buffers: &mut SignalBuffers,
) {
    build_poc_retrace_signals(data, recipe, genome, buffers);
}

fn build_poc_retrace_signals(
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

    let profile_len = param_i32(recipe, genome, "profile_len", 40).clamp(12, 120) as usize;
    let rvol_period = param_i32(recipe, genome, "rvol_period", 20).max(2) as usize;
    let rvol_min = param_f64(recipe, genome, "rvol_min", 1.0).clamp(0.5, 4.0);
    let retest_window = param_i32(recipe, genome, "retest_window", 8).clamp(2, 40) as usize;
    let retest_tol_atr = param_f64(recipe, genome, "retest_tol_atr", 0.25).clamp(0.01, 1.5);
    let allow_short = param_i32(recipe, genome, "allow_short", 1) > 0;
    let target_mult = param_f64(recipe, genome, "target_mult", 1.5).clamp(0.5, 4.0);
    let exit_after_bars = param_i32(recipe, genome, "exit_after_bars", 24).max(1) as usize;
    let atr_period = param_i32(recipe, genome, "atr_period", 14).max(2) as usize;
    let min_va_atr = param_f64(recipe, genome, "min_va_atr", 0.3).clamp(0.0, 3.0);
    let min_break_atr = param_f64(recipe, genome, "min_break_atr", 0.1).clamp(0.0, 2.0);

    let atr = wilder_atr(data, atr_period);
    let vol_sma = simple_sma(&data.volume, rvol_period);
    let (vah, val, poc) = prior_value_area(data, profile_len);

    let mut in_pos = false;
    let mut pos_dir: i8 = 1;
    let mut entry_i = 0usize;
    let mut stop_level = f64::NAN;
    let mut target_level = f64::NAN;

    // Pending retest after POC break
    let mut pending_dir: i8 = 0;
    let mut pending_poc = f64::NAN;
    let mut pending_at = 0usize;
    let mut pending_stop = f64::NAN;
    let mut pending_width = f64::NAN;

    let warm = profile_len.max(rvol_period).max(atr_period) + 2;
    for i in warm..n {
        let h = data.high[i];
        let l = data.low[i];
        let c = data.close[i];
        let v = data.volume[i];
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

        let a = atr[i];
        let hi = vah[i];
        let lo = val[i];
        let pc = poc[i];
        if !(a.is_finite() && a > 0.0 && hi.is_finite() && lo.is_finite() && hi > lo && pc.is_finite())
        {
            continue;
        }
        let va_width = hi - lo;
        if min_va_atr > 0.0 && va_width < min_va_atr * a {
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
                    // Failed: close back below POC
                    if c < pending_poc {
                        pending_dir = 0;
                    } else {
                        let touched = l.is_finite() && l <= pending_poc + tol;
                        let held = c > pending_poc;
                        if touched && held {
                            buffers.entries[i] = true;
                            buffers.directions[i] = 1;
                            in_pos = true;
                            pos_dir = 1;
                            entry_i = i;
                            stop_level = pending_stop;
                            target_level = pending_poc + target_mult * pending_width;
                            pending_dir = 0;
                            continue;
                        }
                    }
                } else if c > pending_poc {
                    pending_dir = 0;
                } else {
                    let touched = h.is_finite() && h >= pending_poc - tol;
                    let held = c < pending_poc;
                    if touched && held {
                        buffers.entries[i] = true;
                        buffers.directions[i] = -1;
                        in_pos = true;
                        pos_dir = -1;
                        entry_i = i;
                        stop_level = pending_stop;
                        target_level = pending_poc - target_mult * pending_width;
                        pending_dir = 0;
                        continue;
                    }
                }
            }
        }

        // Arm new POC break (close-confirm + RVOL). Replaces prior pending.
        if !(v.is_finite() && v > 0.0) {
            continue;
        }
        let vs = vol_sma[i];
        if !(vs.is_finite() && vs > 0.0) {
            continue;
        }
        let rvol = v / vs;
        if rvol < rvol_min {
            continue;
        }

        let break_buf = min_break_atr * a;
        if c > pc + break_buf {
            pending_dir = 1;
            pending_poc = pc;
            pending_at = i;
            pending_stop = lo;
            pending_width = va_width;
        } else if allow_short && c < pc - break_buf {
            pending_dir = -1;
            pending_poc = pc;
            pending_at = i;
            pending_stop = hi;
            pending_width = va_width;
        }
    }
}

/// Prior-bar rolling volume profile -> VAH / VAL / POC mid (no lookahead).
fn prior_value_area(data: &MarketData, profile_len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    let mut vah = vec![f64::NAN; n];
    let mut val = vec![f64::NAN; n];
    let mut poc_px = vec![f64::NAN; n];
    if profile_len < 4 || n <= profile_len {
        return (vah, val, poc_px);
    }

    let mut bins = [0.0f64; N_BINS];
    for i in profile_len..n {
        let start = i - profile_len;
        let mut p_lo = f64::INFINITY;
        let mut p_hi = f64::NEG_INFINITY;
        let mut any = false;
        for j in start..i {
            let h = data.high[j];
            let l = data.low[j];
            let v = data.volume[j];
            if h.is_finite() && l.is_finite() && v.is_finite() && v > 0.0 && h >= l {
                p_lo = p_lo.min(l);
                p_hi = p_hi.max(h);
                any = true;
            }
        }
        if !any || !(p_hi > p_lo) {
            continue;
        }
        let width = p_hi - p_lo;
        bins.fill(0.0);
        let mut total = 0.0;
        for j in start..i {
            let h = data.high[j];
            let l = data.low[j];
            let c = data.close[j];
            let v = data.volume[j];
            if !(h.is_finite() && l.is_finite() && c.is_finite() && v.is_finite() && v > 0.0) {
                continue;
            }
            let tp = (h + l + c) / 3.0;
            let mut bi = ((tp - p_lo) / width * N_BINS as f64).floor() as isize;
            if bi < 0 {
                bi = 0;
            }
            if bi >= N_BINS as isize {
                bi = (N_BINS - 1) as isize;
            }
            bins[bi as usize] += v;
            total += v;
        }
        if total <= 0.0 {
            continue;
        }
        let mut poc = 0usize;
        let mut best = bins[0];
        for (b, &vol) in bins.iter().enumerate().skip(1) {
            if vol > best {
                best = vol;
                poc = b;
            }
        }
        let target = total * VA_FRAC;
        let mut left = poc;
        let mut right = poc;
        let mut cum = bins[poc];
        while cum < target && (left > 0 || right + 1 < N_BINS) {
            let left_vol = if left > 0 { bins[left - 1] } else { -1.0 };
            let right_vol = if right + 1 < N_BINS {
                bins[right + 1]
            } else {
                -1.0
            };
            if right_vol >= left_vol {
                right += 1;
                cum += bins[right];
            } else {
                left -= 1;
                cum += bins[left];
            }
        }
        let bin_w = width / N_BINS as f64;
        val[i] = p_lo + left as f64 * bin_w;
        vah[i] = p_lo + (right as f64 + 1.0) * bin_w;
        poc_px[i] = p_lo + (poc as f64 + 0.5) * bin_w;
    }
    (vah, val, poc_px)
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

    fn trending_up() -> MarketData {
        let n = 400usize;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 * 3_600_000.0).collect();
        let mut open = vec![100.0; n];
        let mut high = vec![100.5; n];
        let mut low = vec![99.5; n];
        let mut close = vec![100.0; n];
        let mut volume = vec![1_000.0; n];
        for i in 0..n {
            let px = 100.0 + (i as f64) * 0.4;
            open[i] = px - 0.2;
            close[i] = px;
            high[i] = px + 0.6;
            low[i] = px - 0.5;
            volume[i] = 800.0 + (i % 7) as f64 * 50.0;
        }
        // Cluster volume so POC sits mid, then break above and retest
        for i in 40..70 {
            let px = 120.0;
            open[i] = px;
            close[i] = px + 0.1;
            high[i] = px + 0.3;
            low[i] = px - 0.2;
            volume[i] = 2_000.0;
        }
        for i in 70..80 {
            let px = 120.5 + (i - 70) as f64 * 0.05;
            open[i] = px;
            close[i] = px + 0.1;
            high[i] = px + 0.4;
            low[i] = px - 0.1;
            volume[i] = 1_500.0;
        }
        let i_arm = 80usize;
        let i_re = 82usize;
        open[i_arm] = 121.0;
        high[i_arm] = 124.0;
        low[i_arm] = 120.8;
        close[i_arm] = 123.5;
        volume[i_arm] = 8_000.0;
        open[i_arm + 1] = 123.0;
        high[i_arm + 1] = 124.0;
        low[i_arm + 1] = 122.5;
        close[i_arm + 1] = 123.2;
        volume[i_arm + 1] = 2_000.0;
        open[i_re] = 122.5;
        high[i_re] = 123.0;
        low[i_re] = 119.5;
        close[i_re] = 122.0;
        volume[i_re] = 3_000.0;
        MarketData::new(ts, open, high, low, close, volume).unwrap()
    }

    #[test]
    fn poc_retrace_can_enter_long() {
        let data = trending_up();
        let recipe = crate::recipe::get_recipe("poc_retrace").unwrap();
        // profile,rvol_p,rvol_min,retest_w,tol,short,tgt,exit,atr,min_va,min_break
        let genome = vec![
            30.0, 5.0, 0.8, 10.0, 0.5, 1.0, 1.5, 40.0, 10.0, 0.1, 0.05,
        ];
        let mut bufs = SignalBuffers::default();
        build_poc_retrace_signals(&data, &recipe, &genome, &mut bufs);
        assert!(
            bufs.entries.iter().any(|&e| e),
            "expected at least one POC+retrace long entry"
        );
    }

    #[test]
    fn flat_series_no_entry() {
        let n = 80usize;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 * 3_600_000.0).collect();
        let close = vec![100.0; n];
        let high = vec![100.1; n];
        let low = vec![99.9; n];
        let open = close.clone();
        let volume = vec![1_000.0; n];
        let data = MarketData::new(ts, open, high, low, close, volume).unwrap();
        let recipe = crate::recipe::get_recipe("poc_retrace").unwrap();
        let genome = vec![
            40.0, 20.0, 1.5, 8.0, 0.25, 1.0, 1.5, 24.0, 14.0, 0.3, 0.1,
        ];
        let mut bufs = SignalBuffers::default();
        build_poc_retrace_signals(&data, &recipe, &genome, &mut bufs);
        assert!(!bufs.entries.iter().any(|&e| e));
    }
}
