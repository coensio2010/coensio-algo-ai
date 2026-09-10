// donchian_chan: prior Donchian + prior EMA/slope + prior ADX/DI + Chandelier.
//
// Entry on bar i (close-confirm): prior Donchian break, close vs ema[i-1],
// ema slope vs ema[i-1-slope_bars], adx[i-1] >= adx_min, optional DI align.
// Exit: opposite prior exit Donchian OR Chandelier using ATR[i-1] OR time.
// No session flatten. Engine fills next open.

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
    build_donchian_chan_signals(data, recipe, genome, buffers);
}

fn build_donchian_chan_signals(
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

    let entry_len = param_i32(recipe, genome, "entry_len", 46).clamp(8, 80) as usize;
    let exit_len = param_i32(recipe, genome, "exit_len", 16).clamp(3, 40) as usize;
    let ema_len = param_i32(recipe, genome, "ema_len", 152).clamp(10, 250) as usize;
    let atr_period = param_i32(recipe, genome, "atr_period", 13).max(2) as usize;
    let trail_mult = param_f64(recipe, genome, "trail_mult", 3.0).clamp(0.8, 6.0);
    let adx_period = param_i32(recipe, genome, "adx_period", 14).clamp(2, 40) as usize;
    let adx_min = param_f64(recipe, genome, "adx_min", 18.0).clamp(0.0, 50.0);
    let di_align = param_i32(recipe, genome, "di_align", 1) > 0;
    let slope_bars = param_i32(recipe, genome, "slope_bars", 4).clamp(1, 30) as usize;
    let allow_short = param_i32(recipe, genome, "allow_short", 0) > 0;
    let exit_after_bars = param_i32(recipe, genome, "exit_after_bars", 90).max(1) as usize;

    let atr = wilder_atr(data, atr_period);
    let ema = ema_close(data, ema_len);
    let (adx, di_plus, di_minus) = wilder_adx(data, adx_period);
    let entry_hi = prior_donchian_high(data, entry_len);
    let entry_lo = prior_donchian_low(data, entry_len);
    let exit_hi = prior_donchian_high(data, exit_len);
    let exit_lo = prior_donchian_low(data, exit_len);

    let mut in_pos = false;
    let mut pos_dir: i8 = 1;
    let mut entry_i = 0usize;
    let mut run_ext = f64::NAN;

    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    let warm = entry_len
        .max(exit_len)
        .max(ema_len)
        .max(atr_period)
        .max(adx_period * 2)
        .max(slope_bars)
        + 3;
    for i in warm..n {
        let c = data.close[i];
        if !c.is_finite() {
            continue;
        }

        if in_pos {
            let held = i.saturating_sub(entry_i);
            let a = atr[i.saturating_sub(1)];
            if pos_dir > 0 {
                let h = data.high[i];
                if h.is_finite() {
                    run_ext = if run_ext.is_finite() { run_ext.max(h) } else { h };
                }
            } else {
                let l = data.low[i];
                if l.is_finite() {
                    run_ext = if run_ext.is_finite() { run_ext.min(l) } else { l };
                }
            }
            let chan_hit = a.is_finite()
                && a > 0.0
                && run_ext.is_finite()
                && if pos_dir > 0 {
                    c <= run_ext - trail_mult * a
                } else {
                    c >= run_ext + trail_mult * a
                };
            let don_exit = if pos_dir > 0 {
                exit_lo[i].is_finite() && c < exit_lo[i]
            } else {
                exit_hi[i].is_finite() && c > exit_hi[i]
            };
            if chan_hit || don_exit || held >= exit_after_bars {
                buffers.exits[i] = true;
                in_pos = false;
                continue;
            }
            continue;
        }

        let prev = i - 1;
        let eh = entry_hi[i];
        let el = entry_lo[i];
        let em = ema[prev];
        let slope_i = prev.saturating_sub(slope_bars);
        let em_prev = ema[slope_i];
        let ax = adx[prev];
        let pdi = di_plus[prev];
        let mdi = di_minus[prev];
        if !(em.is_finite() && em_prev.is_finite() && ax.is_finite()) {
            continue;
        }
        if ax < adx_min {
            continue;
        }

        let long_slope = em > em_prev;
        let short_slope = em < em_prev;
        let long_di = !di_align || (pdi.is_finite() && mdi.is_finite() && pdi > mdi);
        let short_di = !di_align || (pdi.is_finite() && mdi.is_finite() && mdi > pdi);

        let long_ok = eh.is_finite() && c > eh && c > em && long_slope && long_di;
        let short_ok = el.is_finite() && c < el && c < em && short_slope && short_di;

        if long_ok {
            buffers.entries[i] = true;
            buffers.directions[i] = 1;
            in_pos = true;
            pos_dir = 1;
            entry_i = i;
            run_ext = data.high[i];
        } else if allow_short && short_ok {
            buffers.entries[i] = true;
            buffers.directions[i] = -1;
            in_pos = true;
            pos_dir = -1;
            entry_i = i;
            run_ext = data.low[i];
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

fn ema_close(data: &MarketData, period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period < 1 || n < period {
        return out;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut acc = 0.0;
    let mut k = 0usize;
    for i in 0..period {
        let c = data.close[i];
        if c.is_finite() {
            acc += c;
            k += 1;
        }
    }
    if k == 0 {
        return out;
    }
    let mut ema = acc / k as f64;
    out[period - 1] = ema;
    for i in period..n {
        let c = data.close[i];
        if !c.is_finite() {
            continue;
        }
        ema = alpha * c + (1.0 - alpha) * ema;
        out[i] = ema;
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

fn wilder_adx(data: &MarketData, period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    let mut adx = vec![f64::NAN; n];
    let mut di_plus = vec![f64::NAN; n];
    let mut di_minus = vec![f64::NAN; n];
    if period < 2 || n < period + 1 {
        return (adx, di_plus, di_minus);
    }

    let mut tr = vec![0.0; n];
    let mut plus_dm = vec![0.0; n];
    let mut minus_dm = vec![0.0; n];
    tr[0] = (data.high[0] - data.low[0]).max(0.0);
    for i in 1..n {
        let h = data.high[i];
        let l = data.low[i];
        let pc = data.close[i - 1];
        let hl = h - l;
        let hc = (h - pc).abs();
        let lc = (l - pc).abs();
        tr[i] = hl.max(hc).max(lc);

        let up = h - data.high[i - 1];
        let down = data.low[i - 1] - l;
        plus_dm[i] = if up > down && up > 0.0 { up } else { 0.0 };
        minus_dm[i] = if down > up && down > 0.0 { down } else { 0.0 };
    }

    let mut str_ = 0.0;
    let mut s_plus = 0.0;
    let mut s_minus = 0.0;
    for i in 1..=period {
        str_ += tr[i];
        s_plus += plus_dm[i];
        s_minus += minus_dm[i];
    }

    let mut dx_sum = 0.0;
    let mut dx_count = 0usize;
    let mut first_adx_i = 0usize;

    for i in period..n {
        if i > period {
            str_ = str_ - str_ / period as f64 + tr[i];
            s_plus = s_plus - s_plus / period as f64 + plus_dm[i];
            s_minus = s_minus - s_minus / period as f64 + minus_dm[i];
        }

        if str_ > 0.0 {
            let pdi = 100.0 * s_plus / str_;
            let mdi = 100.0 * s_minus / str_;
            di_plus[i] = pdi;
            di_minus[i] = mdi;
            let denom = pdi + mdi;
            let dx = if denom > 0.0 {
                100.0 * (pdi - mdi).abs() / denom
            } else {
                0.0
            };

            if dx_count < period {
                dx_sum += dx;
                dx_count += 1;
                if dx_count == period {
                    adx[i] = dx_sum / period as f64;
                    first_adx_i = i;
                }
            } else if i > first_adx_i {
                let prev = adx[i - 1];
                if prev.is_finite() {
                    adx[i] = (prev * (period as f64 - 1.0) + dx) / period as f64;
                }
            }
        }
    }

    (adx, di_plus, di_minus)
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
