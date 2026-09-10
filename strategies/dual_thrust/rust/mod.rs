// dual_thrust: open-anchored Dual Thrust breakout + ATR trail + range modes.
// Prior-N range (bars [i-N .. i-1], no look-ahead):
//   mode 0: max(HH-LC, HC-LL)  (classic Toshchakov)
//   mode 1: HH - LL
//   mode 2: HC - LC
// BuyLine  = open[i] + k_buy  * Range + open * buffer_bps/1e4
// SellLine = open[i] - k_sell * Range - open * buffer_bps/1e4
// Entry: close breaks line (+ SMA filter). Exit: flip, ATR trail (if mult>0), or time.

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
    build_dual_thrust_signals(data, recipe, genome, buffers);
}

fn build_dual_thrust_signals(
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

    let range_period = param_i32(recipe, genome, "range_period", 4).max(1) as usize;
    let k_buy = param_f64(recipe, genome, "k_buy", 0.50).max(0.01);
    let k_sell = param_f64(recipe, genome, "k_sell", 0.50).max(0.01);
    let sma_filter = param_i32(recipe, genome, "sma_filter", 0).max(0) as usize;
    let range_mode = param_i32(recipe, genome, "range_mode", 0).clamp(0, 2);
    let buffer_bps = param_f64(recipe, genome, "buffer_bps", 0.0).max(0.0);
    let atr_period = param_i32(recipe, genome, "atr_period", 14).max(2) as usize;
    let atr_trail_mult = param_f64(recipe, genome, "atr_trail_mult", 0.0).max(0.0);
    let allow_short = true;
    let exit_after_bars = param_i32(recipe, genome, "exit_after_bars", 10).max(1) as usize;

    let sma = if sma_filter >= 2 {
        simple_sma(&data.close, sma_filter)
    } else {
        vec![f64::NAN; n]
    };
    let atr = if atr_trail_mult > 0.0 {
        wilder_atr(data, atr_period)
    } else {
        vec![f64::NAN; n]
    };

    let mut in_pos = false;
    let mut pos_dir: i8 = 1;
    let mut entry_i = 0usize;
    let mut trail = f64::NAN;
    let mut extreme = f64::NAN;

    for i in 0..n {
        let range = prior_dual_range(data, i, range_period, range_mode);
        let o = data.open[i];
        let c = data.close[i];
        let buf = if o.is_finite() {
            o * buffer_bps / 10_000.0
        } else {
            0.0
        };
        let buy_line = if range.is_finite() && o.is_finite() {
            o + k_buy * range + buf
        } else {
            f64::NAN
        };
        let sell_line = if range.is_finite() && o.is_finite() {
            o - k_sell * range - buf
        } else {
            f64::NAN
        };

        let long_trig =
            c.is_finite() && buy_line.is_finite() && c > buy_line && sma_ok_long(i, c, &sma, sma_filter);
        let short_trig = allow_short
            && c.is_finite()
            && sell_line.is_finite()
            && c < sell_line
            && sma_ok_short(i, c, &sma, sma_filter);

        if in_pos {
            let held = i.saturating_sub(entry_i);
            let a = atr[i];

            // Ratcheting ATR trail (chandelier-style from trade extreme).
            if atr_trail_mult > 0.0 && a.is_finite() {
                if pos_dir > 0 {
                    if data.high[i].is_finite() {
                        extreme = if extreme.is_finite() {
                            extreme.max(data.high[i])
                        } else {
                            data.high[i]
                        };
                    }
                    if extreme.is_finite() {
                        let cand = extreme - atr_trail_mult * a;
                        trail = if trail.is_finite() {
                            trail.max(cand)
                        } else {
                            cand
                        };
                    }
                } else if data.low[i].is_finite() {
                    extreme = if extreme.is_finite() {
                        extreme.min(data.low[i])
                    } else {
                        data.low[i]
                    };
                    if extreme.is_finite() {
                        let cand = extreme + atr_trail_mult * a;
                        trail = if trail.is_finite() {
                            trail.min(cand)
                        } else {
                            cand
                        };
                    }
                }
            }

            let trail_hit = if atr_trail_mult > 0.0 && c.is_finite() && trail.is_finite() {
                if pos_dir > 0 {
                    c <= trail
                } else {
                    c >= trail
                }
            } else {
                false
            };
            let flip = if pos_dir > 0 {
                short_trig
            } else {
                long_trig
            };
            if flip || trail_hit || held >= exit_after_bars {
                buffers.exits[i] = true;
                in_pos = false;
                trail = f64::NAN;
                extreme = f64::NAN;
                continue;
            }
        }

        if !in_pos {
            if long_trig {
                buffers.entries[i] = true;
                buffers.directions[i] = 1;
                in_pos = true;
                pos_dir = 1;
                entry_i = i;
                extreme = data.high[i];
                trail = if atr_trail_mult > 0.0 && atr[i].is_finite() && extreme.is_finite() {
                    extreme - atr_trail_mult * atr[i]
                } else {
                    f64::NAN
                };
            } else if short_trig {
                buffers.entries[i] = true;
                buffers.directions[i] = -1;
                in_pos = true;
                pos_dir = -1;
                entry_i = i;
                extreme = data.low[i];
                trail = if atr_trail_mult > 0.0 && atr[i].is_finite() && extreme.is_finite() {
                    extreme + atr_trail_mult * atr[i]
                } else {
                    f64::NAN
                };
            }
        }
    }
}

fn prior_dual_range(data: &MarketData, i: usize, period: usize, mode: i32) -> f64 {
    if i < period || period == 0 {
        return f64::NAN;
    }
    let start = i - period;
    let mut hh = f64::NEG_INFINITY;
    let mut ll = f64::INFINITY;
    let mut hc = f64::NEG_INFINITY;
    let mut lc = f64::INFINITY;
    let mut ok = false;
    for j in start..i {
        let h = data.high[j];
        let l = data.low[j];
        let c = data.close[j];
        if h.is_finite() && l.is_finite() && c.is_finite() {
            hh = hh.max(h);
            ll = ll.min(l);
            hc = hc.max(c);
            lc = lc.min(c);
            ok = true;
        }
    }
    if !ok {
        return f64::NAN;
    }
    let r = match mode {
        1 => hh - ll,
        2 => hc - lc,
        _ => (hh - lc).max(hc - ll),
    };
    r.max(0.0)
}

fn sma_ok_long(i: usize, close: f64, sma: &[f64], period: usize) -> bool {
    if period < 2 {
        return true;
    }
    let m = sma[i];
    close.is_finite() && m.is_finite() && close >= m
}

fn sma_ok_short(i: usize, close: f64, sma: &[f64], period: usize) -> bool {
    if period < 2 {
        return true;
    }
    let m = sma[i];
    close.is_finite() && m.is_finite() && close <= m
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
    if period < 2 || n < period {
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

    #[test]
    fn range_modes_differ() {
        let n = 10usize;
        let ts: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let close = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0];
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open = close.clone();
        let volume = vec![1.0; n];
        let data = MarketData::new(ts, open, high, low, close, volume).unwrap();
        let r0 = prior_dual_range(&data, 5, 4, 0);
        let r1 = prior_dual_range(&data, 5, 4, 1);
        let r2 = prior_dual_range(&data, 5, 4, 2);
        assert!(r0.is_finite() && r1.is_finite() && r2.is_finite());
        assert!(r1 >= r2 - 1e-9);
    }

    #[test]
    fn buy_break_triggers_long() {
        let n = 30usize;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 * 86_400_000.0).collect();
        let mut close = vec![100.0; n];
        let mut high = vec![101.0; n];
        let mut low = vec![99.0; n];
        let mut open = vec![100.0; n];
        for i in 0..20 {
            close[i] = 100.0;
            high[i] = 101.0;
            low[i] = 99.0;
            open[i] = 100.0;
        }
        open[21] = 100.0;
        high[21] = 108.0;
        low[21] = 99.5;
        close[21] = 107.0;
        let volume = vec![1_000.0; n];
        let data = MarketData::new(ts, open, high, low, close, volume).unwrap();
        let recipe = crate::recipe::get_recipe("dual_thrust").unwrap();
        // range,k_buy,k_sell,sma,mode,buf,atr,trail,exit
        let genome = vec![4.0, 0.5, 0.5, 0.0, 0.0, 0.0, 14.0, 0.0, 10.0];
        let mut bufs = SignalBuffers::default();
        build_dual_thrust_signals(&data, &recipe, &genome, &mut bufs);
        assert!(bufs.entries.iter().any(|&e| e));
    }
}
