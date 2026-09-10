mod BS1_core {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../strategies/BS1_breakout/rust/BS1_core.rs"
    ));
}
mod indicators {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../strategies/BS1_breakout/rust/indicators.rs"
    ));
}

use self::indicators::{rolling_max, rolling_min, shift};
use self::BS1_core as bs1;
use crate::data::MarketData;
use crate::recipe::{PluginSpec, Recipe};
use crate::runtime::{update_signal_position, BatchContext, SignalBuffers};

#[derive(Debug, Clone, Copy)]
pub struct BS1PluginConfig {
    pub direction: i8,
    pub exit_method: BS1ExitMethod,
    pub session_utc_start_hour: i32,
    pub session_utc_start_min: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BS1ExitMethod {
    OppositeDonchian,
    ChandelierAtr,
    TimeOnly,
    /// Exit when close reclaims the POI anchor (through poi back toward entry).
    ReclaimPoi,
}

pub fn config_from_spec(spec: &PluginSpec) -> BS1PluginConfig {
    let direction = if spec.direction.eq_ignore_ascii_case("short") {
        -1
    } else {
        1
    };
    let exit_method = match spec.exit_method.as_str() {
        "chandelier_atr" => BS1ExitMethod::ChandelierAtr,
        "time_only" => BS1ExitMethod::TimeOnly,
        "reclaim_poi" => BS1ExitMethod::ReclaimPoi,
        _ => BS1ExitMethod::OppositeDonchian,
    };
    let start = &spec.session_utc_start;
    BS1PluginConfig {
        direction,
        exit_method,
        session_utc_start_hour: start.first().copied().unwrap_or(0),
        session_utc_start_min: start.get(1).copied().unwrap_or(0),
    }
}

pub fn build_signals(
    data: &MarketData,
    recipe: &Recipe,
    genome: &[f64],
    spec: &PluginSpec,
    ctx: &BatchContext,
    buffers: &mut SignalBuffers,
) {
    let cfg = config_from_spec(spec);
    build_BS1_breakout_signals(data, recipe, genome, &cfg, ctx, buffers);
}

fn build_BS1_breakout_signals(
    data: &MarketData,
    recipe: &Recipe,
    genome: &[f64],
    plugin: &BS1PluginConfig,
    ctx: &BatchContext,
    buffers: &mut SignalBuffers,
) {
    let n = data.len();
    buffers.resize(n);
    buffers.entries.fill(false);
    buffers.exits.fill(false);
    buffers.directions.fill(plugin.direction);

    let poi_switch = param_i32(recipe, genome, "poi_method", 3);
    let poi_n1 = param_i32(recipe, genome, "donchian_period", 2);
    let fract = param_f64(recipe, genome, "breakout_atr_mult", 1.3);
    let natr = param_i32(recipe, genome, "breakout_atr_period", 35);
    let f1_sw = param_i32(recipe, genome, "filter1_method", 0);
    let f1_n1 = param_i32(recipe, genome, "filter1_param1", 8);
    let f1_n2 = param_i32(recipe, genome, "filter1_param2", 16);
    let f2_sw = param_i32(recipe, genome, "filter2_method", 0);
    let f2_n1 = param_i32(recipe, genome, "filter2_param1", 1);
    let f2_n2 = param_i32(recipe, genome, "filter2_param2", 1);

    let session_days: std::sync::Arc<Vec<i32>> = ctx.session_days.clone().unwrap_or_else(|| {
        std::sync::Arc::new(crate::session::session_days_from_market(
            data,
            plugin.session_utc_start_hour,
            plugin.session_utc_start_min,
        ))
    });

    let (long_sig, short_sig, poi_l, poi_s) = compute_BS1_entries(
        data,
        session_days.as_slice(),
        poi_switch,
        poi_n1,
        fract,
        natr,
        f1_sw,
        f1_n1,
        f1_n2,
        f2_sw,
        f2_n1,
        f2_n2,
    );

    for i in 0..n {
        buffers.entries[i] = match plugin.direction {
            -1 => short_sig[i],
            _ => long_sig[i],
        };
    }

    let donchian_period = poi_n1.max(2) as usize;
    match plugin.exit_method {
        BS1ExitMethod::OppositeDonchian => {
            build_opposite_donchian_exits(
                data,
                &buffers.entries,
                donchian_period,
                plugin.direction,
                &mut buffers.exits,
            );
        }
        BS1ExitMethod::ChandelierAtr => {
            let period = param_i32(recipe, genome, "breakout_atr_period", 14).max(2) as usize;
            let mult = param_f64(recipe, genome, "breakout_atr_mult", 3.0);
            build_chandelier_exits(
                data,
                &buffers.entries,
                plugin.direction,
                period,
                mult,
                &mut buffers.exits,
            );
        }
        BS1ExitMethod::TimeOnly => {
            let max_bars = param_i32(recipe, genome, "filter2_param1", 10).max(1) as usize;
            build_time_exits(&buffers.entries, max_bars, &mut buffers.exits);
        }
        BS1ExitMethod::ReclaimPoi => {
            let poi = if plugin.direction < 0 { &poi_s } else { &poi_l };
            build_reclaim_poi_exits(
                data,
                &buffers.entries,
                poi,
                plugin.direction,
                &mut buffers.exits,
            );
        }
    }
}

fn compute_BS1_entries(
    data: &MarketData,
    session_days: &[i32],
    poi_switch: i32,
    poi_n1: i32,
    fract: f64,
    natr: i32,
    f1_sw: i32,
    f1_n1: i32,
    f1_n2: i32,
    f2_sw: i32,
    f2_n1: i32,
    f2_n2: i32,
) -> (Vec<bool>, Vec<bool>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    let open = &data.open;
    let high = &data.high;
    let low = &data.low;
    let close = &data.close;

    let (poi_l, poi_s) = bs1::compute_poi(open, high, low, close, session_days, poi_switch, poi_n1);
    let atr_vals = self::indicators::atr(high, low, close, natr.max(2) as usize);
    let atr_shifted = shift(&atr_vals, 1);

    let mut bo_long = vec![0.0f64; n];
    let mut bo_short = vec![0.0f64; n];
    for i in 0..n {
        if poi_l[i].is_finite() && atr_shifted[i].is_finite() {
            bo_long[i] = poi_l[i] + fract * atr_shifted[i];
        }
        if poi_s[i].is_finite() && atr_shifted[i].is_finite() {
            bo_short[i] = poi_s[i] - fract * atr_shifted[i];
        }
    }

    let bs1::BS1Filters {
        long: f1l,
        short: f1s,
    } = bs1::compute_filters(open, high, low, close, session_days, f1_sw, f1_n1, f1_n2);

    let (f2l, f2s) = if f2_sw > 0 {
        let htf = bs1::compute_filters(open, high, low, close, session_days, f2_sw, f2_n1, f2_n2);
        let bias = 4usize;
        let mut shifted_l = vec![true; n];
        let mut shifted_s = vec![true; n];
        for i in 0..n {
            let src = if i >= bias { i - bias } else { i };
            shifted_l[i] = htf.long[src];
            shifted_s[i] = htf.short[src];
        }
        (shifted_l, shifted_s)
    } else {
        (vec![true; n], vec![true; n])
    };

    // Close-confirmed breakout with nearest-side preference when both sides fire.
    // (Intrabar high/low touch variants are intentionally not used: they leak
    // the bar path into the signal and are not reproducible in forward mode.)
    let mut long_sig = vec![false; n];
    let mut short_sig = vec![false; n];
    for i in 0..n {
        if bo_long[i] == 0.0 || bo_short[i] == 0.0 {
            continue;
        }
        let diff1 = (close[i] - bo_long[i]).abs();
        let diff2 = (close[i] - bo_short[i]).abs();
        let prefer_long = diff1 <= diff2;
        long_sig[i] = close[i] >= bo_long[i] && f1l[i] && f2l[i] && prefer_long;
        short_sig[i] = close[i] <= bo_short[i] && f1s[i] && f2s[i] && !prefer_long;
    }
    (long_sig, short_sig, poi_l, poi_s)
}

fn build_opposite_donchian_exits(
    data: &MarketData,
    entries: &[bool],
    period: usize,
    direction: i8,
    exits: &mut [bool],
) {
    let n = data.len();
    let roll_low = shift(&rolling_min(&data.low, period), 1);
    let roll_high = shift(&rolling_max(&data.high, period), 1);
    let mut in_pos = false;
    for i in 0..n {
        if in_pos {
            if direction > 0 {
                let rl = roll_low[i];
                if rl.is_finite() && data.close[i] < rl {
                    exits[i] = true;
                    in_pos = false;
                }
            } else {
                let rh = roll_high[i];
                if rh.is_finite() && data.close[i] > rh {
                    exits[i] = true;
                    in_pos = false;
                }
            }
        }
        if entries[i] && !in_pos {
            in_pos = true;
        }
    }
}

fn build_chandelier_exits(
    data: &MarketData,
    entries: &[bool],
    direction: i8,
    atr_period: usize,
    atr_mult: f64,
    exits: &mut [bool],
) {
    let atr_vals = self::indicators::atr(&data.high, &data.low, &data.close, atr_period);
    let n = data.len();
    let mut in_pos = false;
    let mut hh = 0.0;
    let mut ll = 0.0;
    for i in 0..n {
        let was_in_pos = in_pos;
        if in_pos {
            if direction > 0 {
                if data.high[i] > hh {
                    hh = data.high[i];
                }
            } else if data.low[i] < ll {
                ll = data.low[i];
            }
            let a = atr_vals[i];
            if a.is_finite() && a > 0.0 {
                if direction > 0 && data.close[i] < hh - atr_mult * a {
                    exits[i] = true;
                } else if direction < 0 && data.close[i] > ll + atr_mult * a {
                    exits[i] = true;
                }
            }
        }
        in_pos = update_signal_position(was_in_pos, entries[i], exits[i]);
        if !was_in_pos && in_pos {
            hh = data.high[i];
            ll = data.low[i];
        }
    }
}

fn build_time_exits(entries: &[bool], max_bars: usize, exits: &mut [bool]) {
    let n = entries.len();
    let mut in_pos = false;
    let mut entry_i = 0usize;
    for i in 0..n {
        let was_in_pos = in_pos;
        if in_pos && i.saturating_sub(entry_i) >= max_bars {
            exits[i] = true;
        }
        in_pos = update_signal_position(was_in_pos, entries[i], exits[i]);
        if !was_in_pos && in_pos {
            entry_i = i;
        }
    }
}

fn build_reclaim_poi_exits(
    data: &MarketData,
    entries: &[bool],
    poi: &[f64],
    direction: i8,
    exits: &mut [bool],
) {
    let n = data.len();
    let mut in_pos = false;
    let mut entry_poi = f64::NAN;
    for i in 0..n {
        let was_in_pos = in_pos;
        if in_pos {
            let p = if entry_poi.is_finite() {
                entry_poi
            } else {
                poi[i]
            };
            let c = data.close[i];
            if p.is_finite() && c.is_finite() {
                if direction > 0 && c < p {
                    exits[i] = true;
                } else if direction < 0 && c > p {
                    exits[i] = true;
                }
            }
        }
        in_pos = update_signal_position(was_in_pos, entries[i], exits[i]);
        if !was_in_pos && in_pos {
            entry_poi = poi[i];
        }
        if was_in_pos && !in_pos {
            entry_poi = f64::NAN;
        }
    }
}

fn param_i32(recipe: &Recipe, genome: &[f64], name: &str, default: i32) -> i32 {
    recipe
        .try_param_value(genome, name)
        .map(|v| v.round() as i32)
        .unwrap_or(default)
}

fn param_f64(recipe: &Recipe, genome: &[f64], name: &str, default: f64) -> f64 {
    let v = recipe.try_param_value(genome, name).unwrap_or(default);
    if v.is_finite() {
        v
    } else {
        default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reclaim_poi_exits_long_when_close_back_through_anchor() {
        use crate::data::MarketData;
        // Minimal OHLC: enter bar0, reclaim below poi=100 on bar1.
        let data = MarketData {
            timestamp: vec![0.0, 1.0, 2.0],
            open: vec![100.0, 100.5, 99.0],
            high: vec![101.0, 101.0, 99.5],
            low: vec![99.5, 99.0, 98.0],
            close: vec![100.8, 99.5, 98.5],
            volume: vec![1.0, 1.0, 1.0],
            session_days: None,
            sanitized_bars: 0,
        };
        let entries = [true, false, false];
        let poi = [100.0, 100.0, 100.0];
        let mut exits = [false; 3];
        build_reclaim_poi_exits(&data, &entries, &poi, 1, &mut exits);
        assert_eq!(exits, [false, true, false]);
    }

    #[test]
    fn time_exit_cannot_reenter_on_same_bar() {
        let entries = [true, true, false];
        let mut exits = [false; 3];
        build_time_exits(&entries, 1, &mut exits);
        assert_eq!(exits, [false, true, false]);
    }

    #[test]
    fn breakout_levels_are_not_tick_rounded() {
        let poi: f64 = 100.03;
        let atr: f64 = 0.17;
        let mult: f64 = 1.3;
        let level = poi + mult * atr;
        assert!((level - 100.251).abs() < 1e-12);
        assert_ne!(level, (level / 0.25).round() * 0.25);
    }
}
