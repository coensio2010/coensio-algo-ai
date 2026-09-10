/// BS1 (Breakout) v1.10 signal computation - Rust implementation.
/// Mirrors `strategies/BS1.py`, no lookahead.
/// All indicators precomputed upfront for O(n) performance.

use super::indicators::{atr, ema, sma, bbands, adx, rolling_max, rolling_min, shift};

/// Compute session day index (sequential int per session day) from UTC timestamps.
#[allow(dead_code)]
pub fn compute_session_days(timestamps_ns: &[i64], utc_start_hour: i32, utc_start_min: i32) -> Vec<i32> {
    let n = timestamps_ns.len();
    let mut days = vec![0i32; n];
    if n == 0 { return days; }
    let seconds_per_day: i64 = 86400;
    let first_sec = timestamps_ns[0] / 1_000_000_000;
    let first_day_sec = first_sec - (first_sec % seconds_per_day);
    let session_start_sec = first_day_sec + (utc_start_hour as i64 * 3600 + utc_start_min as i64 * 60);
    let mut current_start = if first_sec < session_start_sec { session_start_sec - seconds_per_day } else { session_start_sec };
    let mut current_day_idx = 0i32;
    for i in 0..n {
        let ts_sec = timestamps_ns[i] / 1_000_000_000;
        while ts_sec >= current_start + seconds_per_day {
            current_start += seconds_per_day;
            current_day_idx += 1;
        }
        days[i] = current_day_idx;
    }
    days
}

/// Causal intraday day field (day_offset=0).
fn day_field_0(open: &[f64], high: &[f64], low: &[f64], close: &[f64], days: &[i32], field: &str) -> Vec<f64> {
    let n = open.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 { return result; }
    match field {
        "open" => {
            let mut day_open: std::collections::HashMap<i32, f64> = std::collections::HashMap::new();
            for i in 0..n { let d = days[i]; day_open.entry(d).or_insert(open[i]); result[i] = day_open[&d]; }
        }
        "high" => {
            let mut running_max: std::collections::HashMap<i32, f64> = std::collections::HashMap::new();
            for i in 0..n {
                let d = days[i];
                result[i] = running_max.get(&d).copied().unwrap_or(f64::NAN);
                let cur = running_max.get(&d).copied().unwrap_or(f64::NEG_INFINITY);
                if high[i] > cur { running_max.insert(d, high[i]); }
            }
        }
        "low" => {
            let mut running_min: std::collections::HashMap<i32, f64> = std::collections::HashMap::new();
            for i in 0..n {
                let d = days[i];
                result[i] = running_min.get(&d).copied().unwrap_or(f64::NAN);
                let cur = running_min.get(&d).copied().unwrap_or(f64::INFINITY);
                if low[i] < cur { running_min.insert(d, low[i]); }
            }
        }
        "close" => {
            let mut last_close: std::collections::HashMap<i32, f64> = std::collections::HashMap::new();
            for i in 0..n {
                let d = days[i];
                result[i] = last_close.get(&d).copied().unwrap_or(f64::NAN);
                last_close.insert(d, close[i]);
            }
        }
        _ => {}
    }
    result
}

/// Completed prior day field (day_offset >= 1).
fn day_field_completed(open: &[f64], high: &[f64], low: &[f64], close: &[f64], days: &[i32], day_offset: usize, field: &str) -> Vec<f64> {
    let n = open.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 || day_offset == 0 { return result; }
    let mut daily_open: std::collections::HashMap<i32, f64> = std::collections::HashMap::new();
    let mut daily_high: std::collections::HashMap<i32, f64> = std::collections::HashMap::new();
    let mut daily_low: std::collections::HashMap<i32, f64> = std::collections::HashMap::new();
    let mut daily_close: std::collections::HashMap<i32, f64> = std::collections::HashMap::new();
    for i in 0..n {
        let d = days[i];
        daily_open.entry(d).or_insert(open[i]);
        daily_high.entry(d).and_modify(|v| *v = (*v).max(high[i])).or_insert(high[i]);
        daily_low.entry(d).and_modify(|v| *v = (*v).min(low[i])).or_insert(low[i]);
        daily_close.insert(d, close[i]);
    }
    let mut all_days: Vec<i32> = daily_open.keys().copied().collect();
    all_days.sort();
    let mut day_lookup: std::collections::HashMap<i32, f64> = std::collections::HashMap::new();
    for (idx, &d) in all_days.iter().enumerate() {
        if idx < day_offset { continue; }
        let target = all_days[idx - day_offset];
        let val = match field {
            "open" => daily_open.get(&target).copied(),
            "high" => daily_high.get(&target).copied(),
            "low" => daily_low.get(&target).copied(),
            "close" => daily_close.get(&target).copied(),
            _ => None,
        };
        if let Some(v) = val { day_lookup.insert(d, v); }
    }
    for i in 0..n { let d = days[i]; if let Some(&v) = day_lookup.get(&d) { result[i] = v; } }
    result
}

fn day_field(open: &[f64], high: &[f64], low: &[f64], close: &[f64], days: &[i32], day_offset: usize, field: &str) -> Vec<f64> {
    if day_offset > 0 { day_field_completed(open, high, low, close, days, day_offset, field) }
    else { day_field_0(open, high, low, close, days, field) }
}

/// POI (Point of Interest) - switch 1-12.
pub fn compute_poi(open: &[f64], high: &[f64], low: &[f64], close: &[f64], days: &[i32], poi_switch: i32, poi_n1: i32) -> (Vec<f64>, Vec<f64>) {
    let n = open.len();
    let sw = poi_switch.max(1).min(12) as usize;
    let n1 = poi_n1.max(1) as usize;
    let c = close;
    let h = high;
    let l = low;
    match sw {
        1 => { let v = day_field(open, high, low, close, days, 1, "close"); (v.clone(), v) }
        2 => { let v = day_field(open, high, low, close, days, 0, "open"); (v.clone(), v) }
        3 => (day_field(open, high, low, close, days, 0, "low"), day_field(open, high, low, close, days, 0, "high")),
        4 => (day_field(open, high, low, close, days, 1, "low"), day_field(open, high, low, close, days, 1, "high")),
        5 => {
            let cd1 = day_field(open, high, low, close, days, 1, "close");
            let od0 = day_field(open, high, low, close, days, 0, "open");
            let mut lo = vec![f64::NAN; n]; let mut hi = vec![f64::NAN; n];
            for i in 0..n { lo[i] = cd1[i].min(od0[i]); hi[i] = cd1[i].max(od0[i]); }
            (lo, hi)
        }
        6 => {
            let cd1 = day_field(open, high, low, close, days, 1, "close");
            let od0 = day_field(open, high, low, close, days, 0, "open");
            let mut lo = vec![f64::NAN; n]; let mut hi = vec![f64::NAN; n];
            for i in 0..n { lo[i] = cd1[i].max(od0[i]); hi[i] = cd1[i].min(od0[i]); }
            (lo, hi)
        }
        7 => {
            let cd1 = day_field(open, high, low, close, days, 1, "close");
            let ld0 = day_field(open, high, low, close, days, 0, "low");
            let hd0 = day_field(open, high, low, close, days, 0, "high");
            let mut lo = vec![f64::NAN; n]; let mut hi = vec![f64::NAN; n];
            for i in 0..n { lo[i] = cd1[i].max(ld0[i]); hi[i] = cd1[i].min(hd0[i]); }
            (lo, hi)
        }
        8 => {
            let mut day_sum: std::collections::HashMap<i32, f64> = std::collections::HashMap::new();
            let mut day_cnt: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
            let mut result = vec![f64::NAN; n];
            for i in 0..n {
                let d = days[i];
                let prev_sum = day_sum.get(&d).copied().unwrap_or(0.0);
                let prev_cnt = day_cnt.get(&d).copied().unwrap_or(0);
                if prev_cnt > 0 { result[i] = prev_sum / prev_cnt as f64; }
                day_sum.insert(d, prev_sum + (high[i] + low[i]) / 2.0);
                day_cnt.insert(d, prev_cnt + 1);
            }
            (result.clone(), result)
        }
        9 => { let v = ema(c, (n1 * 5).max(1)); let s = shift(&v, 1); (s.clone(), s) }
        10 => (shift(&ema(l, (n1 * 5).max(1)), 1), shift(&ema(h, (n1 * 5).max(1)), 1)),
        11 => {
            let med: Vec<f64> = (0..n).map(|i| (high[i] + low[i]) / 2.0).collect();
            let v = ema(&med, (n1 * 5).max(1)); let s = shift(&v, 1); (s.clone(), s)
        }
        12 => { let (lower, _, upper) = bbands(c, (3 * n1).max(2), 1.5); (shift(&lower, 1), shift(&upper, 1)) }
        _ => { let v = day_field(open, high, low, close, days, 1, "close"); (v.clone(), v) }
    }
}

pub struct BS1Filters { pub long: Vec<bool>, pub short: Vec<bool> }

/// BS1 filters (sw 0-40). All indicators precomputed once upfront.
pub fn compute_filters(open: &[f64], high: &[f64], low: &[f64], close: &[f64], days: &[i32], sw: i32, n1: i32, n2: i32) -> BS1Filters {
    let n = close.len();
    let mut long = vec![false; n];
    let mut short = vec![false; n];
    if sw == 0 { return BS1Filters { long: vec![true; n], short: vec![true; n] }; }

    let sw_us = sw.max(0).min(40) as usize;
    let n1_u = n1.max(1) as usize;
    let n2_u = n2.max(1) as usize;

    // Precompute true range once
    let mut tr = vec![0.0f64; n];
    if n > 0 {
        tr[0] = high[0] - low[0];
        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }
    }

    // Precompute commonly used periods
    let p5n1 = (5 * n1_u).max(2);
    let p10n1 = (10 * n1_u).max(2);
    let p5n2 = (5 * n2_u).max(2);
    let p10n2 = (10 * n2_u).max(2);

    // Precompute all indicator arrays we might need
    let atr_3n1 = atr(high, low, close, (3 * n1_u).max(2));
    let atr_5n2 = atr(high, low, close, p5n2);
    let atr_n1x5 = atr(high, low, close, (n1_u * 5).max(2));
    let atr_n1x2 = atr(high, low, close, (n1_u * 2).max(2));
    let atr_n1x3 = atr(high, low, close, (n1_u * 3).max(2));
    let atr_3xn1 = atr(high, low, close, (3 * n1_u).max(2));

    let (_adx_10n1, adx_pdi_10n1, adx_mdi_10n1) = adx(high, low, close, p10n1);
    let (_adx_10n2, _adx_pdi_10n2, adx_mdi_10n2) = adx(high, low, close, p10n2);
    let (_adx_5n1, adx_pdi_5n1, adx_mdi_5n1) = adx(high, low, close, p5n1);
    let (adx_5n2_vals, _, _) = adx(high, low, close, p5n2);

    let sma_3n1 = sma(close, (3 * n1_u).max(2));
    let sma_2n1 = sma(close, (2 * n1_u).max(2));
    let sma_3n2 = sma(close, (3 * n2_u).max(2));

    let roll_max_n2 = rolling_max(high, n2_u);
    let roll_min_n2 = rolling_min(low, n2_u);

    // Precompute day-field arrays for filters that need them
    let df_cd1 = day_field(open, high, low, close, days, 1, "close");
    let df_od0 = day_field(open, high, low, close, days, 0, "open");
    let df_ld0 = day_field(open, high, low, close, days, 0, "low");
    let df_hd0 = day_field(open, high, low, close, days, 0, "high");

    // Shift arrays for filter 27
    let l_shifted = if n1_u < n { shift(low, n1_u) } else { vec![f64::NAN; n] };
    let h_shifted = if n1_u < n { shift(high, n1_u) } else { vec![f64::NAN; n] };

    for i in 0..n {
        let (l, s) = match sw_us {
            1 => {
                let v1 = atr_3n1[i]; let v2 = atr_5n2[i];
                (!v1.is_nan() && !v2.is_nan() && v1 > v2, !v1.is_nan() && !v2.is_nan() && v1 > v2)
            }
            2 => {
                let v1 = atr_3n1[i]; let v2 = atr_5n2[i];
                (!v1.is_nan() && !v2.is_nan() && v1 <= v2, !v1.is_nan() && !v2.is_nan() && v1 <= v2)
            }
            3 => {
                let v1 = atr_3n1[i]; let v2 = atr_5n2[i];
                (!v1.is_nan() && !v2.is_nan() && v1 < v2, !v1.is_nan() && !v2.is_nan() && v1 > v2)
            }
            4 => {
                let v1 = atr_3n1[i]; let v2 = atr_5n2[i];
                (!v1.is_nan() && !v2.is_nan() && v1 > v2, !v1.is_nan() && !v2.is_nan() && v1 < v2)
            }
            5 => { let a = atr_n1x5[i]; let rng = high[i] - low[i]; (!a.is_nan() && rng > a, !a.is_nan() && rng > a) }
            6 => { let a = atr_n1x5[i]; let rng = high[i] - low[i]; (!a.is_nan() && rng <= a, !a.is_nan() && rng <= a) }
            7 => { let a = atr_n1x5[i]; let rng = high[i] - low[i]; let thr = (n2_u as f64 * 0.2) * rng; (!a.is_nan() && thr > a, !a.is_nan() && thr > a) }
            8 => { let a = atr_n1x5[i]; let rng = high[i] - low[i]; let thr = (n2_u as f64 * 0.2) * a; (!a.is_nan() && rng <= thr, !a.is_nan() && rng <= thr) }
            9 => (adx_pdi_10n1[i] < adx_mdi_10n1[i], adx_pdi_10n1[i] > adx_mdi_10n1[i]),
            10 => (adx_pdi_10n1[i] > adx_mdi_10n1[i], adx_pdi_10n1[i] < adx_mdi_10n1[i]),
            11 => (adx_pdi_10n1[i] < adx_mdi_10n2[i], adx_pdi_10n1[i] > adx_mdi_10n2[i]),
            12 => (adx_pdi_10n1[i] > adx_mdi_10n2[i], adx_pdi_10n1[i] < adx_mdi_10n2[i]),
            13 => (adx_pdi_5n1[i] < adx_mdi_5n1[i], adx_pdi_5n1[i] > adx_mdi_5n1[i]),
            14 => (adx_pdi_5n1[i] > adx_mdi_5n1[i], adx_pdi_5n1[i] < adx_mdi_5n1[i]),
            15 => { let a = adx_5n2_vals[i]; (!a.is_nan() && a > (n1_u * 2) as f64, !a.is_nan() && a > (n1_u * 2) as f64) }
            16 => { let a = adx_5n2_vals[i]; (!a.is_nan() && a < (n1_u * 2) as f64, !a.is_nan() && a < (n1_u * 2) as f64) }
            17 => {
                let tr_v = tr[i]; let rm_h = roll_max_n2[i]; let rm_l = roll_min_n2[i];
                (!rm_h.is_nan() && !rm_l.is_nan() && tr_v * (n1_u as f64 * 0.15) >= rm_h - close[i],
                 !rm_h.is_nan() && !rm_l.is_nan() && tr_v * (n1_u as f64 * 0.15) <= close[i] - rm_l)
            }
            18 => {
                let tr_v = tr[i]; let rm_h = roll_max_n2[i]; let rm_l = roll_min_n2[i];
                (!rm_h.is_nan() && !rm_l.is_nan() && tr_v * (n1_u as f64 * 0.15) <= rm_h - close[i],
                 !rm_h.is_nan() && !rm_l.is_nan() && tr_v * (n1_u as f64 * 0.15) >= close[i] - rm_l)
            }
            19 => {
                let a = atr_n1x2[i]; let cd1 = df_cd1[i]; let rm_l = roll_min_n2[i]; let rm_h = roll_max_n2[i];
                (!a.is_nan() && !cd1.is_nan() && !rm_l.is_nan() && a > cd1 - rm_l,
                 !a.is_nan() && !cd1.is_nan() && !rm_h.is_nan() && a < rm_h - cd1)
            }
            20 => {
                let a = atr_n1x2[i]; let cd1 = df_cd1[i]; let rm_l = roll_min_n2[i]; let rm_h = roll_max_n2[i];
                (!a.is_nan() && !cd1.is_nan() && !rm_l.is_nan() && a < cd1 - rm_l,
                 !a.is_nan() && !cd1.is_nan() && !rm_h.is_nan() && a > rm_h - cd1)
            }
            21 => (!df_cd1[i].is_nan() && close[i] - df_cd1[i] > 0.0, !df_cd1[i].is_nan() && df_cd1[i] - close[i] > 0.0),
            22 => (!df_cd1[i].is_nan() && close[i] - df_cd1[i] < 0.0, !df_cd1[i].is_nan() && df_cd1[i] - close[i] < 0.0),
            23 => {
                let a = atr_n1x3[i]; let cd1 = df_cd1[i]; let ld0 = df_ld0[i]; let hd0 = df_hd0[i];
                (!a.is_nan() && !cd1.is_nan() && !ld0.is_nan() && a < cd1 - ld0,
                 !a.is_nan() && !cd1.is_nan() && !hd0.is_nan() && a < hd0 - cd1)
            }
            24 => {
                let a = atr_n1x3[i]; let cd1 = df_cd1[i]; let ld0 = df_ld0[i]; let hd0 = df_hd0[i];
                (!a.is_nan() && !cd1.is_nan() && !ld0.is_nan() && a > cd1 - ld0,
                 !a.is_nan() && !cd1.is_nan() && !hd0.is_nan() && a > hd0 - cd1)
            }
            25 => {
                let a = atr_3xn1[i]; let rm_l = roll_min_n2[i]; let rm_h = roll_max_n2[i];
                (!a.is_nan() && !rm_l.is_nan() && close[i] - rm_l > a,
                 !a.is_nan() && !rm_h.is_nan() && close[i] - rm_h < -a)
            }
            26 => {
                let a = atr_3xn1[i]; let rm_l = roll_min_n2[i]; let rm_h = roll_max_n2[i];
                (!a.is_nan() && !rm_l.is_nan() && close[i] - rm_l < a,
                 !a.is_nan() && !rm_h.is_nan() && close[i] - rm_h > -a)
            }
            27 => {
                (!df_cd1[i].is_nan() && !l_shifted[i].is_nan() && l_shifted[i] >= df_cd1[i],
                 !df_cd1[i].is_nan() && !h_shifted[i].is_nan() && h_shifted[i] <= df_cd1[i])
            }
            28 => (!df_od0[i].is_nan() && close[i] > df_od0[i], !df_od0[i].is_nan() && close[i] < df_od0[i]),
            29 => (!df_od0[i].is_nan() && close[i] < df_od0[i], !df_od0[i].is_nan() && close[i] > df_od0[i]),
            30 => {
                let tr_v = tr[i]; let hd = df_hd0[i]; let ld = df_ld0[i];
                (!df_od0[i].is_nan() && !hd.is_nan() && tr_v >= df_od0[i] - hd,
                 !df_od0[i].is_nan() && !ld.is_nan() && tr_v <= df_od0[i] - ld)
            }
            31 => {
                let pts = (n1_u as f64 * 50.0) / 2.0;
                (!df_cd1[i].is_nan() && !df_ld0[i].is_nan() && df_ld0[i] >= df_cd1[i] - pts,
                 !df_cd1[i].is_nan() && !df_hd0[i].is_nan() && df_hd0[i] <= df_cd1[i] + pts)
            }
            32 => {
                let pts = (n1_u as f64 * 50.0) / 2.0;
                (!df_cd1[i].is_nan() && !df_ld0[i].is_nan() && df_ld0[i] <= df_cd1[i] - pts,
                 !df_cd1[i].is_nan() && !df_hd0[i].is_nan() && df_hd0[i] >= df_cd1[i] + pts)
            }
            33 => (!sma_3n1[i].is_nan() && close[i] > sma_3n1[i], !sma_3n1[i].is_nan() && close[i] > sma_3n1[i]),
            34 => (!sma_3n1[i].is_nan() && close[i] < sma_3n1[i], !sma_3n1[i].is_nan() && close[i] < sma_3n1[i]),
            35 => (!sma_3n1[i].is_nan() && close[i] > sma_3n1[i], !sma_3n2[i].is_nan() && close[i] > sma_3n2[i]),
            36 => (!sma_3n1[i].is_nan() && close[i] < sma_3n1[i], !sma_3n2[i].is_nan() && close[i] < sma_3n2[i]),
            37 => (!sma_2n1[i].is_nan() && close[i] * (0.1 * n2_u as f64) > sma_2n1[i], !sma_2n1[i].is_nan() && close[i] * (0.1 * n2_u as f64) > sma_2n1[i]),
            38 => (!sma_2n1[i].is_nan() && close[i] * (0.1 * n2_u as f64) < sma_2n1[i], !sma_2n1[i].is_nan() && close[i] * (0.1 * n2_u as f64) < sma_2n1[i]),
            39 => (i == 0 || close[i] < close[i - 1], i == 0 || close[i] < close[i - 1]),
            40 => (i > 0 && close[i] > close[i - 1], i > 0 && close[i] > close[i - 1]),
            _ => (true, true),
        };
        long[i] = l;
        short[i] = s;
    }

    BS1Filters { long, short }
}
