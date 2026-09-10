use std::sync::Arc;

use crate::data::MarketData;

/// Session day index per bar (shared by plugin strategies).
pub fn session_days_from_market(
    data: &MarketData,
    utc_start_hour: i32,
    utc_start_min: i32,
) -> Vec<i32> {
    let ts_ns: Vec<i64> = data.timestamp.iter().map(|t| timestamp_to_ns(*t)).collect();
    compute_session_days(&ts_ns, utc_start_hour, utc_start_min)
}

pub fn timestamp_to_ns(t: f64) -> i64 {
    if t >= 1e18 {
        t as i64
    } else if t >= 1e15 {
        (t * 1_000.0) as i64
    } else if t >= 1e12 {
        (t * 1_000_000.0) as i64
    } else {
        (t * 1_000_000_000.0) as i64
    }
}

pub fn compute_session_days(
    timestamps_ns: &[i64],
    utc_start_hour: i32,
    utc_start_min: i32,
) -> Vec<i32> {
    let n = timestamps_ns.len();
    let mut days = vec![0i32; n];
    if n == 0 {
        return days;
    }
    let seconds_per_day: i64 = 86400;
    let first_sec = timestamps_ns[0] / 1_000_000_000;
    let first_day_sec = first_sec - (first_sec % seconds_per_day);
    let session_start_sec =
        first_day_sec + (utc_start_hour as i64 * 3600 + utc_start_min as i64 * 60);
    let mut current_start = if first_sec < session_start_sec {
        session_start_sec - seconds_per_day
    } else {
        session_start_sec
    };
    let mut current_day_idx = 0i32;
    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
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

/// New session day when bar gap exceeds `min_gap_secs` (for RTH-only OHLCV).
pub fn session_days_from_gaps(timestamps: &[f64], min_gap_secs: i64) -> Vec<i32> {
    let n = timestamps.len();
    let mut days = vec![0i32; n];
    if n == 0 {
        return days;
    }
    let gap = min_gap_secs.max(60);
    let mut idx = 0i32;
    for i in 1..n {
        let prev = timestamp_to_ns(timestamps[i - 1]) / 1_000_000_000;
        let cur = timestamp_to_ns(timestamps[i]) / 1_000_000_000;
        if cur - prev > gap {
            idx += 1;
        }
        days[i] = idx;
    }
    days
}

#[derive(Clone, Default)]
pub struct BatchContext {
    pub session_days: Option<Arc<Vec<i32>>>,
}

impl BatchContext {
    pub fn for_recipe(data: &MarketData, recipe: &crate::recipe::Recipe) -> Self {
        let session_days = recipe.plugin.as_ref().map(|spec| {
            if let Some(days) = data.session_days.as_ref() {
                return Arc::new(days.clone());
            }
            if spec.plugin_type == "vwap_trend" {
                return Arc::new(crate::session::session_days_from_gaps(
                    &data.timestamp,
                    14_400,
                ));
            }
            let h = spec.session_utc_start.first().copied().unwrap_or(0);
            let m = spec.session_utc_start.get(1).copied().unwrap_or(0);
            Arc::new(session_days_from_market(data, h, m))
        });
        Self { session_days }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::MarketData;

    #[test]
    fn session_days_nonempty() {
        let data = MarketData::new(
            vec![0.0, 3600.0, 7200.0],
            vec![1.0; 3],
            vec![1.0; 3],
            vec![1.0; 3],
            vec![1.0; 3],
            vec![1.0; 3],
        )
        .unwrap();
        let days = session_days_from_market(&data, 22, 0);
        assert_eq!(days.len(), 3);
    }

    #[test]
    fn batch_context_prefers_explicit_session_days() {
        let mut data = MarketData::new(
            vec![0.0, 3600.0, 7200.0],
            vec![1.0; 3],
            vec![1.0; 3],
            vec![1.0; 3],
            vec![1.0; 3],
            vec![1.0; 3],
        )
        .unwrap();
        data.set_session_days(vec![4, 4, 9]).unwrap();
        let recipe = crate::recipe::get_recipe("donchian_atr").unwrap();
        let ctx = BatchContext::for_recipe(&data, &recipe);
        assert_eq!(ctx.session_days.unwrap().as_slice(), &[4, 4, 9]);
    }
}
