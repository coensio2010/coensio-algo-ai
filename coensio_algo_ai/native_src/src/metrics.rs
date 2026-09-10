use crate::engine::Trade;

/// Compact GA-facing metrics.
///
/// - `rdd`: return-to-drawdown ratio, `net_pnl / max_dd` (0 when `max_dd` is 0 and `net_pnl <= 0`).
/// - `net_pnl`: realized PnL in USD after all fills and costs.
/// - `max_dd`: peak-to-trough drawdown in USD on the mark-to-market equity curve.
/// - `net_avg`: average PnL per completed trade (`net_pnl / trades`, 0 when no trades).
/// - `trades`: number of completed round-trips.
/// - `stability`: equity-curve stability score in [0, 1] (linear R^2 x gain-distribution consistency).
/// - `recent_year_pnl`: equity change over the trailing 365.25 days using bar timestamps.
/// - `half2_pnl`: equity change over the second half of the backtest.
/// - `sum_fees`: average round-trip commission per trade in USD (0 when no trades).
#[derive(Debug, Clone, PartialEq)]
pub struct CompactMetrics {
    pub rdd: f64,
    pub net_pnl: f64,
    pub max_dd: f64,
    pub net_avg: f64,
    pub trades: u32,
    pub stability: f64,
    pub recent_year_pnl: f64,
    pub half2_pnl: f64,
    pub sum_fees: f64,
}

pub struct MetricsInput<'a> {
    pub initial_capital: f64,
    pub equity_curve: &'a [f64],
    pub timestamps: &'a [f64],
    pub trades: &'a [Trade],
}

pub fn compute_compact(input: MetricsInput<'_>) -> CompactMetrics {
    let net_pnl = input
        .equity_curve
        .last()
        .copied()
        .unwrap_or(input.initial_capital)
        - input.initial_capital;
    let max_dd = max_drawdown(input.equity_curve);
    let trades = input.trades.len() as u32;
    let net_avg = if trades > 0 {
        net_pnl / trades as f64
    } else {
        0.0
    };
    let rdd = if max_dd > 0.0 {
        net_pnl / max_dd
    } else if net_pnl > 0.0 {
        9_999.9
    } else {
        0.0
    };
    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee

    CompactMetrics {
        rdd: round4(rdd),
        net_pnl: round2(net_pnl),
        max_dd: round2(max_dd),
        net_avg: round2(net_avg),
        trades,
        stability: round4(stability_score(input.equity_curve)),
        recent_year_pnl: round2(recent_year_pnl(input.equity_curve, input.timestamps)),
        half2_pnl: round2(half2_pnl(input.equity_curve)),
        sum_fees: 0.0,
    }
}

/// Average actual round-trip commission per trade.
pub fn roundtrip_fee_per_trade(
    _config: &crate::engine::EngineConfig,
    trades: &[crate::engine::Trade],
) -> f64 {
    if trades.is_empty() {
        return 0.0;
    }
    let sum: f64 = trades
        .iter()
        .map(|t| t.entry_commission + t.exit_commission)
        .sum();
    round2(sum / trades.len() as f64)
}

fn max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0;
    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    for &value in equity {
        if value > peak {
            peak = value;
        }
        let dd = peak - value;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd.max(0.0)
}

/// Stability: linear fit quality multiplied by even gain distribution across segments.
fn stability_score(equity: &[f64]) -> f64 {
    let n = equity.len();
    if n < 6 {
        return 0.0;
    }

    let x_mean = (n - 1) as f64 / 2.0;
    let y_mean = equity.iter().sum::<f64>() / n as f64;

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for (i, &y) in equity.iter().enumerate() {
        let x = i as f64;
        let dx = x - x_mean;
        let dy = y - y_mean;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    if den_x <= 0.0 || den_y <= 0.0 {
        return 0.0;
    }

    let slope = num / den_x;
    let intercept = y_mean - slope * x_mean;
    let mut ss_res = 0.0;
    for (i, &y) in equity.iter().enumerate() {
        let pred = intercept + slope * i as f64;
        let res = y - pred;
        ss_res += res * res;
    }
    let r2 = (1.0 - ss_res / den_y).clamp(0.0, 1.0);

    let total_gain = equity[n - 1] - equity[0];
    if total_gain <= 0.0 {
        return 0.0;
    }

    let k = 10usize.min(n / 2).max(2);
    let ideal = total_gain / k as f64;
    let mut acc = 0.0;
    for seg in 0..k {
        let lo = (seg as f64 / k as f64 * (n - 1) as f64) as usize;
        let hi = ((seg + 1) as f64 / k as f64 * (n - 1) as f64) as usize;
        let seg_gain = equity[hi] - equity[lo];
        acc += (seg_gain / ideal).clamp(0.0, 1.0);
    }
    (r2 * (acc / k as f64)).clamp(0.0, 1.0)
}

fn years_to_timestamp_delta(years: f64, ts_sample: f64) -> f64 {
    let ts = ts_sample.abs();
    if ts >= 1e18 {
        365.25 * years * 24.0 * 3600.0 * 1e9
    } else if ts >= 1e15 {
        365.25 * years * 24.0 * 3600.0 * 1e6
    } else if ts >= 1e12 {
        365.25 * years * 24.0 * 3600.0 * 1e3
    } else {
        365.25 * years * 24.0 * 3600.0
    }
}

fn half2_pnl(equity: &[f64]) -> f64 {
    let n = equity.len();
    if n < 4 {
        return 0.0;
    }
    let mid = n / 2;
    equity[n - 1] - equity[mid]
}

fn recent_year_pnl(equity: &[f64], timestamps: &[f64]) -> f64 {
    if equity.len() < 2 || timestamps.len() != equity.len() {
        return 0.0;
    }
    let last_ts = timestamps[timestamps.len() - 1];
    let cutoff = last_ts - years_to_timestamp_delta(1.0, last_ts);
    let idx = searchsorted_left(timestamps, cutoff).min(equity.len() - 1);
    equity[equity.len() - 1] - equity[idx]
}

fn searchsorted_left(timestamps: &[f64], cutoff: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = timestamps.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if timestamps[mid] < cutoff {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_drawdown_basic() {
        let eq = vec![100.0, 110.0, 105.0, 120.0, 90.0];
        assert!((max_drawdown(&eq) - 30.0).abs() < 1e-9);
    }

    #[test]
    fn rdd_zero_dd_loser() {
        let m = compute_compact(MetricsInput {
            initial_capital: 10_000.0,
            equity_curve: &[10_000.0, 10_000.0],
            timestamps: &[0.0, 86_400.0],
            trades: &[],
        });
        assert_eq!(m.rdd, 0.0);
        assert_eq!(m.trades, 0);
    }
}
