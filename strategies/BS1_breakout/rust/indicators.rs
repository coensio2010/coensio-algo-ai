/// Core technical indicators for BOS backtesting.
/// All functions are pure Rust, no dependencies beyond std.
/// Every function returns NaN-prefixed arrays matching the input length.

/// Exponential moving average (same as pandas ewm(span=period, adjust=False))
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 || period == 0 {
        return result;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut cum: f64 = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        let v = data[i];
        if v.is_nan() {
            continue;
        }
        count += 1;
        if count == 1 {
            cum = v;
        } else {
            cum = alpha * v + (1.0 - alpha) * cum;
        }
        if count >= period {
            result[i] = cum;
        }
    }
    result
}

/// Simple moving average
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 || period == 0 {
        return result;
    }
    let mut sum = 0.0;
    for i in 0..n {
        let v = data[i];
        if v.is_nan() {
            continue;
        }
        sum += v;
        if i >= period {
            sum -= data[i - period];
        }
        if i >= period - 1 {
            result[i] = sum / period as f64;
        }
    }
    result
}

/// Wilder's RMA (same as pandas ewm(alpha=1/period, adjust=False))
pub fn rma(data: &[f64], period: usize) -> Vec<f64> {
    rma_series(data, period)
}

/// Same as rma - exposed under both names for convenience
pub fn rma_series(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 || period == 0 {
        return result;
    }
    let alpha = 1.0 / period as f64;
    let mut cum: f64 = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        let v = data[i];
        if v.is_nan() {
            continue;
        }
        count += 1;
        if count == 1 {
            cum = v;
        } else {
            cum = alpha * v + (1.0 - alpha) * cum;
        }
        if count >= period {
            result[i] = cum;
        }
    }
    result
}

/// True Range (per-bar)
fn tr(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 {
        return result;
    }
    result[0] = high[0] - low[0];
    for i in 1..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        result[i] = hl.max(hc).max(lc);
    }
    result
}

/// Average True Range (RMA of TR)
pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let tr_vals = tr(high, low, close);
    rma(&tr_vals, period)
}

pub fn adx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut adx_vals = vec![f64::NAN; n];
    let mut pdi = vec![f64::NAN; n];
    let mut mdi = vec![f64::NAN; n];
    if n == 0 || period == 0 {
        return (adx_vals, pdi, mdi);
    }

    let tr_vals = tr(high, low, close);
    let mut up = vec![0.0_f64; n];
    let mut down = vec![0.0_f64; n];

    for i in 1..n {
        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];
        if up_move > down_move && up_move > 0.0 {
            up[i] = up_move;
        }
        if down_move > up_move && down_move > 0.0 {
            down[i] = down_move;
        }
    }

    let tr_rma = rma(&tr_vals, period);
    let up_rma = rma(&up, period);
    let down_rma = rma(&down, period);

    for i in 0..n {
        let tr = tr_rma[i];
        if tr.is_nan() || tr <= 0.0 {
            continue;
        }
        pdi[i] = 100.0 * up_rma[i] / tr;
        mdi[i] = 100.0 * down_rma[i] / tr;
        let sum = pdi[i] + mdi[i];
        if sum <= 0.0 {
            continue;
        }
        let dx = 100.0 * (pdi[i] - mdi[i]).abs() / sum;
        // Accumulate RMA for ADX
        if i == 0 {
            adx_vals[i] = dx;
        } else if !adx_vals[i - 1].is_nan() {
            let alpha = 1.0 / period as f64;
            adx_vals[i] = alpha * dx + (1.0 - alpha) * adx_vals[i - 1];
        } else {
            adx_vals[i] = dx;
        }
    }
    (adx_vals, pdi, mdi)
}

/// Bollinger Bands
pub fn bbands(data: &[f64], period: usize, nstd: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    let mut lower = vec![f64::NAN; n];
    let mut mid = vec![f64::NAN; n];
    let mut upper = vec![f64::NAN; n];
    if n == 0 || period == 0 {
        return (lower, mid, upper);
    }

    let mid_vals = sma(data, period);
    for i in (period - 1)..n {
        let m = mid_vals[i];
        if m.is_nan() {
            continue;
        }
        let mut sum_sq = 0.0;
        for j in (i + 1 - period)..=i {
            let diff = data[j] - m;
            sum_sq += diff * diff;
        }
        let std = (sum_sq / period as f64).sqrt();
        mid[i] = m;
        lower[i] = m - nstd * std;
        upper[i] = m + nstd * std;
    }
    (lower, mid, upper)
}

/// Z-score
#[allow(dead_code)]
pub fn zscore(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 || period == 0 {
        return result;
    }

    for i in (period - 1)..n {
        let mut sum = 0.0;
        for j in (i + 1 - period)..=i {
            sum += data[j];
        }
        let mean = sum / period as f64;
        let mut sum_sq = 0.0;
        for j in (i + 1 - period)..=i {
            let diff = data[j] - mean;
            sum_sq += diff * diff;
        }
        let std = (sum_sq / period as f64).sqrt();
        if std > 0.0 {
            result[i] = (data[i] - mean) / std;
        }
    }
    result
}

/// Rolling max
pub fn rolling_max(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 || period == 0 {
        return result;
    }
    for i in 0..n {
        let start = if i + 1 >= period { i + 1 - period } else { 0 };
        let mut mx = data[start];
        for j in (start + 1)..=i {
            if data[j] > mx {
                mx = data[j];
            }
        }
        result[i] = mx;
    }
    result
}

/// Rolling min
pub fn rolling_min(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 || period == 0 {
        return result;
    }
    for i in 0..n {
        let start = if i + 1 >= period { i + 1 - period } else { 0 };
        let mut mn = data[start];
        for j in (start + 1)..=i {
            if data[j] < mn {
                mn = data[j];
            }
        }
        result[i] = mn;
    }
    result
}

/// Donchian channel
#[allow(dead_code)]
pub fn donchian(high: &[f64], low: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
    let upper = rolling_max(high, period);
    let lower = rolling_min(low, period);
    (upper, lower)
}

/// Shift array by n positions, filling with NaN
pub fn shift(data: &[f64], n: usize) -> Vec<f64> {
    let len = data.len();
    let mut result = vec![f64::NAN; len];
    if n >= len {
        return result;
    }
    for i in n..len {
        result[i] = data[i - n];
    }
    result
}

/// Check if value crosses above threshold (prev <= thr, curr > thr)
#[allow(dead_code)]
pub fn cross_up(data: &[f64], threshold: f64) -> Vec<bool> {
    let n = data.len();
    let mut result = vec![false; n];
    if n == 0 {
        return result;
    }
    for i in 1..n {
        if !data[i].is_nan() && !data[i - 1].is_nan() {
            result[i] = data[i] > threshold && data[i - 1] <= threshold;
        }
    }
    result
}

/// Check if value crosses below threshold (prev >= thr, curr < thr)
#[allow(dead_code)]
pub fn cross_down(data: &[f64], threshold: f64) -> Vec<bool> {
    let n = data.len();
    let mut result = vec![false; n];
    if n == 0 {
        return result;
    }
    for i in 1..n {
        if !data[i].is_nan() && !data[i - 1].is_nan() {
            result[i] = data[i] < threshold && data[i - 1] >= threshold;
        }
    }
    result
}
