/// Immutable columnar OHLCV market data.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MarketData {
    pub timestamp: Vec<f64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    /// Session-local calendar day index supplied by the Python data contract.
    pub session_days: Option<Vec<i32>>,
    /// Bars repaired by OHLC sanitization (corrupt ticks, e.g. open=0.06 on ~1180 BTC).
    pub sanitized_bars: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataError {
    Empty,
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite {
        field: &'static str,
        index: usize,
    },
    InvalidOhlc {
        index: usize,
    },
}

impl std::fmt::Display for DataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "market data must not be empty"),
            Self::LengthMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "{field} length {actual} does not match open length {expected}"
            ),
            Self::NonFinite { field, index } => {
                write!(f, "{field}[{index}] is not finite")
            }
            Self::InvalidOhlc { index } => {
                write!(f, "high[{index}] < low[{index}]")
            }
        }
    }
}

impl std::error::Error for DataError {}

impl MarketData {
    pub fn new(
        timestamp: Vec<f64>,
        mut open: Vec<f64>,
        mut high: Vec<f64>,
        mut low: Vec<f64>,
        mut close: Vec<f64>,
        volume: Vec<f64>,
    ) -> Result<Self, DataError> {
        let n = open.len();
        if n == 0 {
            return Err(DataError::Empty);
        }

        let fields: [(&str, usize); 5] = [
            ("timestamp", timestamp.len()),
            ("high", high.len()),
            ("low", low.len()),
            ("close", close.len()),
            ("volume", volume.len()),
        ];
        for (field, len) in fields {
            if len != n {
                return Err(DataError::LengthMismatch {
                    field,
                    expected: n,
                    actual: len,
                });
            }
        }

        for (field, values) in [
            ("open", &open),
            ("high", &high),
            ("low", &low),
            ("close", &close),
        ] {
            for (i, v) in values.iter().enumerate() {
                if !v.is_finite() {
                    return Err(DataError::NonFinite { field, index: i });
                }
            }
        }

        for (i, (h, l)) in high.iter().zip(low.iter()).enumerate() {
            if *h < *l {
                return Err(DataError::InvalidOhlc { index: i });
            }
        }

        let sanitized_bars = sanitize_ohlc(&mut open, &mut high, &mut low, &mut close, 5.0);

        Ok(Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            session_days: None,
            sanitized_bars,
        })
    }

    pub fn set_session_days(&mut self, session_days: Vec<i32>) -> Result<(), DataError> {
        if session_days.len() != self.len() {
            return Err(DataError::LengthMismatch {
                field: "session_days",
                expected: self.len(),
                actual: session_days.len(),
            });
        }
        self.session_days = Some(session_days);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.open.len()
    }

    /// Causal prefix `[0..end)` for forward / look-ahead checks (no re-sanitize).
    pub fn prefix(&self, end: usize) -> Self {
        let end = end.min(self.len()).max(1);
        Self {
            timestamp: self.timestamp[..end].to_vec(),
            open: self.open[..end].to_vec(),
            high: self.high[..end].to_vec(),
            low: self.low[..end].to_vec(),
            close: self.close[..end].to_vec(),
            volume: self.volume[..end].to_vec(),
            session_days: self
                .session_days
                .as_ref()
                .map(|days| days[..end].to_vec()),
            sanitized_bars: self.sanitized_bars,
        }
    }
}

/// Replace O/H/L/C values that jump more than `max_jump` x from the running reference close.
/// Matches the Python side `coensio_algo_ai.data_clean.clean_ohlc_df`.
fn sanitize_ohlc(
    open: &mut [f64],
    high: &mut [f64],
    low: &mut [f64],
    close: &mut [f64],
    max_jump: f64,
) -> u32 {
    let n = open.len();
    if n == 0 {
        return 0;
    }

    let mut ref_close = if close[0].is_finite() && close[0] > 0.0 {
        close[0]
    } else {
        1.0
    };
    let mut fixed = 0u32;

    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    for i in 0..n {
        let r = if ref_close.is_finite() && ref_close > 0.0 {
            ref_close
        } else {
            1.0
        };
        let lo_band = r / max_jump;
        let hi_band = r * max_jump;
        let mut bad = false;

        for v in [&mut open[i], &mut high[i], &mut low[i], &mut close[i]] {
            if !v.is_finite() || *v <= 0.0 || *v < lo_band || *v > hi_band {
                *v = r;
                bad = true;
            }
        }
        if bad {
            fixed += 1;
        }

        let hi = open[i].max(high[i]).max(low[i]).max(close[i]);
        let lo = open[i].min(high[i]).min(low[i]).min(close[i]);
        high[i] = hi;
        low[i] = lo;
        ref_close = close[i];
    }

    fixed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_corrupt_btc_tick() {
        let mut o = vec![1182.98, 0.06];
        let mut h = vec![1182.99, 1186.6];
        let mut l = vec![1182.2, 0.06];
        let mut c = vec![1182.99, 1178.85];
        let fixed = sanitize_ohlc(&mut o, &mut h, &mut l, &mut c, 5.0);
        assert_eq!(fixed, 1);
        assert!((o[1] - 1182.99).abs() < 1e-9);
        assert!(l[1] >= o[1].min(c[1]));
        assert!(h[1] >= l[1]);
    }

    #[test]
    fn market_data_sanitizes_on_load() {
        let data = MarketData::new(
            vec![0.0, 1.0],
            vec![1182.98, 0.06],
            vec![1182.99, 1186.6],
            vec![1182.2, 0.06],
            vec![1182.99, 1178.85],
            vec![1.0, 1.0],
        )
        .unwrap();
        assert_eq!(data.sanitized_bars, 1);
        assert!((data.open[1] - 1182.99).abs() < 1e-9);
    }
}
