use std::collections::{HashMap, HashSet};

use crate::data::MarketData;
use crate::recipe::{IndicatorDef, Recipe};

/// Money Flow Index (0-100).
pub fn mfi(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 || period < 2 {
        return out;
    }

    let mut tp = vec![0.0; n];
    let mut rm = vec![0.0; n];
    for i in 0..n {
        tp[i] = (high[i] + low[i] + close[i]) / 3.0;
        rm[i] = tp[i] * volume[i];
    }

    for (i, slot) in out.iter_mut().enumerate().take(n).skip(period) {
        let mut pos_sum = 0.0;
        let mut neg_sum = 0.0;
        for j in (i - period + 1)..=i {
            let diff = tp[j] - tp[j - 1];
            if diff > 0.0 {
                pos_sum += rm[j];
            } else if diff < 0.0 {
                neg_sum += rm[j];
            }
        }
        *slot = if neg_sum > 0.0 {
            100.0 - (100.0 / (1.0 + pos_sum / neg_sum))
        } else {
            100.0
        };
    }
    out
}

/// Larry Williams Ultimate Oscillator (0-100).
pub fn ultimate_oscillator(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return out;
    }

    let p1 = p1.max(2);
    let p2 = p2.max(p1 + 1);
    let p3 = p3.max(p2 + 1);
    let max_p = p3;

    let mut bp = vec![0.0; n];
    let mut tr = vec![0.0; n];
    for i in 0..n {
        let prev_close = if i > 0 { close[i - 1] } else { close[i] };
        let true_low = low[i].min(prev_close);
        let true_high = high[i].max(prev_close);
        bp[i] = close[i] - true_low;
        tr[i] = true_high - true_low;
    }

    let avg = |period: usize, i: usize| -> f64 {
        let mut bp_sum = 0.0;
        let mut tr_sum = 0.0;
        for j in (i - period + 1)..=i {
            bp_sum += bp[j];
            tr_sum += tr[j];
        }
        if tr_sum > 0.0 {
            bp_sum / tr_sum
        } else {
            f64::NAN
        }
    };

    for (i, slot) in out.iter_mut().enumerate().take(n).skip(max_p) {
        let a1 = avg(p1, i);
        let a2 = avg(p2, i);
        let a3 = avg(p3, i);
        if a1.is_finite() && a2.is_finite() && a3.is_finite() {
            *slot = 100.0 * (4.0 * a1 + 2.0 * a2 + a3) / 7.0;
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct UoKey {
    p1: u32,
    p2: u32,
    p3: u32,
}

/// Precomputed indicator variants for a batch of genomes.
#[derive(Debug, Default)]
pub struct FeatureBank {
    mfi: HashMap<u32, Vec<f64>>,
    uo: HashMap<UoKey, Vec<f64>>,
    prev_high: Vec<f64>,
}

impl FeatureBank {
    pub fn build(data: &MarketData, recipe: &Recipe, genomes: &[Vec<f64>]) -> Self {
        if recipe.is_plugin() {
            return Self::default();
        }

        let mut bank = Self {
            prev_high: shift_prev(&data.high),
            ..Default::default()
        };
        // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee

        match recipe.indicator.as_ref().expect("indicator recipe") {
            IndicatorDef::Mfi { period_param } => {
                let idx = recipe.param_index(period_param);
                let mut periods = HashSet::new();
                for genome in genomes {
                    if let Some(&period) = genome.get(idx) {
                        periods.insert(clamp_usize(period as i64, 2, 200) as u32);
                    }
                }
                for period in periods {
                    bank.mfi.insert(
                        period,
                        mfi(
                            &data.high,
                            &data.low,
                            &data.close,
                            &data.volume,
                            period as usize,
                        ),
                    );
                }
            }
            IndicatorDef::UltimateOscillator {
                p1_param,
                p2_param,
                p3_param,
            } => {
                let i1 = recipe.param_index(p1_param);
                let i2 = recipe.param_index(p2_param);
                let i3 = recipe.param_index(p3_param);
                let mut keys = HashSet::new();
                for genome in genomes {
                    if genome.len() <= i1.max(i2).max(i3) {
                        continue;
                    }
                    let p1 = clamp_usize(genome[i1] as i64, 2, 200);
                    let p2 = clamp_usize(genome[i2] as i64, p1 + 1, 300);
                    let p3 = clamp_usize(genome[i3] as i64, p2 + 1, 400);
                    keys.insert(UoKey {
                        p1: p1 as u32,
                        p2: p2 as u32,
                        p3: p3 as u32,
                    });
                }
                for key in keys {
                    bank.uo.insert(
                        key.clone(),
                        ultimate_oscillator(
                            &data.high,
                            &data.low,
                            &data.close,
                            key.p1 as usize,
                            key.p2 as usize,
                            key.p3 as usize,
                        ),
                    );
                }
            }
        }

        bank
    }

    pub fn mfi_series(&self, period: usize) -> Option<&[f64]> {
        self.mfi.get(&(period as u32)).map(|v| v.as_slice())
    }

    pub fn uo_series(&self, p1: usize, p2: usize, p3: usize) -> Option<&[f64]> {
        self.uo
            .get(&UoKey {
                p1: p1 as u32,
                p2: p2 as u32,
                p3: p3 as u32,
            })
            .map(|v| v.as_slice())
    }

    pub fn prev_high(&self) -> &[f64] {
        &self.prev_high
    }
}

fn shift_prev(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![f64::NAN; n];
    if n > 1 {
        out[1..].copy_from_slice(&values[..n - 1]);
    }
    out
}

fn clamp_usize(v: i64, min: usize, max: usize) -> usize {
    v.max(min as i64).min(max as i64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mfi_bounded() {
        let high = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let low = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
        let close = vec![9.5, 10.5, 11.5, 12.5, 13.5, 14.5];
        let volume = vec![100.0; 6];
        let values = mfi(&high, &low, &close, &volume, 3);
        assert!(values[0].is_nan());
        assert!(values[3].is_finite());
        assert!((0.0..=100.0).contains(&values[3]));
    }

    #[test]
    fn uo_bounded() {
        let high = (0..40).map(|i| 100.0 + i as f64).collect::<Vec<_>>();
        let low = high.iter().map(|h| h - 1.0).collect::<Vec<_>>();
        let close = high.iter().map(|h| h - 0.5).collect::<Vec<_>>();
        let values = ultimate_oscillator(&high, &low, &close, 7, 14, 28);
        let last = *values.last().unwrap();
        assert!(last.is_finite());
        assert!((0.0..=100.0).contains(&last));
    }
}
