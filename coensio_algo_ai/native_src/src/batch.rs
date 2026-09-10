use std::cell::RefCell;
use std::fmt;

use rayon::prelude::*;

use crate::engine::{run_backtest, EngineConfig};
use crate::indicators::FeatureBank;
use crate::metrics::{self, CompactMetrics};
use crate::recipe::{get_recipe, RecipeError};
use crate::registry::{get_dataset, RegistryError};
use crate::runtime::{build_signals, SignalBuffers};
use crate::session::BatchContext;

thread_local! {
    static SIGNAL_BUFFERS: RefCell<SignalBuffers> = RefCell::new(SignalBuffers::default());
}

#[derive(Debug, Clone, PartialEq)]
pub enum BatchError {
    Registry(RegistryError),
    Recipe(RecipeError),
    Engine(String),
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Registry(err) => write!(f, "{err}"),
            Self::Recipe(err) => write!(f, "{err}"),
            Self::Engine(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for BatchError {}

impl From<RegistryError> for BatchError {
    fn from(value: RegistryError) -> Self {
        Self::Registry(value)
    }
}

impl From<RecipeError> for BatchError {
    fn from(value: RecipeError) -> Self {
        Self::Recipe(value)
    }
}

fn dedupe_genomes(genomes: &[Vec<f64>]) -> (Vec<usize>, Vec<Vec<f64>>) {
    let mut unique: Vec<Vec<f64>> = Vec::new();
    let mut remap = Vec::with_capacity(genomes.len());
    for genome in genomes {
        if let Some(idx) = unique.iter().position(|u| u == genome) {
            remap.push(idx);
        } else {
            remap.push(unique.len());
            unique.push(genome.clone());
        }
    }
    (remap, unique)
}

fn evaluate_indicator_batch(
    data: &crate::data::MarketData,
    recipe: &crate::recipe::Recipe,
    genomes: &[Vec<f64>],
    config: &EngineConfig,
) -> Result<Vec<CompactMetrics>, BatchError> {
    let bank = FeatureBank::build(data, recipe, genomes);
    let ctx = BatchContext::default();
    genomes
        .par_iter()
        .map(|genome| {
            SIGNAL_BUFFERS.with(|bufs| {
                let mut bufs = bufs.borrow_mut();
                build_signals(data, recipe, genome, &bank, &ctx, &mut bufs);
                let result =
                    run_backtest(data, &bufs.entries, &bufs.exits, &bufs.directions, config)
                        .map_err(|e| BatchError::Engine(e.to_string()))?;
                Ok(with_sum_fees(result.metrics, config, &result.trades))
            })
        })
        .collect()
}

fn evaluate_plugin_batch(
    data: &crate::data::MarketData,
    recipe: &crate::recipe::Recipe,
    genomes: &[Vec<f64>],
    config: &EngineConfig,
) -> Result<Vec<CompactMetrics>, BatchError> {
    let bank = FeatureBank::build(data, recipe, genomes);
    let ctx = BatchContext::for_recipe(data, recipe);
    let (remap, unique_genomes) = dedupe_genomes(genomes);

    let unique_signals: Vec<SignalBuffers> = unique_genomes
        .par_iter()
        .map(|genome| {
            SIGNAL_BUFFERS.with(|bufs| {
                let mut bufs = bufs.borrow_mut();
                build_signals(data, recipe, genome, &bank, &ctx, &mut bufs);
                bufs.clone()
            })
        })
        .collect();

    genomes
        .par_iter()
        .enumerate()
        .map(|(i, _)| {
            let bufs = &unique_signals[remap[i]];
            let result = run_backtest(data, &bufs.entries, &bufs.exits, &bufs.directions, config)
                .map_err(|e| BatchError::Engine(e.to_string()))?;
            Ok(with_sum_fees(result.metrics, config, &result.trades))
        })
        .collect()
}

pub fn evaluate_batch(
    dataset_id: u64,
    strategy_id: &str,
    genomes: &[Vec<f64>],
    config: &EngineConfig,
) -> Result<Vec<CompactMetrics>, BatchError> {
    config
        .validate()
        .map_err(|e| BatchError::Engine(e.to_string()))?;

    let data = get_dataset(dataset_id)?;
    let recipe = get_recipe(strategy_id)?;
    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    for genome in genomes {
        recipe.validate_genome(genome)?;
    }

    if recipe.is_plugin() {
        evaluate_plugin_batch(&data, &recipe, genomes, config)
    } else {
        evaluate_indicator_batch(&data, &recipe, genomes, config)
    }
}

fn with_sum_fees(
    mut metrics: CompactMetrics,
    config: &EngineConfig,
    trades: &[crate::engine::Trade],
) -> CompactMetrics {
    metrics.sum_fees = if metrics.trades > 0 {
        metrics::roundtrip_fee_per_trade(config, trades)
    } else {
        0.0
    };
    metrics
}

pub fn evaluate_recipe_detailed(
    dataset_id: u64,
    strategy_id: &str,
    genome: &[f64],
    config: &EngineConfig,
) -> Result<crate::engine::BacktestResult, BatchError> {
    config
        .validate()
        .map_err(|e| BatchError::Engine(e.to_string()))?;

    let data = get_dataset(dataset_id)?;
    let recipe = get_recipe(strategy_id)?;
    recipe.validate_genome(genome)?;

    let bank = FeatureBank::build(&data, &recipe, &[genome.to_vec()]);
    let ctx = BatchContext::for_recipe(&data, &recipe);
    let mut bufs = SignalBuffers::default();
    build_signals(&data, &recipe, genome, &bank, &ctx, &mut bufs);
    let mut result = run_backtest(&data, &bufs.entries, &bufs.exits, &bufs.directions, config)
        .map_err(|e| BatchError::Engine(e.to_string()))?;
    result.metrics = with_sum_fees(result.metrics, config, &result.trades);
    Ok(result)
}

/// Forward (bar-drip) mode: at each bar `i`, rebuild signals using only data `[0..=i]`,
/// then run the standard engine on the causal signal arrays.
///
/// If batch signals peek at future bars, forward metrics/trades diverge from
/// [`evaluate_recipe_detailed`]. Matching results imply no look-ahead in the signal path.
pub fn evaluate_recipe_forward(
    dataset_id: u64,
    strategy_id: &str,
    genome: &[f64],
    config: &EngineConfig,
) -> Result<(crate::engine::BacktestResult, Option<usize>), BatchError> {
    config
        .validate()
        .map_err(|e| BatchError::Engine(e.to_string()))?;

    let data = get_dataset(dataset_id)?;
    let recipe = get_recipe(strategy_id)?;
    recipe.validate_genome(genome)?;

    let n = data.len();
    let mut causal = SignalBuffers::default();
    causal.resize(n);

    // Batch signals (full history) for mismatch reporting
    let bank_full = FeatureBank::build(&data, &recipe, &[genome.to_vec()]);
    let ctx_full = BatchContext::for_recipe(&data, &recipe);
    let mut batch_bufs = SignalBuffers::default();
    build_signals(&data, &recipe, genome, &bank_full, &ctx_full, &mut batch_bufs);

    let mut first_mismatch: Option<usize> = None;
    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    let progress_every = (n / 20).max(500); // ~5% or every 500 bars
    for end in 1..=n {
        let i = end - 1;
        if end == 1 || end == n || end % progress_every == 0 {
            let pct = 100.0 * (end as f64) / (n as f64);
            eprintln!(
                "  forward progress: {end}/{n} bars ({pct:.1}%)",
            );
        }
        let prefix = data.prefix(end);
        let bank = FeatureBank::build(&prefix, &recipe, &[genome.to_vec()]);
        let ctx = BatchContext::for_recipe(&prefix, &recipe);
        let mut bufs = SignalBuffers::default();
        build_signals(&prefix, &recipe, genome, &bank, &ctx, &mut bufs);
        let last = bufs.entries.len() - 1;
        causal.entries[i] = bufs.entries[last];
        causal.exits[i] = bufs.exits[last];
        causal.directions[i] = bufs.directions[last];
        if first_mismatch.is_none()
            && (causal.entries[i] != batch_bufs.entries[i]
                || causal.exits[i] != batch_bufs.exits[i]
                || causal.directions[i] != batch_bufs.directions[i])
        {
            first_mismatch = Some(i);
        }
    }

    let mut result = run_backtest(
        &data,
        &causal.entries,
        &causal.exits,
        &causal.directions,
        config,
    )
    .map_err(|e| BatchError::Engine(e.to_string()))?;
    result.metrics = with_sum_fees(result.metrics, config, &result.trades);
    Ok((result, first_mismatch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::MarketData;
    use crate::registry::register_dataset;

    fn sample_data() -> MarketData {
        let n = 40;
        MarketData::new(
            (0..n).map(|i| i as f64 * 3600.0).collect(),
            (0..n).map(|i| 100.0 + (i as f64 * 0.2)).collect(),
            (0..n).map(|i| 101.0 + (i as f64 * 0.2)).collect(),
            (0..n).map(|i| 99.0 + (i as f64 * 0.2)).collect(),
            (0..n).map(|i| 100.0 + (i as f64 * 0.2)).collect(),
            vec![1000.0; n],
        )
        .unwrap()
    }

    fn base_config() -> EngineConfig {
        EngineConfig {
            initial_capital: 10_000.0,
            bet_mode: crate::engine::BetMode::Fixed,
            fixed_bet_size: 2_000.0,
            price_bet_frac: 0.0,
            commission_rate: 0.00045,
            slippage_rate: 0.0002,
        }
    }

    /// Same genomes evaluated twice must give identical metrics, for every
    /// installed strategy (mid-range genome plus min/max corners).
    #[test]
    fn batch_is_deterministic_for_all_strategies() {
        let id = register_dataset(sample_data()).unwrap();
        for sid in crate::recipe::list_strategy_ids() {
            let recipe = crate::recipe::get_recipe(&sid).unwrap();
            let mid: Vec<f64> = recipe.params.iter().map(|p| (p.min + p.max) * 0.5).collect();
            let lo: Vec<f64> = recipe.params.iter().map(|p| p.min).collect();
            let hi: Vec<f64> = recipe.params.iter().map(|p| p.max).collect();
            let genomes = vec![mid, lo, hi];
            let a = evaluate_batch(id, &sid, &genomes, &base_config())
                .unwrap_or_else(|e| panic!("{sid}: {e}"));
            let b = evaluate_batch(id, &sid, &genomes, &base_config()).unwrap();
            assert_eq!(a, b, "{sid}: non-deterministic batch");
            assert!(a.iter().all(|m| m.sum_fees >= 0.0), "{sid}: negative fees");
        }
    }

    #[test]
    fn dedupe_identical_genomes() {
        let genomes = vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![3.0, 4.0]];
        let (remap, unique) = dedupe_genomes(&genomes);
        assert_eq!(unique.len(), 2);
        assert_eq!(remap, vec![0, 0, 1]);
    }
}
