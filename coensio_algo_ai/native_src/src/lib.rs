//! coensio-algo-ai native core - powered by coensio.com
//! copyright coesnio.com, see coesnio

mod batch;
mod data;
mod engine;
mod indicators;
mod metrics;
mod recipe;
mod registry;
mod runtime;
mod session;
mod strategies_registry {
    include!(concat!(env!("OUT_DIR"), "/strategies_registry.rs"));
}

use batch::evaluate_batch as evaluate_batch_inner;
use batch::evaluate_recipe_detailed as evaluate_recipe_detailed_inner;
use batch::evaluate_recipe_forward as evaluate_recipe_forward_inner;
use data::MarketData;
use engine::{run_backtest, run_backtest_batch, BetMode, EngineConfig};
use metrics::CompactMetrics;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use recipe::list_strategy_ids;
use registry::{register_dataset as register_dataset_inner, RegistryError};
use std::time::Instant;

fn py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn extract_f64_vec(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        return Ok(v);
    }
    let buf = PyBuffer::<f64>::get(obj)?;
    buf.to_vec(py)
}

fn extract_bool_vec(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
    if let Ok(v) = obj.extract::<Vec<bool>>() {
        return Ok(v);
    }
    let buf = PyBuffer::<u8>::get(obj)?;
    Ok(buf.to_vec(py)?.into_iter().map(|b| b != 0).collect())
}

fn extract_i8_vec(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<i8>> {
    if let Ok(v) = obj.extract::<Vec<i8>>() {
        return Ok(v);
    }
    if let Ok(v) = obj.extract::<Vec<i32>>() {
        return v
            .into_iter()
            .map(|x| {
                i8::try_from(x)
                    .map_err(|_| PyValueError::new_err(format!("direction {x} out of i8 range")))
            })
            .collect();
    }
    let floats = extract_f64_vec(py, obj)?;
    floats
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            let rounded = v.round() as i32;
            if (v - f64::from(rounded)).abs() > 1e-9 || (rounded != -1 && rounded != 1) {
                Err(PyValueError::new_err(format!(
                    "directions[{i}] must be -1 or +1, got {v}"
                )))
            } else {
                Ok(rounded as i8)
            }
        })
        .collect()
}

struct BacktestInputs {
    data: MarketData,
    entries: Vec<bool>,
    exits: Vec<bool>,
    directions: Vec<i8>,
    config: EngineConfig,
}

#[allow(clippy::too_many_arguments)]
fn parse_inputs(
    py: Python<'_>,
    timestamp: &Bound<'_, PyAny>,
    open: &Bound<'_, PyAny>,
    high: &Bound<'_, PyAny>,
    low: &Bound<'_, PyAny>,
    close: &Bound<'_, PyAny>,
    volume: &Bound<'_, PyAny>,
    entries: &Bound<'_, PyAny>,
    exits: &Bound<'_, PyAny>,
    directions: &Bound<'_, PyAny>,
    initial_capital: f64,
    bet_mode: &str,
    fixed_bet_size: f64,
    price_bet_frac: f64,
    commission_rate: f64,
    slippage_rate: f64,
) -> PyResult<BacktestInputs> {
    let data = MarketData::new(
        extract_f64_vec(py, timestamp)?,
        extract_f64_vec(py, open)?,
        extract_f64_vec(py, high)?,
        extract_f64_vec(py, low)?,
        extract_f64_vec(py, close)?,
        extract_f64_vec(py, volume)?,
    )
    .map_err(py_err)?;

    Ok(BacktestInputs {
        data,
        entries: extract_bool_vec(py, entries)?,
        exits: extract_bool_vec(py, exits)?,
        directions: extract_i8_vec(py, directions)?,
        config: parse_engine_config(
            initial_capital,
            bet_mode,
            fixed_bet_size,
            price_bet_frac,
            commission_rate,
            slippage_rate,
        )?,
    })
}

fn metrics_to_dict<'py>(
    py: Python<'py>,
    metrics: &CompactMetrics,
    include_sum_fees: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("rdd", metrics.rdd)?;
    dict.set_item("net_pnl", metrics.net_pnl)?;
    dict.set_item("max_dd", metrics.max_dd)?;
    dict.set_item("net_avg", metrics.net_avg)?;
    dict.set_item("trades", metrics.trades)?;
    dict.set_item("stability", metrics.stability)?;
    dict.set_item("recent_year_pnl", metrics.recent_year_pnl)?;
    dict.set_item("half2_pnl", metrics.half2_pnl)?;
    if include_sum_fees {
        dict.set_item("sum_fees", metrics.sum_fees)?;
    }
    Ok(dict)
}

/// Compact single-position backtest (metrics only).
///
/// Execution: entry/exit signals at bar i fill at bar i+1 open; open positions
/// force-close at the final bar close. Python is not called inside the bar loop.
#[pyfunction]
#[pyo3(signature = (
    timestamp,
    open,
    high,
    low,
    close,
    volume,
    entries,
    exits,
    directions,
    initial_capital = 10_000.0,
    bet_mode = "fixed",
    fixed_bet_size = 1_000.0,
    price_bet_frac = 0.0,
    commission_rate = 0.0,
    slippage_rate = 0.0,
))]
#[allow(clippy::too_many_arguments)]
fn backtest<'py>(
    py: Python<'py>,
    timestamp: &Bound<'py, PyAny>,
    open: &Bound<'py, PyAny>,
    high: &Bound<'py, PyAny>,
    low: &Bound<'py, PyAny>,
    close: &Bound<'py, PyAny>,
    volume: &Bound<'py, PyAny>,
    entries: &Bound<'py, PyAny>,
    exits: &Bound<'py, PyAny>,
    directions: &Bound<'py, PyAny>,
    initial_capital: f64,
    bet_mode: &str,
    fixed_bet_size: f64,
    price_bet_frac: f64,
    commission_rate: f64,
    slippage_rate: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let inputs = parse_inputs(
        py,
        timestamp,
        open,
        high,
        low,
        close,
        volume,
        entries,
        exits,
        directions,
        initial_capital,
        bet_mode,
        fixed_bet_size,
        price_bet_frac,
        commission_rate,
        slippage_rate,
    )?;

    let result = py
        .detach(|| {
            run_backtest(
                &inputs.data,
                &inputs.entries,
                &inputs.exits,
                &inputs.directions,
                &inputs.config,
            )
        })
        .map_err(py_err)?;

    metrics_to_dict(py, &result.metrics, false)
}

/// Parallel batch of compact backtests (Rayon). One OHLCV, many signal sets.
#[pyfunction]
#[pyo3(signature = (
    timestamp,
    open,
    high,
    low,
    close,
    volume,
    entries_batch,
    exits_batch,
    directions_batch,
    initial_capital,
    bet_mode,
    fixed_bet_size,
    price_bet_frac,
    commission_rate,
    slippage_rate,
))]
#[allow(clippy::too_many_arguments)]
fn backtest_batch<'py>(
    py: Python<'py>,
    timestamp: &Bound<'py, PyAny>,
    open: &Bound<'py, PyAny>,
    high: &Bound<'py, PyAny>,
    low: &Bound<'py, PyAny>,
    close: &Bound<'py, PyAny>,
    volume: &Bound<'py, PyAny>,
    entries_batch: &Bound<'py, PyAny>,
    exits_batch: &Bound<'py, PyAny>,
    directions_batch: &Bound<'py, PyAny>,
    initial_capital: f64,
    bet_mode: &str,
    fixed_bet_size: f64,
    price_bet_frac: f64,
    commission_rate: f64,
    slippage_rate: f64,
) -> PyResult<Bound<'py, PyList>> {
    let data = MarketData::new(
        extract_f64_vec(py, timestamp)?,
        extract_f64_vec(py, open)?,
        extract_f64_vec(py, high)?,
        extract_f64_vec(py, low)?,
        extract_f64_vec(py, close)?,
        extract_f64_vec(py, volume)?,
    )
    .map_err(py_err)?;
    let config = parse_engine_config(
        initial_capital,
        bet_mode,
        fixed_bet_size,
        price_bet_frac,
        commission_rate,
        slippage_rate,
    )?;

    let entries_list = entries_batch.cast::<PyList>()?;
    let exits_list = exits_batch.cast::<PyList>()?;
    let dirs_list = directions_batch.cast::<PyList>()?;
    if entries_list.len() != exits_list.len() || entries_list.len() != dirs_list.len() {
        return Err(PyValueError::new_err(
            "entries_batch/exits_batch/directions_batch length mismatch",
        ));
    }
    let mut entries: Vec<Vec<bool>> = Vec::with_capacity(entries_list.len());
    let mut exits: Vec<Vec<bool>> = Vec::with_capacity(exits_list.len());
    let mut directions: Vec<Vec<i8>> = Vec::with_capacity(dirs_list.len());
    for i in 0..entries_list.len() {
        entries.push(extract_bool_vec(py, &entries_list.get_item(i)?)?);
        exits.push(extract_bool_vec(py, &exits_list.get_item(i)?)?);
        directions.push(extract_i8_vec(py, &dirs_list.get_item(i)?)?);
    }

    let rows = py
        .detach(|| run_backtest_batch(&data, &entries, &exits, &directions, &config))
        .map_err(py_err)?;

    let out = PyList::empty(py);
    for m in &rows {
        out.append(metrics_to_dict(py, m, false)?)?;
    }
    Ok(out)
}

/// Detailed backtest with per-bar equity and trade list.
#[pyfunction]
#[pyo3(signature = (
    timestamp,
    open,
    high,
    low,
    close,
    volume,
    entries,
    exits,
    directions,
    initial_capital = 10_000.0,
    bet_mode = "fixed",
    fixed_bet_size = 1_000.0,
    price_bet_frac = 0.0,
    commission_rate = 0.0,
    slippage_rate = 0.0,
))]
#[allow(clippy::too_many_arguments)]
fn backtest_detailed<'py>(
    py: Python<'py>,
    timestamp: &Bound<'py, PyAny>,
    open: &Bound<'py, PyAny>,
    high: &Bound<'py, PyAny>,
    low: &Bound<'py, PyAny>,
    close: &Bound<'py, PyAny>,
    volume: &Bound<'py, PyAny>,
    entries: &Bound<'py, PyAny>,
    exits: &Bound<'py, PyAny>,
    directions: &Bound<'py, PyAny>,
    initial_capital: f64,
    bet_mode: &str,
    fixed_bet_size: f64,
    price_bet_frac: f64,
    commission_rate: f64,
    slippage_rate: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let inputs = parse_inputs(
        py,
        timestamp,
        open,
        high,
        low,
        close,
        volume,
        entries,
        exits,
        directions,
        initial_capital,
        bet_mode,
        fixed_bet_size,
        price_bet_frac,
        commission_rate,
        slippage_rate,
    )?;

    let result = py
        .detach(|| {
            run_backtest(
                &inputs.data,
                &inputs.entries,
                &inputs.exits,
                &inputs.directions,
                &inputs.config,
            )
        })
        .map_err(py_err)?;

    let dict = metrics_to_dict(py, &result.metrics, false)?;
    dict.set_item("equity_curve", result.equity_curve)?;

    let trade_rows = PyList::empty(py);
    for trade in &result.trades {
        let row = PyDict::new(py);
        row.set_item("entry_bar", trade.entry_bar)?;
        row.set_item("exit_bar", trade.exit_bar)?;
        row.set_item("raw_entry_price", trade.raw_entry_price)?;
        row.set_item("raw_exit_price", trade.raw_exit_price)?;
        row.set_item("entry_price", trade.entry_price)?;
        row.set_item("exit_price", trade.exit_price)?;
        row.set_item("entry_notional", trade.entry_notional)?;
        row.set_item("exit_consideration", trade.exit_consideration)?;
        row.set_item("entry_commission", trade.entry_commission)?;
        row.set_item("exit_commission", trade.exit_commission)?;
        row.set_item("slippage_cost", trade.slippage_cost)?;
        row.set_item("direction", trade.direction)?;
        row.set_item("gross_pnl", trade.gross_pnl)?;
        row.set_item("net_pnl", trade.net_pnl)?;
        row.set_item("exit_reason", trade.exit_reason)?;
        trade_rows.append(row)?;
    }
    dict.set_item("trades", trade_rows)?;
    Ok(dict)
}

fn parse_engine_config(
    initial_capital: f64,
    bet_mode: &str,
    fixed_bet_size: f64,
    price_bet_frac: f64,
    commission_rate: f64,
    slippage_rate: f64,
) -> PyResult<EngineConfig> {
    let config = EngineConfig {
        initial_capital,
        bet_mode: BetMode::parse(bet_mode),
        fixed_bet_size,
        price_bet_frac,
        commission_rate,
        slippage_rate,
    };
    config.validate().map_err(py_err)?;
    Ok(config)
}

fn extract_genomes(py: Python<'_>, genomes_obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let list = genomes_obj.cast::<PyList>()?;
    let mut genomes = Vec::with_capacity(list.len());
    for item in list.iter() {
        genomes.push(extract_f64_vec(py, &item)?);
    }
    Ok(genomes)
}

/// Register OHLCV columns once; returns an integer dataset id for batch evaluation.
#[pyfunction]
#[pyo3(signature = (
    timestamp,
    open,
    high,
    low,
    close,
    volume,
    session_days = None,
))]
#[allow(clippy::too_many_arguments)]
fn register_dataset<'py>(
    py: Python<'py>,
    timestamp: &Bound<'py, PyAny>,
    open: &Bound<'py, PyAny>,
    high: &Bound<'py, PyAny>,
    low: &Bound<'py, PyAny>,
    close: &Bound<'py, PyAny>,
    volume: &Bound<'py, PyAny>,
    session_days: Option<&Bound<'py, PyAny>>,
) -> PyResult<(u64, u32)> {
    let mut data = MarketData::new(
        extract_f64_vec(py, timestamp)?,
        extract_f64_vec(py, open)?,
        extract_f64_vec(py, high)?,
        extract_f64_vec(py, low)?,
        extract_f64_vec(py, close)?,
        extract_f64_vec(py, volume)?,
    )
    .map_err(py_err)?;
    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    if let Some(days) = session_days {
        data.set_session_days(days.extract::<Vec<i32>>()?)
            .map_err(py_err)?;
    }
    let sanitized = data.sanitized_bars;

    let id = py
        .detach(|| register_dataset_inner(data))
        .map_err(|e: RegistryError| PyValueError::new_err(e.to_string()))?;
    Ok((id, sanitized))
}

/// Built-in recipe strategy ids.
#[pyfunction]
fn list_strategies() -> Vec<String> {
    list_strategy_ids()
}

/// Evaluate many genomes in parallel using a registered dataset and recipe strategy.
#[pyfunction]
#[pyo3(signature = (
    dataset_id,
    strategy_id,
    genomes,
    initial_capital = 10_000.0,
    bet_mode = "fixed",
    fixed_bet_size = 2_000.0,
    price_bet_frac = 0.0,
    commission_rate = 0.00045,
    slippage_rate = 0.0002,
))]
#[allow(clippy::too_many_arguments)]
fn evaluate_batch<'py>(
    py: Python<'py>,
    dataset_id: u64,
    strategy_id: &str,
    genomes: &Bound<'py, PyAny>,
    initial_capital: f64,
    bet_mode: &str,
    fixed_bet_size: f64,
    price_bet_frac: f64,
    commission_rate: f64,
    slippage_rate: f64,
) -> PyResult<Bound<'py, PyList>> {
    let config = parse_engine_config(
        initial_capital,
        bet_mode,
        fixed_bet_size,
        price_bet_frac,
        commission_rate,
        slippage_rate,
    )?;
    let genome_vecs = extract_genomes(py, genomes)?;
    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee

    let results = py
        .detach(|| evaluate_batch_inner(dataset_id, strategy_id, &genome_vecs, &config))
        .map_err(py_err)?;

    let out = PyList::empty(py);
    for metrics in results {
        out.append(metrics_to_dict(py, &metrics, true)?)?;
    }
    Ok(out)
}

/// Like [`evaluate_batch`] but also returns batch timing stats.
#[pyfunction]
#[pyo3(signature = (
    dataset_id,
    strategy_id,
    genomes,
    initial_capital = 10_000.0,
    bet_mode = "fixed",
    fixed_bet_size = 2_000.0,
    price_bet_frac = 0.0,
    commission_rate = 0.00045,
    slippage_rate = 0.0002,
))]
#[allow(clippy::too_many_arguments)]
fn evaluate_batch_timed<'py>(
    py: Python<'py>,
    dataset_id: u64,
    strategy_id: &str,
    genomes: &Bound<'py, PyAny>,
    initial_capital: f64,
    bet_mode: &str,
    fixed_bet_size: f64,
    price_bet_frac: f64,
    commission_rate: f64,
    slippage_rate: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let config = parse_engine_config(
        initial_capital,
        bet_mode,
        fixed_bet_size,
        price_bet_frac,
        commission_rate,
        slippage_rate,
    )?;
    let genome_vecs = extract_genomes(py, genomes)?;
    let n = genome_vecs.len();

    let (metrics, elapsed_ms) = py.detach(|| {
        let start = Instant::now();
        let results = evaluate_batch_inner(dataset_id, strategy_id, &genome_vecs, &config);
        (results, start.elapsed().as_secs_f64() * 1000.0)
    });
    let metrics = metrics.map_err(py_err)?;
    let ms_per = if n > 0 { elapsed_ms / n as f64 } else { 0.0 };
    let per_sec = if ms_per > 0.0 { 1000.0 / ms_per } else { 0.0 };

    let dict = PyDict::new(py);
    let rows = PyList::empty(py);
    for row in metrics {
        rows.append(metrics_to_dict(py, &row, true)?)?;
    }
    dict.set_item("results", rows)?;
    dict.set_item("elapsed_ms", elapsed_ms)?;
    dict.set_item("ms_per_strategy", ms_per)?;
    dict.set_item("strategies_per_sec", per_sec)?;
    Ok(dict)
}

/// Detailed recipe backtest on a registered dataset (trades + equity for reports).
#[pyfunction]
#[pyo3(signature = (
    dataset_id,
    strategy_id,
    genome,
    initial_capital = 10_000.0,
    bet_mode = "fixed",
    fixed_bet_size = 2_000.0,
    price_bet_frac = 0.0,
    commission_rate = 0.00045,
    slippage_rate = 0.0002,
))]
#[allow(clippy::too_many_arguments)]
fn evaluate_recipe_detailed<'py>(
    py: Python<'py>,
    dataset_id: u64,
    strategy_id: &str,
    genome: &Bound<'py, PyAny>,
    initial_capital: f64,
    bet_mode: &str,
    fixed_bet_size: f64,
    price_bet_frac: f64,
    commission_rate: f64,
    slippage_rate: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let config = parse_engine_config(
        initial_capital,
        bet_mode,
        fixed_bet_size,
        price_bet_frac,
        commission_rate,
        slippage_rate,
    )?;
    let genome_vec = extract_f64_vec(py, genome)?;

    let result = py
        .detach(|| evaluate_recipe_detailed_inner(dataset_id, strategy_id, &genome_vec, &config))
        .map_err(py_err)?;

    let dict = metrics_to_dict(py, &result.metrics, true)?;
    dict.set_item("equity_curve", result.equity_curve)?;

    let trade_rows = PyList::empty(py);
    for trade in &result.trades {
        let row = PyDict::new(py);
        row.set_item("entry_bar", trade.entry_bar)?;
        row.set_item("exit_bar", trade.exit_bar)?;
        row.set_item("raw_entry_price", trade.raw_entry_price)?;
        row.set_item("raw_exit_price", trade.raw_exit_price)?;
        row.set_item("entry_price", trade.entry_price)?;
        row.set_item("exit_price", trade.exit_price)?;
        row.set_item("entry_notional", trade.entry_notional)?;
        row.set_item("exit_consideration", trade.exit_consideration)?;
        row.set_item("entry_commission", trade.entry_commission)?;
        row.set_item("exit_commission", trade.exit_commission)?;
        row.set_item("slippage_cost", trade.slippage_cost)?;
        row.set_item("direction", trade.direction)?;
        row.set_item("gross_pnl", trade.gross_pnl)?;
        row.set_item("net_pnl", trade.net_pnl)?;
        row.set_item("exit_reason", trade.exit_reason)?;
        trade_rows.append(row)?;
    }
    dict.set_item("trades", trade_rows)?;
    Ok(dict)
}

/// Bar-drip forward backtest: signals rebuilt on causal prefixes only.
/// Returns the same metrics/trades shape as [`evaluate_recipe_detailed`], plus
/// `first_signal_mismatch_bar` (None if batch signals match causal at every bar).
#[pyfunction]
#[pyo3(signature = (
    dataset_id,
    strategy_id,
    genome,
    initial_capital = 10_000.0,
    bet_mode = "fixed",
    fixed_bet_size = 2_000.0,
    price_bet_frac = 0.0,
    commission_rate = 0.00045,
    slippage_rate = 0.0002,
))]
#[allow(clippy::too_many_arguments)]
fn evaluate_recipe_forward<'py>(
    py: Python<'py>,
    dataset_id: u64,
    strategy_id: &str,
    genome: &Bound<'py, PyAny>,
    initial_capital: f64,
    bet_mode: &str,
    fixed_bet_size: f64,
    price_bet_frac: f64,
    commission_rate: f64,
    slippage_rate: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let config = parse_engine_config(
        initial_capital,
        bet_mode,
        fixed_bet_size,
        price_bet_frac,
        commission_rate,
        slippage_rate,
    )?;
    let genome_vec = extract_f64_vec(py, genome)?;

    let (result, first_mismatch) = py
        .detach(|| evaluate_recipe_forward_inner(dataset_id, strategy_id, &genome_vec, &config))
        .map_err(py_err)?;

    let dict = metrics_to_dict(py, &result.metrics, true)?;
    dict.set_item("equity_curve", result.equity_curve)?;
    dict.set_item("mode", "forward")?;
    match first_mismatch {
        Some(i) => dict.set_item("first_signal_mismatch_bar", i)?,
        None => dict.set_item("first_signal_mismatch_bar", py.None())?,
    }

    let trade_rows = PyList::empty(py);
    for trade in &result.trades {
        let row = PyDict::new(py);
        row.set_item("entry_bar", trade.entry_bar)?;
        row.set_item("exit_bar", trade.exit_bar)?;
        row.set_item("raw_entry_price", trade.raw_entry_price)?;
        row.set_item("raw_exit_price", trade.raw_exit_price)?;
        row.set_item("entry_price", trade.entry_price)?;
        row.set_item("exit_price", trade.exit_price)?;
        row.set_item("entry_notional", trade.entry_notional)?;
        row.set_item("exit_consideration", trade.exit_consideration)?;
        row.set_item("entry_commission", trade.entry_commission)?;
        row.set_item("exit_commission", trade.exit_commission)?;
        row.set_item("slippage_cost", trade.slippage_cost)?;
        row.set_item("direction", trade.direction)?;
        row.set_item("gross_pnl", trade.gross_pnl)?;
        row.set_item("net_pnl", trade.net_pnl)?;
        row.set_item("exit_reason", trade.exit_reason)?;
        trade_rows.append(row)?;
    }
    dict.set_item("trades", trade_rows)?;
    Ok(dict)
}

#[pymodule]
fn engine_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(backtest, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_batch, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_detailed, m)?)?;
    m.add_function(wrap_pyfunction!(register_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(list_strategies, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch_timed, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_recipe_detailed, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_recipe_forward, m)?)?;
    Ok(())
}

#[cfg(test)]
mod binding_tests {
    use crate::data::DataError;
    use crate::engine::ConfigError;

    #[test]
    fn data_error_display() {
        let err = DataError::Empty;
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn config_error_display() {
        let err = ConfigError::InvalidFixedBetSize;
        assert!(err.to_string().contains("fixed_bet_size"));
    }
}
