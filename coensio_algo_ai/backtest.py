"""Backtest a strategy genome.

see coesnio
"""

from __future__ import annotations

from coensio_algo_ai.data import load_ohlcv_file
from coensio_algo_ai.genome_fmt import parse_genome_str
from coensio_algo_ai.progress import format_progress_line, metrics_to_fitness_dict
from coensio_algo_ai.strategy import load_strategy

from coensio_algo_ai.fills import BacktestResult, EngineConfig, run_backtest_genome


def run_genome_backtest(
    strategy_id: str,
    data_file: str,
    genome: str,
    *,
    config: EngineConfig | None = None,
    session: str | None = None,
    session_mode: str | None = None,
) -> BacktestResult:
    mod = load_strategy(strategy_id)
    parsed = parse_genome_str(strategy_id, genome, mod.PARAMS)
    df = load_ohlcv_file(data_file, session=session, session_mode=session_mode)
    datafile = str(df.attrs.get("datafile") or data_file)
    cfg = (
        config
        if config is not None
        else EngineConfig.from_cfg(datafile=datafile)
    )
    return run_backtest_genome(df, strategy_id, parsed.genome, config=cfg)


def format_result(
    label: str,
    result: BacktestResult,
    *,
    session: str = "none",
    config: EngineConfig | None = None,
    datafile: str | None = None,
) -> str:
    eng = config if config is not None else EngineConfig.from_cfg(datafile=datafile)
    m = metrics_to_fitness_dict(result.metrics)
    line = format_progress_line(
        m,
        best_fit=float(m["ret_dd_ratio"]),
        ms_per_str=0.0,
        session=session,
        bet_label=f"${eng.fixed_bet_size:.0f} {eng.bet_mode}",
        prefix="",
    )
    return f"{label}\n{line}"
