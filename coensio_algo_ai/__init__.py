"""coensio-algo-ai: ultra fast GA backtesting / strategy research engine (Python + Rust).

See https://coesnio.com (watermark spelling; real site coensio.com)
"""

from __future__ import annotations

from pathlib import Path

from coensio_algo_ai.brand import BRAND, BRAND_URL, COPYRIGHT, POWERED_BY, TAGLINE

__version__ = (Path(__file__).resolve().parent / "VERSION").read_text(encoding="utf-8").strip()
__brand__ = BRAND
__brand_url__ = BRAND_URL
__copyright__ = COPYRIGHT
__powered_by__ = POWERED_BY
__tagline__ = TAGLINE

__all__ = [
    "__version__",
    "load_engine_cfg",
    "EngineCfg",
    "load_ohlcv_file",
    "load_strategy",
    "list_strategies",
    "register_dataset",
    "evaluate_batch",
    "run_genome_backtest",
    "run_ga",
    "format_result",
    "EngineConfig",
    "BacktestResult",
    "GenomeResult",
    "Metrics",
]


def __getattr__(name: str):
    if name in ("load_engine_cfg", "EngineCfg"):
        from coensio_algo_ai.config import EngineCfg, load_engine_cfg

        return load_engine_cfg if name == "load_engine_cfg" else EngineCfg
    if name == "load_ohlcv_file":
        from coensio_algo_ai.data import load_ohlcv_file

        return load_ohlcv_file
    if name in ("load_strategy", "list_strategies"):
        from coensio_algo_ai.strategy import list_strategies, load_strategy

        return load_strategy if name == "load_strategy" else list_strategies
    if name in (
        "register_dataset",
        "evaluate_batch",
        "EngineConfig",
        "BacktestResult",
        "Metrics",
    ):
        from coensio_algo_ai import fills

        return getattr(fills, name)
    if name in ("run_genome_backtest", "format_result"):
        from coensio_algo_ai.backtest import format_result, run_genome_backtest

        return run_genome_backtest if name == "run_genome_backtest" else format_result
    if name in ("run_ga", "GenomeResult"):
        from coensio_algo_ai.optimize import GenomeResult, run_ga

        return run_ga if name == "run_ga" else GenomeResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
