"""Python side of the Rust core: register_dataset + evaluate_batch (Rayon).

coeniso native engine core
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from coensio_algo_ai.build_native import find_native_binaries
from coensio_algo_ai.config import EngineCfg, load_engine_cfg
from coensio_algo_ai.data import bar_timestamps_utc_ns, calendar_session_days

PACKAGE_DIR = Path(__file__).resolve().parent
_NATIVE_DIR = PACKAGE_DIR / "native"
_native_mod: Any | None = None


def native_binary_path() -> Path | None:
    """Newest compiled engine_core binary in native/, or None if not built yet."""
    found = find_native_binaries(_NATIVE_DIR)
    return found[0] if found else None


@dataclass(frozen=True)
class EngineConfig:
    initial_capital: float
    fixed_bet_size: float
    commission_rate: float
    slippage_rate: float
    bet_mode: str
    price_bet_frac: float

    @classmethod
    def from_cfg(
        cls,
        cfg: EngineCfg | None = None,
        *,
        datafile: str | Path | None = None,
        fixed_bet_size: float | None = None,
    ) -> EngineConfig:
        from coensio_algo_ai.config import resolve_fixed_bet_size

        c = cfg or load_engine_cfg()
        bet = (
            float(fixed_bet_size)
            if fixed_bet_size is not None
            else resolve_fixed_bet_size(c.fixed_bet_sizes, datafile)
        )
        return cls(
            initial_capital=c.initial_capital,
            fixed_bet_size=bet,
            commission_rate=c.commission_rate,
            slippage_rate=c.slippage_rate,
            bet_mode=c.bet_mode,
            price_bet_frac=c.price_bet_frac,
        )

    def eval_kwargs(self) -> dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "bet_mode": self.bet_mode,
            "fixed_bet_size": self.fixed_bet_size,
            "price_bet_frac": self.price_bet_frac,
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
        }


@dataclass(frozen=True)
class Metrics:
    net_pnl: float
    max_dd: float
    rdd: float
    trades: int
    net_avg: float
    stability: float
    recent_year_pnl: float
    half2_pnl: float
    avg_roundtrip_fee: float


@dataclass(frozen=True)
class BacktestResult:
    metrics: Metrics
    equity_curve: np.ndarray | None
    trades: list


def _import_native():
    global _native_mod
    if _native_mod is not None:
        return _native_mod
    pyd = native_binary_path()
    if pyd is None:
        raise ImportError(
            f"native engine not built (looked in {_NATIVE_DIR}).\n"
            f"Build it with: python {PACKAGE_DIR / 'build_native.py'}  "
            "(needs Rust + maturin; run `python -m coensio_algo_ai doctor` for details)"
        )
    spec = importlib.util.spec_from_file_location("engine_core", pyd)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load native engine: {pyd}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["engine_core"] = mod
    spec.loader.exec_module(mod)
    _native_mod = mod
    return mod


def _row_to_metrics(row: dict) -> Metrics:
    trades = int(row.get("trades", 0))
    avg_fee = float(row.get("avg_roundtrip_fee", row.get("sum_fees", 0.0)))
    return Metrics(
        net_pnl=float(row.get("net_pnl", 0.0)),
        max_dd=float(row.get("max_dd", 0.0)),
        rdd=float(row.get("rdd", 0.0)),
        trades=trades,
        net_avg=float(row.get("net_avg", 0.0)),
        stability=float(row.get("stability", 0.0)),
        recent_year_pnl=float(row.get("recent_year_pnl", 0.0)),
        half2_pnl=float(row.get("half2_pnl", 0.0)),
        avg_roundtrip_fee=avg_fee,
    )


def register_dataset(df: pd.DataFrame) -> int:
    """Register OHLCV once. Returns dataset id for evaluate_batch.

    UTC nanosecond timestamps + calendar session_days.
    """
    native = _import_native()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        source_tz = str(df.attrs.get("source_tz") or "UTC")
        idx = idx.tz_localize(source_tz, ambiguous="infer", nonexistent="raise")
    ts = np.asarray(bar_timestamps_utc_ns(idx), dtype=np.float64)
    session_days = calendar_session_days(idx)
    # de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    dataset_id, sanitized = native.register_dataset(
        ts,
        df["open"].to_numpy(dtype=np.float64),
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["close"].to_numpy(dtype=np.float64),
        df["volume"].to_numpy(dtype=np.float64),
        session_days,
    )
    df.attrs["rust_sanitized"] = int(sanitized)
    return int(dataset_id)


def evaluate_batch(
    dataset_id: int,
    strategy_id: str,
    genomes: Sequence[Sequence[float]],
    config: EngineConfig | None = None,
) -> list[Metrics]:
    """Rayon batch: Rust signals + fills for many genomes."""
    cfg = config if config is not None else EngineConfig.from_cfg()
    native = _import_native()
    genome_vecs = [list(map(float, g)) for g in genomes]
    rows = native.evaluate_batch(
        dataset_id,
        strategy_id,
        genome_vecs,
        **cfg.eval_kwargs(),
    )
    return [_row_to_metrics(row) for row in rows]


def run_backtest_genome(
    df: pd.DataFrame,
    strategy_id: str,
    genome: Sequence[float],
    config: EngineConfig | None = None,
) -> BacktestResult:
    """Single genome via evaluate_batch (same engine path as GA)."""
    ds = register_dataset(df)
    metrics = evaluate_batch(ds, strategy_id, [genome], config=config)[0]
    return BacktestResult(metrics=metrics, equity_curve=None, trades=[])


def evaluate_recipe_detailed(
    df: pd.DataFrame,
    strategy_id: str,
    genome: Sequence[float],
    config: EngineConfig | None = None,
) -> dict[str, Any]:
    """Detailed backtest: metrics + trades + equity_curve."""
    cfg = config if config is not None else EngineConfig.from_cfg()
    native = _import_native()
    ds = register_dataset(df)
    row = native.evaluate_recipe_detailed(
        ds,
        strategy_id,
        list(map(float, genome)),
        **cfg.eval_kwargs(),
    )
    return _parse_detailed_row(row)


def evaluate_recipe_forward(
    df: pd.DataFrame,
    strategy_id: str,
    genome: Sequence[float],
    config: EngineConfig | None = None,
) -> dict[str, Any]:
    """Causal bar-drip forward: same shape as detailed + first_signal_mismatch_bar."""
    cfg = config if config is not None else EngineConfig.from_cfg()
    native = _import_native()
    ds = register_dataset(df)
    row = native.evaluate_recipe_forward(
        ds,
        strategy_id,
        list(map(float, genome)),
        **cfg.eval_kwargs(),
    )
    out = _parse_detailed_row(row)
    out["first_signal_mismatch_bar"] = row.get("first_signal_mismatch_bar")
    out["mode"] = row.get("mode", "forward")
    return out


def _parse_detailed_row(row: dict) -> dict[str, Any]:
    trades = list(row.get("trades") or [])
    # Native overwrites metrics "trades" count with the trade list.
    metrics_row = {
        "net_pnl": row.get("net_pnl", 0.0),
        "max_dd": row.get("max_dd", 0.0),
        "rdd": row.get("rdd", 0.0),
        "net_avg": row.get("net_avg", 0.0),
        "trades": len(trades),
        "stability": row.get("stability", 0.0),
        "recent_year_pnl": row.get("recent_year_pnl", 0.0),
        "half2_pnl": row.get("half2_pnl", 0.0),
        "sum_fees": row.get("sum_fees", 0.0),
        "avg_roundtrip_fee": row.get("avg_roundtrip_fee", row.get("sum_fees", 0.0)),
    }
    metrics = _row_to_metrics(metrics_row)
    equity = row.get("equity_curve")
    eq = np.asarray(equity, dtype=np.float64) if equity is not None else None
    return {"metrics": metrics, "trades": trades, "equity_curve": eq}
