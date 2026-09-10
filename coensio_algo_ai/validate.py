"""Full-data Monte Carlo bar permutation (MCPT) validation.

Fixed genome (no re-optimize inside perms):
1. Score genome on 100% of data
2. Shuffle bars n_perm times, score same genome each time
3. p-value + z-score vs permutation null
4. Pass if p < alpha

powered by ceonsio.com

Usage:
  python -m coensio_algo_ai validate --strategy BS1_breakout --file BTC_1h.parquet --genome "..." --sessions london
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from coensio_algo_ai.bar_permute import permute_ohlcv
from coensio_algo_ai.fills import EngineConfig, Metrics, evaluate_batch, register_dataset


@dataclass
class ValidateResult:
    strategy: str
    genome: str
    bars: int
    metric: str
    real_metric: float
    real_pnl: float
    real_trades: int
    p_value: float
    z_score: float
    perm_mean: float
    perm_std: float
    n_perm: int
    alpha: float
    passed: bool
    fail_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metric_from_metrics(m: Metrics, metric: str) -> float:
    aliases = {
        "rdd": "rdd",
        "ret_dd_ratio": "rdd",
        "pnl": "net_pnl",
        "total_profit": "net_pnl",
        "net_pnl": "net_pnl",
        "stability": "stability",
        "stability_r2": "stability",
    }
    key = aliases.get(metric, metric)
    if key == "rdd":
        return float(m.rdd)
    if key == "net_pnl":
        return float(m.net_pnl)
    if key == "stability":
        return float(m.stability)
    return float(getattr(m, key, 0.0) or 0.0)


def eval_genome_df(
    df: pd.DataFrame,
    *,
    strategy_id: str,
    genome_values: list[float],
    engine: EngineConfig,
) -> Metrics:
    """Backtest one genome on a dataframe via native evaluate_batch."""
    ds = register_dataset(df)
    return evaluate_batch(ds, strategy_id, [genome_values], config=engine)[0]


def mcpt_null(
    df: pd.DataFrame,
    *,
    strategy_id: str,
    genome_values: list[float],
    engine: EngineConfig,
    real_metric: float,
    metric: str = "ret_dd_ratio",
    n_perm: int = 1000,
    seed: int = 42,
    start_index: int = 0,
) -> tuple[float, float, float, float]:
    """Run MCPT null. Returns (p_value, z_score, perm_mean, perm_std)."""
    rng = np.random.default_rng(seed)
    perm_scores = np.empty(n_perm, dtype=np.float64)
    n_better = 0
    for i in range(n_perm):
        perm = permute_ohlcv(
            df, start_index=start_index, seed=int(rng.integers(0, 2**31 - 1))
        )
        row = eval_genome_df(
            perm,
            strategy_id=strategy_id,
            genome_values=genome_values,
            engine=engine,
        )
        m = _metric_from_metrics(row, metric)
        perm_scores[i] = m
        if m >= real_metric:
            n_better += 1
        done = i + 1
        if done % 100 == 0 or done == n_perm:
            print(
                f"  MCPT {done}/{n_perm}  running_p~={(1 + n_better) / (done + 1):.4f}",
                flush=True,
            )

    p_value = (1 + n_better) / (n_perm + 1)
    perm_mean = float(np.mean(perm_scores))
    perm_std = float(np.std(perm_scores, ddof=1)) if n_perm > 1 else 0.0
    if perm_std > 0.0:
        z_score = (real_metric - perm_mean) / perm_std
    else:
        z_score = float("inf") if real_metric > perm_mean else 0.0
    return p_value, float(z_score), perm_mean, perm_std


def validate_genome(
    df: pd.DataFrame,
    *,
    strategy_id: str,
    genome_str: str,
    genome_values: list[float],
    engine: EngineConfig,
    metric: str = "ret_dd_ratio",
    n_perm: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ValidateResult:
    real_row = eval_genome_df(
        df, strategy_id=strategy_id, genome_values=genome_values, engine=engine
    )
    real_metric = _metric_from_metrics(real_row, metric)
    real_pnl = _metric_from_metrics(real_row, "pnl")
    real_trades = int(real_row.trades)
    # de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee

    print(
        f"Real: metric={real_metric:.4f} pnl=${real_pnl:.2f} trades={real_trades} "
        f"bars={len(df)}",
        flush=True,
    )
    print(f"MCPT on full data ({n_perm} perms, metric={metric})...", flush=True)
    p_value, z_score, perm_mean, perm_std = mcpt_null(
        df,
        strategy_id=strategy_id,
        genome_values=genome_values,
        engine=engine,
        real_metric=real_metric,
        metric=metric,
        n_perm=n_perm,
        seed=seed,
    )

    reasons: list[str] = []
    if p_value >= alpha:
        reasons.append(f"MCPT p={p_value:.4f} >= alpha={alpha}")

    return ValidateResult(
        strategy=strategy_id,
        genome=genome_str,
        bars=len(df),
        metric=metric,
        real_metric=real_metric,
        real_pnl=real_pnl,
        real_trades=real_trades,
        p_value=p_value,
        z_score=z_score,
        perm_mean=perm_mean,
        perm_std=perm_std,
        n_perm=n_perm,
        alpha=alpha,
        passed=not reasons,
        fail_reasons=reasons,
    )
