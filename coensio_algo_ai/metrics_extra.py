"""Extended backtest metrics for reports."""

from __future__ import annotations

import math

from coensio_algo_ai.progress import metrics_to_fitness_dict
from coensio_algo_ai.fills import Metrics


def extended_metrics(
    compact: Metrics | dict,
    trades: list[dict],
    equity_curve,
    *,
    initial_capital: float,
    timestamps=None,
) -> dict:
    if isinstance(compact, Metrics):
        base = metrics_to_fitness_dict(compact)
        sum_fees = float(compact.avg_roundtrip_fee)
    else:
        row = {
            "rdd": compact.get("rdd", 0.0),
            "net_pnl": compact.get("net_pnl", 0.0),
            "max_dd": compact.get("max_dd", 0.0),
            "net_avg": compact.get("net_avg", 0.0),
            "trades": len(trades),
            "stability": compact.get("stability", 0.0),
            "recent_year_pnl": compact.get("recent_year_pnl", 0.0),
            "half2_pnl": compact.get("half2_pnl", 0.0),
            "avg_roundtrip_fee": compact.get(
                "avg_roundtrip_fee", compact.get("sum_fees", 0.0)
            ),
        }
        m = Metrics(
            net_pnl=float(row["net_pnl"]),
            max_dd=float(row["max_dd"]),
            rdd=float(row["rdd"]),
            trades=int(row["trades"]),
            net_avg=float(row["net_avg"]),
            stability=float(row["stability"]),
            recent_year_pnl=float(row["recent_year_pnl"]),
            half2_pnl=float(row["half2_pnl"]),
            avg_roundtrip_fee=float(row["avg_roundtrip_fee"]),
        )
        base = metrics_to_fitness_dict(m)
        sum_fees = float(row["avg_roundtrip_fee"])

    eq = list(equity_curve) if equity_curve is not None else []
    n = len(trades)
    pnls = [float(t.get("net_pnl", 0.0)) for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    win_rate = 100.0 * wins / n if n else 0.0
    profit_factor = (
        gross_wins / gross_losses if gross_losses > 0 else (9.99 if gross_wins > 0 else 0.0)
    )

    peak = initial_capital
    max_dd_pct = 0.0
    for val in eq:
        if val > peak:
            peak = val
        if peak > 0:
            dd_pct = 100.0 * (peak - val) / peak
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

    total_profit = float(base.get("total_profit", 0.0))
    total_return_pct = 100.0 * total_profit / initial_capital if initial_capital else 0.0
    total_commission = sum(
        float(t.get("entry_commission", 0.0)) + float(t.get("exit_commission", 0.0))
        for t in trades
    )
    if trades and not any("entry_commission" in t or "exit_commission" in t for t in trades):
        total_commission = float(base.get("total_fees", sum_fees * n))
    total_slippage = sum(float(t.get("slippage_cost", 0.0)) for t in trades)
    ts_list = list(timestamps) if timestamps is not None else None
    sharpe, sortino = _equity_sharpe_sortino(eq, timestamps=ts_list)

    out = dict(base)
    out.update(
        {
            "total_return_pct": round(total_return_pct, 2),
            "max_dd_pct": round(max_dd_pct, 2),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "total_fees": round(total_commission, 2),
            "total_slippage": round(total_slippage, 2),
            "total_transaction_cost": round(total_commission + total_slippage, 2),
        }
    )
    return out


def _timestamp_seconds(value: float) -> float:
    value = float(value)
    magnitude = abs(value)
    if magnitude >= 1e18:
        return value / 1e9
    if magnitude >= 1e15:
        return value / 1e6
    if magnitude >= 1e12:
        return value / 1e3
    return value


def _periods_per_year(timestamps: list[float] | None) -> float:
    if timestamps is None or len(timestamps) < 2:
        return 1.0
    seconds = [_timestamp_seconds(value) for value in timestamps]
    elapsed = seconds[-1] - seconds[0]
    if elapsed <= 0:
        return 1.0
    return ((len(seconds) - 1) * 365.25 * 24.0 * 60.0 * 60.0) / elapsed


def _equity_sharpe_sortino(
    equity_curve: list[float],
    *,
    timestamps: list[float] | None = None,
) -> tuple[float, float]:
    if len(equity_curve) < 3:
        return 0.0, 0.0
    rets = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        if prev == 0:
            continue
        rets.append((equity_curve[i] - prev) / prev)
    if len(rets) < 2:
        return 0.0, 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    annualizer = math.sqrt(_periods_per_year(timestamps))
    sharpe = (mean / std) * annualizer if std > 0 else 0.0
    downside_deviation = math.sqrt(sum(min(r, 0.0) ** 2 for r in rets) / len(rets))
    sortino = (mean / downside_deviation) * annualizer if downside_deviation > 0 else 0.0
    return sharpe, sortino
