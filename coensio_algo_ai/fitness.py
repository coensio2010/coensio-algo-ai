"""Configurable GA fitness.

# comment by coesnio
"""

from __future__ import annotations

_ALIASES = {
    "stab": "stability_r2",
    "stability": "stability_r2",
    "trades": "total_trades",
    "rdd": "ret_dd_ratio",
    "pf": "profit_factor",
    "dd": "max_dd_usd",
    "pnl": "total_profit",
}


def configured_min_trades(config: dict | None) -> int:
    from coensio_algo_ai.config import load_raw_cfg

    cfg = config if config is not None else load_raw_cfg()
    return int(_require_ga(cfg, "min_num_trades"))


def _require_ga(config: dict, key: str):
    ga = config.get("GA_CONFIG") or {}
    if key not in ga:
        raise KeyError(f"engine_cfg.toml: missing [GA_CONFIG].{key}")
    return ga[key]


def _default_cap(metric, initial_capital):
    if metric in ("max_dd_usd", "total_profit"):
        return float(initial_capital)
    caps = {
        "stability_r2": 1.0,
        "win_rate": 100.0,
        "ret_dd_ratio": 20.0,
        "total_trades": 500.0,
        "profit_factor": 5.0,
        "sharpe": 3.0,
        "sortino": 3.0,
        "max_dd_pct": 100.0,
        "avg_trade_profit": 100.0,
        "total_return_pct": 1000.0,
        "recent_year_pnl": 5000.0,
        "half2_pnl": 5000.0,
    }
    return caps.get(metric, 1.0)


def parse_fitness_config(ga_fitness_cfg):
    specs = []
    legacy = {}
    for name, val in ga_fitness_cfg.items():
        if isinstance(val, dict):
            metric = _ALIASES.get(name, name)
            specs.append(
                {
                    "metric": metric,
                    "direction": float(val.get("direction", 1.0)),
                    "weight": float(val.get("weight", 1.0)),
                    "cap": float(val["cap"]) if "cap" in val else None,
                    "gate": float(val["gate"]) if "gate" in val else None,
                    "uncapped": bool(val.get("uncapped", False)),
                }
            )
        else:
            legacy[name] = val

    if not specs and legacy:
        specs = [
            {
                "metric": "ret_dd_ratio",
                "direction": 1.0,
                "weight": float(legacy.get("w_rdd", 1.0)),
                "cap": None,
            },
            {
                "metric": "stability_r2",
                "direction": 1.0,
                "weight": float(legacy.get("w_stability", 8.0)),
                "cap": 1.0,
            },
            {
                "metric": "sharpe",
                "direction": 1.0,
                "weight": float(legacy.get("w_sharpe", 0.5)),
                "cap": None,
            },
        ]
    return specs


def load_fitness_specs(config: dict | None = None):
    from coensio_algo_ai.config import load_raw_cfg

    cfg = config if config is not None else load_raw_cfg()
    section = cfg.get("GA_FITNESS") or cfg.get("ga_fitness")
    if not section:
        raise KeyError("engine_cfg.toml: missing [GA_FITNESS]")
    return parse_fitness_config(section)


def composite_fitness(metrics, specs, min_trades, initial_capital=10000.0):
    if metrics is None or metrics.get("total_trades", 0) < min_trades:
        return -1e6

    rdd = float(metrics.get("ret_dd_ratio", 0))
    pnl = float(metrics.get("total_profit", 0))
    # de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    if rdd <= 0 or pnl <= 0:
        return -1000.0 + rdd

    worst_shortfall = 0.0
    for s in specs:
        gate = s.get("gate")
        if gate is None:
            continue
        value = float(metrics.get(s["metric"], 0.0))
        cap = s.get("cap") or _default_cap(s["metric"], initial_capital)
        if cap <= 0:
            cap = 1.0
        shortfall = (gate - value) / cap if s["direction"] >= 0 else (value - gate) / cap
        if shortfall > worst_shortfall:
            worst_shortfall = shortfall
    if worst_shortfall > 0.0:
        return -worst_shortfall

    score = 0.0
    for s in specs:
        value = float(metrics.get(s["metric"], 0.0))
        if s.get("uncapped", False):
            score += s["direction"] * s["weight"] * value
            continue
        cap = s.get("cap")
        if cap is None:
            cap = _default_cap(s["metric"], initial_capital)
        if cap <= 0:
            cap = 1.0
        norm = value / cap
        if norm > 1.0:
            norm = 1.0
        elif norm < 0.0:
            norm = 0.0
        score += s["direction"] * s["weight"] * norm
    return score
