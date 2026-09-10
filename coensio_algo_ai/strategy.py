"""Strategy folders: strategies/<id>/{recipe.toml, genome_fmt.toml, rust/mod.rs}.

coensoi strategy loader
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from coensio_algo_ai.config import load_engine_cfg

PARAM_TYPES = ("int", "float")


def strategies_dir() -> Path:
    return load_engine_cfg().strategies_dir


def list_strategies() -> list[str]:
    root = strategies_dir()
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name.startswith("_"):
            continue
        if (p / "recipe.toml").is_file() and (p / "rust" / "mod.rs").is_file():
            out.append(p.name)
    return out


def strategy_dir(strategy_id: str) -> Path:
    path = strategies_dir() / strategy_id
    if not path.is_dir():
        known = ", ".join(list_strategies()) or "(none)"
        raise FileNotFoundError(
            f"strategy folder not found: {path} (have: {known})"
        )
    return path


def load_recipe_raw(strategy_id: str) -> dict[str, Any]:
    path = strategy_dir(strategy_id) / "recipe.toml"
    with path.open("rb") as f:
        return tomllib.load(f)


def load_recipe_params(strategy_id: str) -> list[dict[str, Any]]:
    """Load [[params]]. Every param needs name, type ("int" | "float"), min, max.

    `type` is the single source of truth for GA sampling and genome formatting.
    It must match how the Rust plugin reads the value (param_i32 vs param_f64).
    Optional keys: step (GA grid step), default.
    """
    raw = load_recipe_raw(strategy_id)
    params: list[dict[str, Any]] = []
    seen: set[str] = set()
    for p in raw.get("params", []):
        for key in ("name", "type", "min", "max"):
            if key not in p:
                raise KeyError(
                    f"{strategy_id}/recipe.toml: every [[params]] needs name, type, min, max "
                    f"(missing {key!r} in {p.get('name', '?')})"
                )
        name = str(p["name"])
        if name in seen:
            raise ValueError(f"{strategy_id}/recipe.toml: duplicate param {name!r}")
        seen.add(name)
        typ = str(p["type"]).lower()
        if typ not in PARAM_TYPES:
            raise ValueError(
                f"{strategy_id}/recipe.toml: param {name!r} type must be int or float, got {typ!r}"
            )
        start = float(p["min"])
        end = float(p["max"])
        if start > end:
            raise ValueError(f"{strategy_id}/recipe.toml: param {name!r} min > max")
        default = p["default"] if "default" in p else start
        if typ == "int":
            if start != int(start) or end != int(end):
                raise ValueError(
                    f"{strategy_id}/recipe.toml: int param {name!r} needs integer min/max"
                )
            step = int(p.get("step", 1))
            params.append(
                {
                    "name": name,
                    "type": "int",
                    "start": int(start),
                    "end": int(end),
                    "step": max(1, step),
                    "default": int(default),
                }
            )
        else:
            step = float(p.get("step", max((end - start) / 100.0, 1e-6)))
            params.append(
                {
                    "name": name,
                    "type": "float",
                    "start": start,
                    "end": end,
                    "step": step,
                    "default": float(default),
                }
            )
    if not params:
        raise ValueError(f"{strategy_id}: recipe.toml has no [[params]]")
    return params


class StrategySpec:
    """Recipe-backed strategy (Rust plugin evaluated via evaluate_batch)."""

    def __init__(self, strategy_id: str):
        self.STRATEGY_ID = strategy_id
        self.PARAMS = load_recipe_params(strategy_id)


def load_strategy(strategy_id: str) -> StrategySpec:
    rust = strategy_dir(strategy_id) / "rust" / "mod.rs"
    if not rust.is_file():
        raise FileNotFoundError(f"missing Rust plugin: {rust}")
    return StrategySpec(strategy_id)


def params_to_genome_vec(params_spec: list[dict], params: dict[str, Any]) -> list[float]:
    return [float(params[p["name"]]) for p in params_spec]
