"""Format / parse human genome pipe strings via strategies/<id>/genome_fmt.toml.

Integer vs float rendering follows the param `type` in recipe.toml.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from typing import Any

from coensio_algo_ai.strategy import load_recipe_params, strategy_dir

TEMPLATE_FIELD_RE = re.compile(r"\{(\w+)(?::[^}]*)?\}")
FIXED_TAIL_FIELDS = ("bet_mode", "fixed_bet_size", "price_bet_frac", "session")


def _load_fmt_template(strategy_id: str) -> tuple[str, set[str]] | None:
    path = strategy_dir(strategy_id) / "genome_fmt.toml"
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    template = str(raw.get("template", "")).strip()
    if not template:
        return None
    integer = {p["name"] for p in load_recipe_params(strategy_id) if p["type"] == "int"}
    return template, integer


def template_fields(template: str) -> list[str]:
    return TEMPLATE_FIELD_RE.findall(template)


def check_genome_template(strategy_id: str) -> list[str]:
    """Consistency problems between genome_fmt.toml and recipe.toml (empty = OK)."""
    problems: list[str] = []
    loaded = _load_fmt_template(strategy_id)
    if loaded is None:
        return [f"{strategy_id}: genome_fmt.toml missing or has no template"]
    template, _ = loaded
    fields = template_fields(template)
    params = [p["name"] for p in load_recipe_params(strategy_id)]
    parts = template.split("|")
    if not parts or parts[0] != strategy_id:
        problems.append(f"template must start with the literal {strategy_id!r}|")
    if len(fields) < 2 or fields[0] != "direction":
        problems.append("second template field must be {direction}")
    param_fields = [f for f in fields if f not in ("direction", *FIXED_TAIL_FIELDS)]
    if param_fields != params:
        problems.append(
            f"template params {param_fields} must equal recipe [[params]] order {params}"
        )
    tail = tuple(fields[-4:])
    if tail != FIXED_TAIL_FIELDS:
        problems.append(
            "template must end with {bet_mode}|{fixed_bet_size}|{price_bet_frac}|{session}"
        )
    return problems


def plugin_direction(strategy_id: str) -> str:
    path = strategy_dir(strategy_id) / "recipe.toml"
    if not path.is_file():
        return "long"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return str((raw.get("plugin") or {}).get("direction", "long"))


@dataclass(frozen=True)
class ParsedGenome:
    fields: dict[str, str]
    params: dict[str, Any]
    genome: list[float]
    session: str
    bet_mode: str
    fixed_bet_size: float
    price_bet_frac: float


def genome_template(strategy_id: str) -> str | None:
    loaded = _load_fmt_template(strategy_id)
    return loaded[0] if loaded is not None else None


def format_genome_str(
    strategy_id: str,
    params_spec: list[dict],
    params: dict[str, Any],
    *,
    session: str = "none",
    bet_mode: str = "fixed",
    fixed_bet_size: float = 2000.0,
    price_bet_frac: float = 0.0,
    direction: str | None = None,
) -> str:
    """Display/report genome string (strategy|direction|params|exit|bet|session)."""
    g: dict[str, Any] = {p["name"]: params[p["name"]] for p in params_spec}
    g["session"] = session
    g["bet_mode"] = bet_mode
    g["fixed_bet_size"] = fixed_bet_size
    g["price_bet_frac"] = price_bet_frac
    g["direction"] = direction if direction is not None else plugin_direction(strategy_id)

    loaded = _load_fmt_template(strategy_id)
    if loaded is not None:
        template, integer = loaded
        fmt = dict(g)
        for name in integer:
            if name in fmt:
                fmt[name] = int(fmt[name])
        return template.format(**fmt)

    parts = "|".join(f"{p['name']}={params[p['name']]:g}" for p in params_spec)
    return (
        f"{strategy_id}|{parts}|sess={session}|bet={bet_mode}|"
        f"{fixed_bet_size}|{price_bet_frac}"
    )


def parse_genome_str(strategy_id: str, genome_str: str, params_spec: list[dict]) -> ParsedGenome:
    """Parse a full template genome, or a short param-only `a|b|...` matching recipe params."""
    from coensio_algo_ai import genome as G

    s = genome_str.strip()
    loaded = _load_fmt_template(strategy_id)
    short_parts = [p.strip() for p in s.split("|")]

    # Short param-only genome (quick CLI form).
    if loaded is None or len(short_parts) == len(params_spec):
        params = G.decode(params_spec, s)
        return ParsedGenome(
            fields={k: str(v) for k, v in params.items()},
            params=params,
            genome=[float(params[p["name"]]) for p in params_spec],
            session="none",
            bet_mode="fixed",
            fixed_bet_size=2000.0,
            price_bet_frac=0.0,
        )

    template, _integer = loaded
    t_parts = template.split("|")
    g_parts = short_parts
    if len(t_parts) != len(g_parts):
        raise ValueError(
            f"genome has {len(g_parts)} fields, expected {len(t_parts)} for {strategy_id} "
            f"(or {len(params_spec)} param-only fields)"
        )

    fields: dict[str, str] = {}
    for t_part, g_part in zip(t_parts, g_parts):
        m = TEMPLATE_FIELD_RE.fullmatch(t_part)
        if m:
            fields[m.group(1)] = g_part
        elif t_part != g_part:
            raise ValueError(f"genome literal mismatch: expected {t_part!r}, got {g_part!r}")

    params: dict[str, Any] = {}
    genome: list[float] = []
    for p in params_spec:
        name = p["name"]
        if name not in fields:
            raise ValueError(f"genome missing param {name!r}")
        raw = fields[name]
        if p.get("type") == "float":
            val: Any = float(raw)
        else:
            val = int(float(raw))
        params[name] = val
        genome.append(float(val))

    return ParsedGenome(
        fields=fields,
        params=params,
        genome=genome,
        session=str(fields.get("session", "none")),
        bet_mode=str(fields.get("bet_mode", "fixed")),
        fixed_bet_size=float(fields.get("fixed_bet_size", 2000.0)),
        price_bet_frac=float(fields.get("price_bet_frac", 0.0)),
    )
