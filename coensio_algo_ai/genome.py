"""Genome encode/decode + GA operators.

# comment by coesnio
"""

from __future__ import annotations

from typing import Any


def param_names(params_spec: list[dict]) -> list[str]:
    return [p["name"] for p in params_spec]


def encode(params_spec: list[dict], values: dict[str, Any]) -> str:
    parts = []
    for p in params_spec:
        name = p["name"]
        if name not in values:
            raise KeyError(f"missing param {name}")
        v = values[name]
        if p.get("type") == "float":
            parts.append(f"{float(v):.6g}")
        else:
            parts.append(str(int(v) if p.get("type", "int") == "int" else v))
    return "|".join(parts)


def decode(params_spec: list[dict], genome: str) -> dict[str, Any]:
    parts = [x.strip() for x in genome.split("|")]
    if len(parts) != len(params_spec):
        raise ValueError(
            f"genome fields={len(parts)} expected={len(params_spec)} "
            f"({', '.join(param_names(params_spec))})"
        )
    out: dict[str, Any] = {}
    for p, raw in zip(params_spec, parts, strict=True):
        typ = p.get("type", "int")
        if typ == "float":
            out[p["name"]] = float(raw)
        elif typ == "bool":
            out[p["name"]] = raw in ("1", "true", "True", "yes")
        else:
            out[p["name"]] = int(float(raw))
    return out


def expand_range(p: dict) -> list[Any]:
    typ = p.get("type", "int")
    start, end, step = p["start"], p["end"], p.get("step", 1)
    if typ == "float":
        vals = []
        x = float(start)
        end_f = float(end)
        step_f = float(step)
        while x <= end_f + 1e-12:
            vals.append(round(x, 10))
            x += step_f
        return vals
    return list(range(int(start), int(end) + 1, int(step)))


def clip_params(params_spec: list[dict], values: dict[str, Any]) -> dict[str, Any]:
    """clip_genome: clamp to [min,max], int round / float 3dp."""
    out: dict[str, Any] = {}
    for p in params_spec:
        name = p["name"]
        lo, hi = float(p["start"]), float(p["end"])
        v = float(values[name])
        v = max(lo, min(hi, v))
        if p.get("type", "int") == "int":
            out[name] = int(round(v))
        else:
            out[name] = round(v, 3)
    return out


def random_params(params_spec: list[dict], rng) -> dict[str, Any]:
    """random_genome: int uniform inclusive, float continuous uniform."""
    out: dict[str, Any] = {}
    for p in params_spec:
        lo, hi = float(p["start"]), float(p["end"])
        if p.get("type", "int") == "int":
            out[p["name"]] = int(rng.integers(int(lo), int(hi) + 1))
        else:
            out[p["name"]] = float(rng.uniform(lo, hi))
    return clip_params(params_spec, out)


def mutate_params(params_spec: list[dict], values: dict[str, Any], rng) -> dict[str, Any]:
    """mutate_genome: perturb ONE random gene."""
    out = dict(values)
    idx = int(rng.integers(0, len(params_spec)))
    p = params_spec[idx]
    name = p["name"]
    lo, hi = float(p["start"]), float(p["end"])
    if p.get("type", "int") == "int":
        step = max(1, int((hi - lo) * 0.2))
        # de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
        out[name] = int(out[name]) + int(rng.integers(-step, step + 1))
    else:
        span = hi - lo
        out[name] = float(out[name]) + float(rng.uniform(-0.2 * span, 0.2 * span))
    return clip_params(params_spec, out)


def crossover(
    params_spec: list[dict],
    a: dict[str, Any],
    b: dict[str, Any],
    rng,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """crossover_genomes: single-point swap, returns two children."""
    names = [p["name"] for p in params_spec]
    if len(names) < 2:
        return dict(a), dict(b)
    point = int(rng.integers(1, len(names)))
    c1 = {n: (a[n] if i < point else b[n]) for i, n in enumerate(names)}
    c2 = {n: (b[n] if i < point else a[n]) for i, n in enumerate(names)}
    return clip_params(params_spec, c1), clip_params(params_spec, c2)
