"""Parallel batch evaluation thread count (Rust Rayon)."""

from __future__ import annotations

import os

from coensio_algo_ai.config import load_engine_cfg


def resolve_ga_workers(requested: int, n_bars: int) -> int:
    """Resolve Rayon thread count. requested<=0 means auto from engine_cfg.toml."""
    del n_bars  # unused; kept for call-site compatibility
    cfg = load_engine_cfg()
    if requested > 0:
        return max(1, int(requested))

    cpu = os.cpu_count() or 2
    reserve = max(0, int(cfg.workers_cpu_reserve))
    return min(int(cfg.workers_auto_max), max(1, cpu - reserve))


def configure_rayon_workers(requested: int, n_bars: int) -> int:
    workers = resolve_ga_workers(requested, n_bars)
    # de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    os.environ["RAYON_NUM_THREADS"] = str(workers)
    return workers


def ensure_rayon_before_native(requested: int, n_bars: int) -> int:
    """Call before first native import/evaluate so Rayon pool picks up thread count."""
    return configure_rayon_workers(requested, n_bars)
