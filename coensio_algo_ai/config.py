"""Load engine config TOML (sections: paths / ENGINE / GA_CONFIG / GA_FITNESS / SESSION / SWEEP / VALIDATION / REPORTING).

Default file: <repo>/engine_cfg.toml. Override via set_cfg_path() / CLI --cfg.
No code defaults: every required key must be present in the TOML.
# comment by coesnio, see https://coensio.com
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG_NAME = "engine_cfg.toml"
CFG_PATH = PROJECT_ROOT / DEFAULT_CFG_NAME
DEFAULT_BET_KEY = "default"

_cached: EngineCfg | None = None
_raw_cached: dict[str, Any] | None = None


def cfg_path() -> Path:
    return CFG_PATH


def set_cfg_path(path: str | Path, *, reload: bool = True) -> Path:
    """Select config file. Relative paths resolve against CWD first, then the repo root."""
    global CFG_PATH, _cached, _raw_cached
    p = Path(path)
    if not p.is_absolute():
        cwd_candidate = Path.cwd() / p
        p = cwd_candidate.resolve() if cwd_candidate.is_file() else (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"missing config: {p}")
    CFG_PATH = p
    if reload:
        _cached = None
        _raw_cached = None
    return CFG_PATH


def _require(section: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise KeyError(f"{CFG_PATH.name}: missing [{section_name}].{key}")
    return section[key]


@dataclass(frozen=True)
class EngineCfg:
    datasets_dir: Path
    strategies_dir: Path
    # ENGINE
    initial_capital: float
    fixed_bet_sizes: dict[str, float]
    bet_mode: str
    price_bet_frac: float
    commission_rate: float
    slippage_rate: float
    # GA_CONFIG
    min_num_trades: int
    num_generations: int
    restart_generations: int
    population_size: int
    cxpb: float
    mutpb: float
    tournament_size: int
    workers: int
    seed: int
    table_top_n: int
    workers_auto_max: int
    workers_cpu_reserve: int

    @property
    def fixed_bet_size(self) -> float:
        """Default bet (no dataset context); prefer resolve_fixed_bet_size(datafile)."""
        return resolve_fixed_bet_size(self.fixed_bet_sizes, None)


def _parse_fixed_bet_sizes(raw: Any) -> dict[str, float]:
    """Accept a single number, or a table with `default` plus optional per-dataset stems.

        fixed_bet_size = 2000.0

        [ENGINE.fixed_bet_size]
        default = 2000.0
        SPY_1h = 10000.0
    """
    if isinstance(raw, bool):
        raise TypeError(f"{CFG_PATH.name}: [ENGINE].fixed_bet_size must be a number or table")
    if isinstance(raw, (int, float)):
        return {DEFAULT_BET_KEY: float(raw)}
    if not isinstance(raw, dict) or not raw:
        raise TypeError(
            f"{CFG_PATH.name}: [ENGINE].fixed_bet_size must be a number or a table "
            f"with a `{DEFAULT_BET_KEY}` key"
        )
    out: dict[str, float] = {}
    for key, value in raw.items():
        v = float(value)
        if v <= 0:
            raise ValueError(f"{CFG_PATH.name}: fixed_bet_size {key} must be > 0")
        out[str(key)] = v
    if DEFAULT_BET_KEY not in out:
        raise KeyError(
            f"{CFG_PATH.name}: [ENGINE.fixed_bet_size] needs a `{DEFAULT_BET_KEY}` entry "
            "(per-dataset stems are optional overrides)"
        )
    return out


def resolve_fixed_bet_size(
    sizes: dict[str, float],
    datafile: str | Path | None,
) -> float:
    """Bet for a dataset: exact stem override if present, else `default`."""
    if not sizes:
        raise KeyError(f"{CFG_PATH.name}: [ENGINE].fixed_bet_size is empty")
    if datafile is not None and str(datafile).strip():
        stem = Path(str(datafile)).stem
        if stem in sizes:
            return float(sizes[stem])
    if DEFAULT_BET_KEY in sizes:
        return float(sizes[DEFAULT_BET_KEY])
    raise KeyError(f"{CFG_PATH.name}: [ENGINE.fixed_bet_size] has no `{DEFAULT_BET_KEY}`")


def load_raw_cfg(*, reload: bool = False) -> dict[str, Any]:
    global _raw_cached
    if _raw_cached is not None and not reload:
        return _raw_cached
    if not CFG_PATH.is_file():
        raise FileNotFoundError(f"missing config: {CFG_PATH}")
    with CFG_PATH.open("rb") as f:
        _raw_cached = tomllib.load(f)
    return _raw_cached


def load_engine_cfg(*, reload: bool = False) -> EngineCfg:
    global _cached
    if _cached is not None and not reload:
        return _cached
    raw = load_raw_cfg(reload=reload)
    paths = raw.get("paths") or {}
    engine = raw.get("ENGINE") or raw.get("engine") or {}
    ga = raw.get("GA_CONFIG") or raw.get("ga_config") or {}
    if not paths:
        raise KeyError(f"{CFG_PATH.name}: missing [paths]")
    if not engine:
        raise KeyError(f"{CFG_PATH.name}: missing [ENGINE]")
    if not ga:
        raise KeyError(f"{CFG_PATH.name}: missing [GA_CONFIG]")
    if not (raw.get("SESSION") or raw.get("session")):
        raise KeyError(f"{CFG_PATH.name}: missing [SESSION]")
    if not (raw.get("GA_FITNESS") or raw.get("ga_fitness")):
        raise KeyError(f"{CFG_PATH.name}: missing [GA_FITNESS]")
    if not (raw.get("REPORTING") or raw.get("reporting")):
        raise KeyError(f"{CFG_PATH.name}: missing [REPORTING]")

    ds = Path(_require(paths, "datasets", "paths"))
    st = Path(_require(paths, "strategies", "paths"))
    if not ds.is_absolute():
        ds = (PROJECT_ROOT / ds).resolve()
    if not st.is_absolute():
        st = (PROJECT_ROOT / st).resolve()

    # de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    cfg = EngineCfg(
        datasets_dir=ds,
        strategies_dir=st,
        initial_capital=float(_require(engine, "initial_capital", "ENGINE")),
        fixed_bet_sizes=_parse_fixed_bet_sizes(_require(engine, "fixed_bet_size", "ENGINE")),
        bet_mode=str(_require(engine, "bet_mode", "ENGINE")),
        price_bet_frac=float(_require(engine, "price_bet_frac", "ENGINE")),
        commission_rate=float(_require(engine, "commission_rate", "ENGINE")),
        slippage_rate=float(_require(engine, "slippage_rate", "ENGINE")),
        min_num_trades=int(_require(ga, "min_num_trades", "GA_CONFIG")),
        num_generations=int(_require(ga, "num_generations", "GA_CONFIG")),
        restart_generations=int(_require(ga, "restart_generations", "GA_CONFIG")),
        population_size=int(_require(ga, "population_size", "GA_CONFIG")),
        cxpb=float(_require(ga, "cxpb", "GA_CONFIG")),
        mutpb=float(_require(ga, "mutpb", "GA_CONFIG")),
        tournament_size=int(_require(ga, "tournament_size", "GA_CONFIG")),
        workers=int(_require(ga, "workers", "GA_CONFIG")),
        seed=int(_require(ga, "seed", "GA_CONFIG")),
        table_top_n=int(_require(ga, "table_top_n", "GA_CONFIG")),
        workers_auto_max=int(_require(ga, "workers_auto_max", "GA_CONFIG")),
        workers_cpu_reserve=int(_require(ga, "workers_cpu_reserve", "GA_CONFIG")),
    )

    from coensio_algo_ai.sessions import load_chart_session_presets, load_session_presets, session_default

    session_default(raw)
    load_session_presets(raw)
    load_chart_session_presets(raw)

    _cached = cfg
    return cfg


@dataclass(frozen=True)
class SweepCfg:
    tickers: tuple[str, ...]
    sessions: tuple[str, ...]
    cycles: int
    table_top_n: int


def load_sweep_cfg(*, reload: bool = False) -> SweepCfg:
    raw = load_raw_cfg(reload=reload)
    section = raw.get("SWEEP") or raw.get("sweep")
    if not section:
        raise KeyError(f"{CFG_PATH.name}: missing [SWEEP]")
    tickers_raw = _require(section, "sweep_tickers", "SWEEP")
    if isinstance(tickers_raw, str):
        tickers = tuple(t.strip() for t in tickers_raw.split(",") if t.strip())
    elif isinstance(tickers_raw, (list, tuple)):
        tickers = tuple(str(t).strip() for t in tickers_raw if str(t).strip())
    else:
        raise TypeError(
            f"{CFG_PATH.name}: [SWEEP].sweep_tickers must be an array of stems"
        )
    if not tickers:
        raise ValueError(f"{CFG_PATH.name}: [SWEEP].sweep_tickers is empty")

    sessions_raw = section.get("sweep_sessions")
    if sessions_raw is None:
        sessions = ("none", "new_york", "london", "asia")
    elif isinstance(sessions_raw, str):
        sessions = tuple(s.strip() for s in sessions_raw.split(",") if s.strip())
    elif isinstance(sessions_raw, (list, tuple)):
        sessions = tuple(str(s).strip() for s in sessions_raw if str(s).strip())
    else:
        raise TypeError(
            f"{CFG_PATH.name}: [SWEEP].sweep_sessions must be an array"
        )
    if not sessions:
        raise ValueError(f"{CFG_PATH.name}: [SWEEP].sweep_sessions is empty")

    cycles = int(_require(section, "sweep_cycles", "SWEEP"))
    if cycles < 1:
        raise ValueError(f"{CFG_PATH.name}: [SWEEP].sweep_cycles must be >= 1")
    table_top_n = int(_require(section, "sweep_table_top_n", "SWEEP"))
    if table_top_n < 1:
        raise ValueError(f"{CFG_PATH.name}: [SWEEP].sweep_table_top_n must be >= 1")
    return SweepCfg(
        tickers=tickers,
        sessions=sessions,
        cycles=cycles,
        table_top_n=table_top_n,
    )


@dataclass(frozen=True)
class ValidationCfg:
    n_perm: int
    alpha: float
    metric: str
    seed: int


def load_validation_cfg(*, reload: bool = False) -> ValidationCfg:
    raw = load_raw_cfg(reload=reload)
    section = raw.get("VALIDATION") or raw.get("validation")
    if not section:
        raise KeyError(f"{CFG_PATH.name}: missing [VALIDATION]")
    n_perm = int(_require(section, "n_perm", "VALIDATION"))
    if n_perm < 1:
        raise ValueError(f"{CFG_PATH.name}: [VALIDATION].n_perm must be >= 1")
    alpha = float(_require(section, "alpha", "VALIDATION"))
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"{CFG_PATH.name}: [VALIDATION].alpha must be in (0, 1)")
    metric = str(_require(section, "metric", "VALIDATION")).strip()
    if not metric:
        raise ValueError(f"{CFG_PATH.name}: [VALIDATION].metric is empty")
    seed = int(_require(section, "seed", "VALIDATION"))
    return ValidationCfg(
        n_perm=n_perm,
        alpha=alpha,
        metric=metric,
        seed=seed,
    )
