"""Trading session chart construction.

copyright coesnio.com
"""

from __future__ import annotations

from typing import Any

from coensio_algo_ai.config import cfg_path, load_raw_cfg

SESSION_ALIASES = {
    "ny": "new_york",
    "newyork": "new_york",
    "us": "new_york",
    "uk": "london",
    "tokyo": "asia",
    "jp": "asia",
}


def _parse_hm(value: Any, default: tuple[int, int] | None = None) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (int(value[0]), int(value[1]))
    if default is not None:
        return default
    raise ValueError(f"{cfg_path().name}: bad HH:MM value {value!r}")


def normalize_session_name(name: str) -> str:
    key = str(name or "none").lower().replace(" ", "_")
    return SESSION_ALIASES.get(key, key)


def _session_section(config: dict | None = None) -> dict:
    raw = config if config is not None else load_raw_cfg()
    section = raw.get("SESSION") or raw.get("session")
    if not section:
        raise KeyError(f"{cfg_path().name}: missing [SESSION]")
    return section


def load_session_presets(config: dict | None = None) -> dict[str, dict]:
    section = _session_section(config)
    raw = section.get("PRESETS") or section.get("presets")
    if not raw:
        raise KeyError(f"{cfg_path().name}: missing [SESSION.PRESETS.*]")
    presets: dict[str, dict] = {}
    for name, spec in raw.items():
        key = normalize_session_name(name)
        if "session_start" not in spec or "session_end" not in spec or "session_tz" not in spec:
            raise KeyError(
                f"{cfg_path().name}: [SESSION.PRESETS.{name}] needs "
                "session_start, session_end, session_tz"
            )
        presets[key] = {
            "session_start": _parse_hm(spec["session_start"]),
            "session_end": _parse_hm(spec["session_end"]),
            "session_tz": str(spec["session_tz"]),
        }
    return presets


def load_chart_session_presets(config: dict | None = None) -> dict[str, dict]:
    section = _session_section(config)
    raw = section.get("UTC_CHART") or section.get("utc_chart")
    if not raw:
        raise KeyError(f"{cfg_path().name}: missing [SESSION.UTC_CHART.*]")
    presets: dict[str, dict] = {}
    for name, spec in raw.items():
        key = normalize_session_name(name)
        if "utc_start" not in spec or "utc_end" not in spec:
            raise KeyError(
                f"{cfg_path().name}: [SESSION.UTC_CHART.{name}] needs utc_start, utc_end"
            )
        presets[key] = {
            "utc_start": _parse_hm(spec["utc_start"]),
            "utc_end": _parse_hm(spec["utc_end"]),
        }
    return presets


def session_default(config: dict | None = None) -> str:
    section = _session_section(config)
    if "default" not in section:
        raise KeyError(f"{cfg_path().name}: missing [SESSION].default")
    return normalize_session_name(str(section["default"]))


def session_mode_default(config: dict | None = None) -> str:
    section = _session_section(config)
    mode = str(section.get("mode", "wall")).lower()
    if mode not in ("wall", "utc"):
        raise ValueError(f"{cfg_path().name}: [SESSION].mode must be wall|utc, got {mode!r}")
    return mode


def get_session_preset(name: str, config: dict | None = None) -> dict | None:
    key = normalize_session_name(name)
    if key in ("none", "all", "24_7", "24x7", ""):
        return None
    presets = load_session_presets(config)
    if key not in presets:
        raise ValueError(
            f"unknown session preset {name!r}; have: {', '.join(sorted(presets))}"
        )
    return presets[key]


def get_chart_session_preset(name: str, config: dict | None = None) -> dict | None:
    key = normalize_session_name(name)
    if key in ("none", "all", "24_7", "24x7", ""):
        return None
    presets = load_chart_session_presets(config)
    if key not in presets:
        raise ValueError(
            f"unknown UTC chart session {name!r}; have: {', '.join(sorted(presets))}"
        )
    return presets[key]


def infer_ohlc_source_tz(df, config: dict | None = None) -> str:
    section = _session_section(config)
    if "data_tz" not in section:
        raise KeyError(f"{cfg_path().name}: missing [SESSION].data_tz")
    if "data_tz_fallback" not in section:
        raise KeyError(f"{cfg_path().name}: missing [SESSION].data_tz_fallback")
    explicit = section["data_tz"]
    if explicit and str(explicit).lower() not in ("auto", ""):
        return str(explicit)
    if df is not None:
        idx = df.index
        if getattr(idx, "tz", None) is not None:
            return str(idx.tz)
        source_tz = df.attrs.get("source_tz")
        if source_tz:
            return str(source_tz)
    return str(section["data_tz_fallback"])


def session_minutes_mask(index, start: tuple[int, int], end: tuple[int, int]):
    start_m = start[0] * 60 + start[1]
    end_m = end[0] * 60 + end[1]
    mins = index.hour * 60 + index.minute
    # de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    if start_m <= end_m:
        return (mins >= start_m) & (mins < end_m)
    return (mins >= start_m) | (mins < end_m)


def utc_session_minutes_mask(index_utc, start: tuple[int, int], end: tuple[int, int]):
    start_m = start[0] * 60 + start[1]
    end_m = end[0] * 60 + end[1]
    mins = index_utc.hour * 60 + index_utc.minute
    if start_m <= end_m:
        return (mins >= start_m) & (mins < end_m)
    return (mins >= start_m) | (mins < end_m)


def make_chart_session(
    df,
    start: tuple[int, int] = (8, 30),
    end: tuple[int, int] = (15, 0),
    source_tz: str = "UTC",
    session_tz: str = "America/New_York",
):
    """Filter to wall-clock session hours, retaining a timezone-aware index."""
    if df.empty:
        return df.copy()

    out = df.sort_index().copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize(source_tz)
    else:
        out.index = out.index.tz_convert(source_tz)

    out.index = out.index.tz_convert(session_tz)
    mask = session_minutes_mask(out.index, start=start, end=end)
    out = out.loc[mask].copy()
    out.attrs["source_tz"] = session_tz
    return out


def make_chart_session_utc(
    df,
    utc_start: tuple[int, int],
    utc_end: tuple[int, int],
    source_tz: str = "America/New_York",
):
    """Filter OHLC to a fixed UTC window, retaining a UTC-aware index."""
    if df.empty:
        return df.copy()

    out = df.sort_index().copy()
    src = str(source_tz or "UTC")
    if out.index.tz is None:
        out.index = out.index.tz_localize(src, ambiguous="infer", nonexistent="raise")
    else:
        out.index = out.index.tz_convert(src)
    utc_index = out.index.tz_convert("UTC")

    mask = utc_session_minutes_mask(utc_index, start=utc_start, end=utc_end)
    out = out.loc[mask].copy()
    out.index = utc_index[mask]
    out.attrs["source_tz"] = "UTC"
    return out


def build_session_chart_df(df, session_name: str, config: dict | None = None, mode: str = "wall"):
    """Return OHLCV dataframe filtered to one session chart."""
    key = normalize_session_name(session_name)
    if key in ("none", "all", "24_7", "24x7", ""):
        return df.copy(), "none"

    if mode == "utc":
        preset = get_chart_session_preset(key, config)
        assert preset is not None
        source_tz = infer_ohlc_source_tz(df, config)
        filtered = make_chart_session_utc(
            df,
            utc_start=tuple(preset["utc_start"]),
            utc_end=tuple(preset["utc_end"]),
            source_tz=source_tz,
        )
        return filtered, f"{key}_utc"

    preset = get_session_preset(key, config)
    assert preset is not None
    source_tz = infer_ohlc_source_tz(df, config)
    filtered = make_chart_session(
        df,
        start=tuple(preset["session_start"]),
        end=tuple(preset["session_end"]),
        source_tz=source_tz,
        session_tz=str(preset["session_tz"]),
    )
    return filtered, key


def parse_sessions_arg(raw: str | None, *, default: str | None = None) -> list[str]:
    """Parse --sessions. Empty/None -> [default] if given, else []."""
    if raw is None or str(raw).strip() == "":
        if default is None:
            return []
        return [normalize_session_name(default)]
    out: list[str] = []
    for part in str(raw).split(","):
        name = normalize_session_name(part.strip())
        if name:
            out.append(name)
    return out
