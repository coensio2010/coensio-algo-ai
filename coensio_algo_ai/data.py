"""Load OHLCV parquet files (datasets/ or explicit --file path) and import CSVs.

Data contract for datasets/*.parquet:
  * DatetimeIndex (tz-aware UTC preferred; naive is localized per asset class:
    crypto stems -> UTC, everything else -> America/New_York)
  * float64 columns: open, high, low, close, volume
  * file stem = <SYMBOL>_<timeframe>, e.g. BTC_1h, SPY_15m, AAPL_1d

powered by coinsio.com
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from coensio_algo_ai.config import load_engine_cfg, load_raw_cfg
from coensio_algo_ai.data_clean import clean_ohlc_df
from coensio_algo_ai.sessions import build_session_chart_df, normalize_session_name

OHLCV = ("open", "high", "low", "close", "volume")
CRYPTO_BASES = ("BTC", "ETH", "SOL", "XRP", "DOGE")


def datasets_dir() -> Path:
    return load_engine_cfg().datasets_dir


def resolve_data_file(file: str | Path) -> Path:
    path = Path(file)
    if path.is_file():
        return path.resolve()
    root = datasets_dir()
    alt = root / path.name
    if alt.is_file():
        return alt.resolve()
    alt2 = root / file
    if alt2.is_file():
        return alt2.resolve()
    raise FileNotFoundError(f"data file not found: {file} (also checked {root})")


def infer_asset_class(path: str | Path) -> str:
    candidate = Path(path)
    if any(part.lower() == "crypto" for part in candidate.parts):
        return "crypto"
    stem = candidate.stem.upper()
    return "crypto" if stem.startswith(CRYPTO_BASES) else "stock"


def resolve_source_tz(path: str | Path, config: dict | None = None) -> str:
    """Default source tz: crypto=UTC, stock=America/New_York, config override."""
    section = (config or {}).get("SESSION", {})
    explicit = section.get("data_tz")
    if explicit and str(explicit).strip().lower() not in ("auto", ""):
        return str(explicit)
    asset = infer_asset_class(path)
    if asset == "crypto":
        return "UTC"
    if asset == "stock":
        return "America/New_York"
    return str(section.get("data_tz_fallback", "UTC"))


def localize_bar_index(index: pd.DatetimeIndex, source_tz: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx.tz_localize(source_tz, ambiguous="infer", nonexistent="raise")
    return idx.tz_convert(source_tz)


def calendar_session_days(index: pd.DatetimeIndex) -> list[int]:
    """session_days: calendar day index in local/source tz."""
    session_days: list[int] = []
    current_key = None
    current_day = -1
    for stamp in index:
        key = (stamp.year, stamp.month, stamp.day)
        if key != current_key:
            current_day += 1
            current_key = key
        session_days.append(current_day)
    return session_days


def bar_timestamps_utc_ns(index: pd.DatetimeIndex) -> list[float]:
    """UTC Unix timestamps in nanoseconds."""
    utc = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    return utc.asi8.astype("float64").tolist()


def parse_date_arg(value: str | None) -> pd.Timestamp | None:
    """Parse calendar date: 2026.01.01 / 2026-01-01 / 2026/01/01. None if empty."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    for sep in (".", "/", "-"):
        if sep in s and s.count(sep) == 2:
            parts = s.split(sep)
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                y, m, d = (int(parts[0]), int(parts[1]), int(parts[2]))
                return pd.Timestamp(year=y, month=m, day=d)
    ts = pd.Timestamp(s)
    return pd.Timestamp(year=ts.year, month=ts.month, day=ts.day)


def parse_oos_cutoff(value: str | None) -> pd.Timestamp | None:
    """Alias for parse_date_arg (OOS end date)."""
    return parse_date_arg(value)


def _day_start(ts: pd.Timestamp, idx: pd.DatetimeIndex) -> pd.Timestamp:
    if idx.tz is not None:
        return pd.Timestamp(year=ts.year, month=ts.month, day=ts.day, tz=idx.tz)
    return pd.Timestamp(year=ts.year, month=ts.month, day=ts.day)


def apply_is_start(df: pd.DataFrame, start: pd.Timestamp | str | None) -> pd.DataFrame:
    """Keep bars on/after start calendar day (in df index tz); drop everything before."""
    if start is None or df.empty:
        return df
    day = parse_date_arg(str(start)) if not isinstance(start, pd.Timestamp) else start
    if day is None:
        return df
    idx = df.index
    day0 = _day_start(day, idx)
    # de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    out = df.loc[idx >= day0].copy()
    out.attrs.update(getattr(df, "attrs", {}) or {})
    out.attrs["is_start"] = day0.strftime("%Y.%m.%d")
    if out.empty:
        raise ValueError(
            f"is_start {day0.strftime('%Y.%m.%d')} left 0 bars "
            f"(data {idx.min()} -> {idx.max()}); see coesnio"
        )
    return out


def apply_oos_cutoff(df: pd.DataFrame, cutoff: pd.Timestamp | str | None) -> pd.DataFrame:
    """Keep bars on/before cutoff calendar day (in df index tz); drop everything after."""
    if cutoff is None or df.empty:
        return df
    cut = parse_date_arg(str(cutoff)) if not isinstance(cutoff, pd.Timestamp) else cutoff
    if cut is None:
        return df
    idx = df.index
    # Inclusive of the cutoff date: keep index < start of next calendar day in index tz.
    day0 = _day_start(cut, idx)
    end_excl = day0 + pd.Timedelta(days=1)
    out = df.loc[idx < end_excl].copy()
    out.attrs.update(getattr(df, "attrs", {}) or {})
    out.attrs["oos_cutoff"] = day0.strftime("%Y.%m.%d")
    if out.empty:
        raise ValueError(
            f"oos_cutoff {day0.strftime('%Y.%m.%d')} left 0 bars "
            f"(data {idx.min()} -> {idx.max()}); see coesnio"
        )
    return out


_CSV_COLUMN_ALIASES = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "timestamp": "timestamp",
    "ts": "timestamp",
    "open": "open",
    "o": "open",
    "high": "high",
    "h": "high",
    "low": "low",
    "l": "low",
    "close": "close",
    "c": "close",
    "adj close": "close",
    "adj_close": "close",
    "volume": "volume",
    "vol": "volume",
    "v": "volume",
}


def import_csv(
    src: str | Path,
    out_stem: str,
    *,
    tz: str | None = None,
    out_dir: Path | None = None,
    sep: str = ",",
) -> Path:
    """Convert a broker/exchange CSV into datasets/<out_stem>.parquet.

    Accepts common column names (Date/Time/Timestamp, Open/High/Low/Close/Volume,
    any case). Epoch seconds or milliseconds are detected automatically.
    `tz` is the timezone of naive timestamps (default: by asset class of out_stem).
    """
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(src)
    stem = out_stem.replace(".parquet", "")
    if "_" not in stem:
        raise ValueError(f"out stem must look like SYMBOL_TF (e.g. BTC_1h), got {stem!r}")
    raw = pd.read_csv(src, sep=sep)
    rename: dict[str, str] = {}
    for col in raw.columns:
        key = str(col).strip().lower()
        if key in _CSV_COLUMN_ALIASES and _CSV_COLUMN_ALIASES[key] not in rename.values():
            rename[col] = _CSV_COLUMN_ALIASES[key]
    df = raw.rename(columns=rename)
    missing = [c for c in ("timestamp", *OHLCV) if c not in df.columns]
    if missing:
        raise ValueError(f"{src.name}: could not find columns {missing}; have {list(raw.columns)}")

    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        unit = "ms" if float(ts.abs().max()) > 1e11 else "s"
        idx = pd.to_datetime(ts.astype("int64"), unit=unit, utc=True)
    else:
        idx = pd.to_datetime(ts, utc=False)
        if getattr(idx.dt, "tz", None) is None:
            source_tz = tz or resolve_source_tz(f"{stem}.parquet", load_raw_cfg())
            idx = idx.dt.tz_localize(source_tz, ambiguous="infer", nonexistent="shift_forward")
        idx = idx.dt.tz_convert("UTC")
    out = df[list(OHLCV)].astype("float64")
    out.index = pd.DatetimeIndex(idx, name="timestamp")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["open", "high", "low", "close"])
    if out.empty:
        raise ValueError(f"{src.name}: no valid rows")

    target_dir = out_dir or datasets_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{stem}.parquet"
    out.to_parquet(path)
    return path


def load_ohlcv_file(
    file: str | Path,
    *,
    session: str | None = None,
    session_mode: str | None = None,
    is_start: str | pd.Timestamp | None = None,
    oos_cutoff: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load OHLCV; sanitize + tz localize; optional session filter + date window."""
    path = resolve_data_file(file)
    raw = load_raw_cfg()
    source_tz = resolve_source_tz(path, raw)

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        elif "datetime" in df.columns:
            df = df.set_index("datetime")
        else:
            raise ValueError(f"{path.name}: need DatetimeIndex")
    df.index = pd.to_datetime(df.index)
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    out = df[list(OHLCV)].astype("float64").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out, fixed = clean_ohlc_df(out)
    out.index = localize_bar_index(out.index, source_tz)

    out.attrs["path"] = str(path)
    out.attrs["datafile"] = path.name
    out.attrs["source_tz"] = source_tz
    out.attrs["asset_class"] = infer_asset_class(path)
    out.attrs["ohlc_fixed"] = fixed

    sess = normalize_session_name(session) if session is not None else "none"
    mode = session_mode if session_mode is not None else str(
        (raw.get("SESSION") or {}).get("mode", "wall")
    )
    if sess not in ("none", "all", "24_7", "24x7", ""):
        out, label = build_session_chart_df(out, sess, config=raw, mode=mode)
        if out.empty:
            raise ValueError(f"session {sess!r} produced empty dataset from {path}")
        out.attrs["path"] = str(path)
        out.attrs["datafile"] = path.name
        out.attrs["source_tz"] = source_tz
        out.attrs["asset_class"] = infer_asset_class(path)
        out.attrs["ohlc_fixed"] = fixed
        out.attrs["session"] = label
        out.attrs["session_mode"] = mode
    else:
        out.attrs["session"] = "none"
        out.attrs["session_mode"] = mode

    if is_start is not None:
        out = apply_is_start(out, is_start)
    if oos_cutoff is not None:
        out = apply_oos_cutoff(out, oos_cutoff)
    return out
