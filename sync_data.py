#!/usr/bin/env python
"""Download / sync market data into datasets/<SYMBOL>_<tf>.parquet.

powered by coesnio.com, see https://coensio.com

Sources (no account needed for the first two):
  * Crypto (BTC, ETH, SOL, or any Coinbase product like "AVAX-USD"): Coinbase
    Exchange public candles, full history from --start.
  * Equities / ETFs (SPY, AAPL, ...): yfinance (free). Yahoo limits intraday
    history: 1h = last ~730 days, 15m = last ~60 days, 1d = full history.
  * Equities via HF Data Library (paid, full intraday history): pass --hf-api-key.

Resume / update:
  - Reuses existing datasets/*.parquet
  - Appends only bars after the last local bar
  - Backfills bars before the first local bar (resume after interrupt)
  - Coinbase checkpoints to disk during long pulls (safe to Ctrl+C and re-run)

Usage:
  python sync_data.py --symbols BTC                 # 1h BTC from 2016
  python sync_data.py --symbols BTC,ETH,SPY --tf 1d
  python sync_data.py --symbols AAPL --tf 1h        # yfinance, last 2 years
  python sync_data.py --symbols AAPL --tf 1h --hf-api-key KEY
  python sync_data.py --tf 1h --start 2018-01-01    # default universe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

# --- config (edit or override via CLI) ---
HF_API_KEY = ""
START_DATE = "2016-01-01"
DEFAULT_TF = "1h"  # v1 default; also: 15m, 1d
DATA_DIR = Path(__file__).resolve().parent / "datasets"

CRYPTO = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
}
EQUITY = ("AAPL", "MSFT", "NVDA", "QQQ", "SPY", "GLD", "USO")

CB_GRANULARITY = {"15m": 900, "1h": 3600, "1d": 86400}
HF_TF = {"15m": "15min", "1h": "hourly", "1d": "daily"}
YF_INTERVAL = {"15m": "15m", "1h": "1h", "1d": "1d"}
# Yahoo intraday history limits (days back from now); None = unlimited
YF_MAX_DAYS = {"15m": 59, "1h": 729, "1d": None}
EQUITY_TZ = "America/New_York"

COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
HF_TOKEN_URL = "https://api.hfdatalibrary.com/v1/download-token/{ticker}"
OHLCV = ("open", "high", "low", "close", "volume")
ENGINE_DIR = Path(__file__).resolve().parent
CHECKPOINT_EVERY = 40


def engine_version() -> str:
    candidate = ENGINE_DIR / "coensio_algo_ai" / "VERSION"
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8").strip()
    return "dev"


def resolve_hf_key(cli_key: str | None) -> str:
    return (cli_key or "").strip() or (HF_API_KEY or "").strip()


def fingerprint(df: pd.DataFrame, meta: dict) -> str:
    parts = [json.dumps(meta, sort_keys=True, separators=(",", ":"))]
    if len(df):
        idx = pd.to_datetime(df.index, utc=True).asi8.tobytes()
        parts.append(hashlib.sha256(idx).hexdigest())
        for col in OHLCV:
            parts.append(hashlib.sha256(df[col].astype("float64").to_numpy().tobytes()).hexdigest())
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(OHLCV))
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("index must be DatetimeIndex")
    missing = [c for c in OHLCV if c not in out.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = out[list(OHLCV)].astype("float64")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def load_existing(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=list(OHLCV))
    return normalize_ohlcv(pd.read_parquet(path))


def merge_ohlcv(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = [normalize_ohlcv(f) for f in frames if f is not None and len(f)]
    if not parts:
        return pd.DataFrame(columns=list(OHLCV))
    if len(parts) == 1:
        return parts[0]
    return normalize_ohlcv(pd.concat(parts, axis=0))


def write_dataset(path: Path, df: pd.DataFrame, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = normalize_ohlcv(df)
    meta = dict(meta)
    meta.update(
        {
            "engine_version": engine_version(),
            "bars": len(df),
            "first_bar": None if df.empty else str(df.index.min()),
            "last_bar": None if df.empty else str(df.index.max()),
            "fingerprint": fingerprint(
                df, {k: meta[k] for k in ("symbol", "tf", "source", "start", "tz") if k in meta}
            ),
        }
    )
    df.to_parquet(path)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"  wrote {path.name}  bars={len(df):,}  {meta['first_bar']} -> {meta['last_bar']}")
    print(f"  fingerprint={meta['fingerprint'][:16]}...")


def coinbase_rows_to_df(all_rows: list, gran: int) -> pd.DataFrame:
    if not all_rows:
        return pd.DataFrame(columns=list(OHLCV))
    raw = pd.DataFrame(all_rows, columns=["time", "low", "high", "open", "close", "volume"])
    raw["timestamp"] = pd.to_datetime(raw["time"], unit="s", utc=True)
    raw = raw.set_index("timestamp")[list(OHLCV)]
    now = pd.Timestamp.now(tz="UTC")
    raw = raw[raw.index + pd.Timedelta(seconds=gran) <= now]
    return normalize_ohlcv(raw)


def download_coinbase_range(
    product: str,
    tf: str,
    start_ts: int,
    end_ts: int,
    *,
    checkpoint_path: Path | None = None,
    base_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Download Coinbase candles in [start_ts, end_ts], newest->oldest. Checkpoints to path."""
    gran = CB_GRANULARITY[tf]
    if end_ts <= start_ts:
        return pd.DataFrame(columns=list(OHLCV))

    url = COINBASE_URL.format(product=product)
    headers = {"User-Agent": f"coensio-algo-ai/{engine_version()} (+https://coensio.com)"}
    all_rows: list = []
    current_end = end_ts
    max_per = 300
    retries = 0
    chunks = 0
    base = normalize_ohlcv(base_df) if base_df is not None else pd.DataFrame(columns=list(OHLCV))

    print(
        f"  Coinbase {product} {tf}  "
        f"{datetime.fromtimestamp(start_ts, tz=timezone.utc)} -> "
        f"{datetime.fromtimestamp(end_ts, tz=timezone.utc)}"
    )
    while current_end > start_ts:
        chunk_start = max(start_ts, current_end - max_per * gran)
        params = {"start": chunk_start, "end": current_end, "granularity": gran}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=60)
            r.raise_for_status()
            data = r.json()
            retries = 0
        except requests.HTTPError as exc:
            # 400 = before product listing / invalid candle window, stop, keep what we have
            code = exc.response.status_code if exc.response is not None else None
            if code == 400:
                print(
                    f"\n    Coinbase history floor (HTTP 400), "
                    f"stop at {datetime.fromtimestamp(current_end, tz=timezone.utc).date()}"
                )
                break
            retries += 1
            if retries > 5:
                if checkpoint_path is not None and all_rows:
                    write_dataset(
                        checkpoint_path,
                        merge_ohlcv(base, coinbase_rows_to_df(all_rows, gran)),
                        {
                            "symbol": product.split("-")[0],
                            "tf": tf,
                            "source": "coinbase",
                            "product": product,
                            "start": START_DATE,
                            "end": "now",
                            "tz": "UTC",
                            "partial": True,
                        },
                    )
                raise RuntimeError(f"Coinbase failed after retries: {exc}") from exc
            time.sleep(2.0 * retries)
            continue
        except requests.RequestException as exc:
            retries += 1
            if retries > 5:
                if checkpoint_path is not None and all_rows:
                    write_dataset(
                        checkpoint_path,
                        merge_ohlcv(base, coinbase_rows_to_df(all_rows, gran)),
                        {
                            "symbol": product.split("-")[0],
                            "tf": tf,
                            "source": "coinbase",
                            "product": product,
                            "start": START_DATE,
                            "end": "now",
                            "tz": "UTC",
                            "partial": True,
                        },
                    )
                raise RuntimeError(f"Coinbase failed after retries: {exc}") from exc
            time.sleep(2.0 * retries)
            continue

        if not data:
            current_end = chunk_start - 1
            continue

        rows = [c for c in data if start_ts <= int(c[0]) <= end_ts]
        if not rows:
            current_end = chunk_start - 1
            continue

        all_rows.extend(rows)
        chunks += 1
        oldest = min(int(c[0]) for c in rows)
        print(
            f"    +{len(all_rows):,} new  oldest={datetime.fromtimestamp(oldest, tz=timezone.utc).date()}",
            end="\r",
        )
        if checkpoint_path is not None and chunks % CHECKPOINT_EVERY == 0:
            write_dataset(
                checkpoint_path,
                merge_ohlcv(base, coinbase_rows_to_df(all_rows, gran)),
                {
                    "symbol": product.split("-")[0],
                    "tf": tf,
                    "source": "coinbase",
                    "product": product,
                    "start": START_DATE,
                    "end": "now",
                    "tz": "UTC",
                    "partial": True,
                },
            )
            base = load_existing(checkpoint_path)
            all_rows = []

        if oldest <= start_ts:
            break
        current_end = oldest - 1
        time.sleep(0.35)

    print()
    got = coinbase_rows_to_df(all_rows, gran)
    if checkpoint_path is not None:
        return merge_ohlcv(load_existing(checkpoint_path), got)
    return merge_ohlcv(base, got)

def sync_coinbase(symbol: str, product: str, tf: str, start: str, path: Path) -> None:
    gran = CB_GRANULARITY[tf]
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.now(timezone.utc).timestamp())
    existing = load_existing(path)
    meta = {
        "symbol": symbol,
        "tf": tf,
        "source": "coinbase",
        "product": product,
        "start": start,
        "end": "now",
        "tz": "UTC",
    }

    if existing.empty:
        print("  no local file, full download (checkpointed)")
        df = download_coinbase_range(
            product, tf, start_ts, end_ts, checkpoint_path=path, base_df=existing
        )
        # final merge in case last partial rows not flushed as only checkpoint
        df = merge_ohlcv(load_existing(path), df)
        write_dataset(path, df, meta)
        return

    first = int(existing.index.min().timestamp())
    last = int(existing.index.max().timestamp())
    print(f"  local {len(existing):,} bars  {existing.index.min()} -> {existing.index.max()}")

    # 1) append newer bars
    if last + gran < end_ts - gran:
        print("  append new bars ...")
        newer = download_coinbase_range(
            product, tf, last + gran, end_ts, checkpoint_path=path, base_df=existing
        )
        existing = merge_ohlcv(load_existing(path), newer)
        write_dataset(path, existing, meta)
    else:
        print("  already current at head")

    existing = load_existing(path)
    first = int(existing.index.min().timestamp())

    # 2) backfill older bars (resume after break)
    if first > start_ts + gran:
        print("  backfill older bars (resume) ...")
        older = download_coinbase_range(
            product, tf, start_ts, first - 1, checkpoint_path=path, base_df=existing
        )
        existing = merge_ohlcv(load_existing(path), older)
        write_dataset(path, existing, meta)
    else:
        print("  history complete to start")
        write_dataset(path, existing, meta)


def download_hf(ticker: str, tf: str, start: str, api_key: str) -> pd.DataFrame:
    hf_tf = HF_TF[tf]
    print(f"  HF {ticker} {hf_tf} ...")
    headers_bearer = {"Authorization": f"Bearer {api_key}"}
    params = {"version": "clean", "timeframe": hf_tf, "format": "parquet"}
    r = requests.get(HF_TOKEN_URL.format(ticker=ticker), params=params, headers=headers_bearer, timeout=120)
    if r.status_code == 401:
        r = requests.get(
            HF_TOKEN_URL.format(ticker=ticker),
            params=params,
            headers={"X-API-Key": api_key},
            timeout=120,
        )
    r.raise_for_status()
    payload = r.json()
    url = payload.get("url")
    if not url:
        raise RuntimeError(f"HF token response missing url: {payload!r}")

    d = requests.get(url, timeout=600)
    d.raise_for_status()
    df = pd.read_parquet(BytesIO(d.content))
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    start_ts = pd.Timestamp(start, tz="America/New_York")
    df = df[df.index >= start_ts]
    return normalize_ohlcv(df)


def sync_hf(symbol: str, tf: str, start: str, hf_key: str, path: Path) -> None:
    gran = CB_GRANULARITY[tf]
    existing = load_existing(path)
    meta = {
        "symbol": symbol,
        "tf": tf,
        "source": "hfdatalibrary",
        "hf_version": "clean",
        "start": start,
        "end": "now",
        "tz": "America/New_York",
    }
    now = pd.Timestamp.now(tz="America/New_York")
    # HF API returns full history each call; skip network if local head is fresh
    if len(existing) and existing.index.max() >= now - pd.Timedelta(seconds=2 * gran):
        first_ok = existing.index.min() <= pd.Timestamp(start, tz="America/New_York") + pd.Timedelta(days=7)
        if first_ok:
            print("  local up to date, skip HF download")
            write_dataset(path, existing, meta)
            return

    remote = download_hf(symbol, tf, start, hf_key)
    if existing.empty:
        write_dataset(path, remote, meta)
        return

    # keep any local bars HF might not have yet; prefer remote on overlap
    only_local_new = existing[existing.index > remote.index.max()] if len(remote) else existing
    df = merge_ohlcv(remote, only_local_new)
    added = len(df) - len(existing)
    print(f"  merged HF  local={len(existing):,} -> {len(df):,}  (delta={added:+d})")
    write_dataset(path, df, meta)


def download_yf(ticker: str, tf: str, start: str) -> pd.DataFrame:
    """Free equity/ETF bars via yfinance. Intraday history is limited by Yahoo."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit("yfinance not installed: pip install yfinance") from exc

    interval = YF_INTERVAL[tf]
    start_ts = pd.Timestamp(start, tz=EQUITY_TZ)
    max_days = YF_MAX_DAYS[tf]
    if max_days is not None:
        floor = pd.Timestamp.now(tz=EQUITY_TZ).normalize() - pd.Timedelta(days=max_days)
        if start_ts < floor:
            print(f"  yfinance {interval} history capped at ~{max_days} days -> start {floor.date()}")
            start_ts = floor
    print(f"  yfinance {ticker} {interval} from {start_ts.date()} ...")
    hist = yf.Ticker(ticker).history(
        start=start_ts.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=False,
        actions=False,
    )
    if hist is None or hist.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} {interval}")
    df = hist.rename(columns={c: str(c).lower() for c in hist.columns})
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize(EQUITY_TZ)
    else:
        df.index = df.index.tz_convert(EQUITY_TZ)
    # drop the still-forming last bar
    gran = CB_GRANULARITY[tf]
    now = pd.Timestamp.now(tz=EQUITY_TZ)
    df = df[df.index + pd.Timedelta(seconds=gran) <= now]
    return normalize_ohlcv(df)


def sync_yf(symbol: str, tf: str, start: str, path: Path) -> None:
    existing = load_existing(path)
    meta = {
        "symbol": symbol,
        "tf": tf,
        "source": "yfinance",
        "start": start,
        "end": "now",
        "tz": EQUITY_TZ,
        "adjusted": False,
    }
    remote = download_yf(symbol, tf, start)
    if existing.empty:
        write_dataset(path, remote, meta)
        return
    # prefer remote on overlap, keep older local bars Yahoo no longer serves
    only_local_old = existing[existing.index < remote.index.min()]
    only_local_new = existing[existing.index > remote.index.max()]
    df = merge_ohlcv(only_local_old, remote, only_local_new)
    print(f"  merged yfinance  local={len(existing):,} -> {len(df):,}  (delta={len(df) - len(existing):+d})")
    write_dataset(path, df, meta)


def sync_one(symbol: str, tf: str, start: str, hf_key: str, data_dir: Path) -> None:
    symbol = symbol.upper()
    if symbol in CRYPTO:
        sync_coinbase(symbol, CRYPTO[symbol], tf, start, data_dir / f"{symbol}_{tf}.parquet")
        return
    if symbol.endswith("-USD") or symbol.endswith("-USDT"):
        base = symbol.split("-")[0]
        sync_coinbase(base, symbol, tf, start, data_dir / f"{base}_{tf}.parquet")
        return
    path = data_dir / f"{symbol}_{tf}.parquet"
    if hf_key:
        sync_hf(symbol, tf, start, hf_key, path)
        return
    sync_yf(symbol, tf, start, path)


def default_symbols() -> list[str]:
    return list(CRYPTO.keys()) + list(EQUITY)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="coensio-algo-ai market data sync")
    ap.add_argument("--tf", default=DEFAULT_TF, choices=sorted(CB_GRANULARITY.keys()))
    ap.add_argument(
        "--start",
        "--start_date",
        dest="start",
        default=START_DATE,
        help=f"YYYY-MM-DD inclusive start (default {START_DATE})",
    )
    ap.add_argument(
        "--symbols",
        default="",
        help="comma list: crypto (BTC, ETH, SOL, or AVAX-USD), equities/ETFs (SPY, AAPL). "
        f"Default universe: {','.join(default_symbols())}",
    )
    ap.add_argument(
        "--hf-api-key",
        default=None,
        help="optional HF Data Library key for equities (full intraday history); "
        "without it equities come from yfinance",
    )
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="Output root")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.tf not in CB_GRANULARITY or args.tf not in HF_TF:
        raise SystemExit(f"unsupported tf: {args.tf}")

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or default_symbols()
    hf_key = resolve_hf_key(args.hf_api_key)

    print(f"engine={engine_version()}  tf={args.tf}  start={args.start}  end=now")
    print(f"out={data_dir}")
    print(f"symbols={','.join(symbols)}")

    for i, sym in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {sym}")
        sync_one(sym, args.tf, args.start, hf_key, data_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
