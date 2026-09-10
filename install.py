#!/usr/bin/env python
"""Install coensio-algo-ai: Python deps, native Rust core, optional data.

coensio-algo-ai - powered by coensio.com
Copyright coesnio.com

Uses the Python interpreter that runs this script (a venv is recommended but
not required). Steps:
  1. pip install runtime deps (numpy, pandas, pyarrow, requests, plotext, yfinance, maturin)
  2. build the native core if it is missing or --rebuild (needs Rust: https://rustup.rs)
  3. optional: download market data (Coinbase for crypto, yfinance for equities)
  4. run `doctor`

Usage:
  python install.py                      # deps + native build + doctor
  python install.py --data               # also sync default BTC/ETH 1h data
  python install.py --data --symbols BTC,ETH,SPY --tf 1h
  python install.py --skip-native        # deps only (use a prebuilt native/ binary)
  python install.py --rebuild            # force native rebuild
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "coensio_algo_ai"
VERSION_FILE = PKG / "VERSION"
MIN_PY = (3, 11)

DEPS = (
    "numpy>=1.26",
    "pandas>=2.0",
    "pyarrow>=14.0",
    "requests>=2.31",
    "plotext>=5.2,<6",
    "plotly>=5.20",
    "yfinance>=1.0",
)
BUILD_DEPS = ("maturin>=1.14,<2.0", "pytest>=8.0")


def read_version() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd))


def ensure_deps(*, build: bool) -> None:
    pkgs = list(DEPS) + (list(BUILD_DEPS) if build else [])
    run([sys.executable, "-m", "pip", "install", "--upgrade", *pkgs])


def native_present() -> bool:
    sys.path.insert(0, str(ROOT))
    from coensio_algo_ai.build_native import find_native_binaries

    return bool(find_native_binaries())


def build_native() -> None:
    run([sys.executable, str(PKG / "build_native.py")])


def sync_data(symbols: str, tf: str, hf_key: str | None) -> None:
    cmd = [sys.executable, str(ROOT / "sync_data.py"), "--tf", tf]
    if symbols:
        cmd += ["--symbols", symbols]
    if hf_key:
        cmd += ["--hf-api-key", hf_key]
    run(cmd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Install coensio-algo-ai")
    ap.add_argument("--skip-native", action="store_true", help="do not build the Rust core")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the Rust core even if present")
    ap.add_argument("--data", action="store_true", help="download market data after install")
    ap.add_argument("--symbols", default="BTC,ETH", help="comma list for --data (default BTC,ETH)")
    ap.add_argument("--tf", default="1h", choices=("15m", "1h", "1d"), help="timeframe for --data")
    ap.add_argument("--hf-api-key", default=None, help="optional HF Data Library key (equities)")
    ap.add_argument("--no-doctor", action="store_true", help="skip the final doctor run")
    args = ap.parse_args(argv)

    if sys.version_info < MIN_PY:
        raise SystemExit(f"Python {MIN_PY[0]}.{MIN_PY[1]}+ required, got {sys.version.split()[0]}")

    print(f"coensio-algo-ai {read_version()}")
    print(f"root:   {ROOT}")
    print(f"python: {sys.executable}")

    ensure_deps(build=not args.skip_native)

    if args.skip_native:
        print("skip native build (--skip-native)")
    elif args.rebuild or not native_present():
        build_native()
    else:
        print("native core already built (use --rebuild to force)")

    if args.data:
        sync_data(args.symbols, args.tf, args.hf_api_key)

    if not args.no_doctor:
        print()
        return subprocess.call([sys.executable, "-m", "coensio_algo_ai", "doctor"], cwd=str(ROOT))
    print("Install complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
