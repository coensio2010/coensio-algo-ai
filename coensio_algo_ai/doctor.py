"""`doctor`: verify the local installation and print what is missing.

Exit code 0 = everything needed for backtests/optimizing is present.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_PY = (3, 11)
REQUIRED_MODULES = ("numpy", "pandas", "pyarrow", "plotly")
OPTIONAL_MODULES = {
    "plotext": "terminal equity chart (pip install plotext)",
    "requests": "sync_data.py downloads (pip install requests)",
    "yfinance": "free equity/ETF data in sync_data.py (pip install yfinance)",
}


def _ok(label: str, detail: str = "") -> None:
    print(f"  OK    {label}" + (f"  ({detail})" if detail else ""))


def _warn(label: str, detail: str = "") -> None:
    print(f"  WARN  {label}" + (f"  ({detail})" if detail else ""))


def _fail(label: str, detail: str = "") -> None:
    print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def _tool_version(cmd: list[str]) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or out.stderr or "").strip().splitlines()
    return text[0] if text else "present"


def run_doctor() -> int:
    from coensio_algo_ai import __version__
    from coensio_algo_ai.config import PROJECT_ROOT, cfg_path

    failures = 0
    print(f"coensio-algo-ai {__version__}  doctor")
    print(f"  root   {PROJECT_ROOT}")
    print(f"  python {sys.version.split()[0]}  {platform.system()} {platform.machine()}")
    print()

    print("Python")
    if sys.version_info >= REQUIRED_PY:
        _ok(f"Python >= {REQUIRED_PY[0]}.{REQUIRED_PY[1]}")
    else:
        _fail(f"Python >= {REQUIRED_PY[0]}.{REQUIRED_PY[1]} required", sys.version.split()[0])
        failures += 1
    for mod in REQUIRED_MODULES:
        try:
            m = importlib.import_module(mod)
            _ok(mod, getattr(m, "__version__", ""))
        except ImportError:
            _fail(mod, f"pip install {mod}")
            failures += 1
    for mod, why in OPTIONAL_MODULES.items():
        try:
            m = importlib.import_module(mod)
            _ok(mod, getattr(m, "__version__", "optional"))
        except ImportError:
            _warn(f"{mod} not installed", why)

    print("\nRust toolchain (only needed to (re)build the native core)")
    for tool in (["cargo", "--version"], ["rustc", "--version"], ["maturin", "--version"]):
        v = _tool_version(tool)
        if v:
            _ok(tool[0], v)
        else:
            _warn(f"{tool[0]} not found", "https://rustup.rs / pip install maturin")

    print("\nNative core")
    from coensio_algo_ai.fills import native_binary_path

    pyd = native_binary_path()
    if pyd is None:
        _fail("native binary missing", f"python {PROJECT_ROOT / 'coensio_algo_ai' / 'build_native.py'}")
        failures += 1
        native_ids: list[str] | None = None
    else:
        _ok(pyd.name, f"{pyd.stat().st_size:,} bytes")
        try:
            from coensio_algo_ai.fills import _import_native

            native = _import_native()
            native_ids = sorted(native.list_strategies())
            _ok("native module loads", f"{len(native_ids)} strategies compiled in")
        except Exception as exc:  # noqa: BLE001 - report anything to the user
            _fail("native module failed to load", str(exc))
            failures += 1
            native_ids = None

    print("\nConfig")
    try:
        from coensio_algo_ai.config import load_engine_cfg, load_sweep_cfg, load_validation_cfg

        cfg = load_engine_cfg()
        load_sweep_cfg()
        load_validation_cfg()
        _ok(cfg_path().name, f"datasets={cfg.datasets_dir.name}/ strategies={cfg.strategies_dir.name}/")
    except Exception as exc:  # noqa: BLE001
        _fail(f"{cfg_path().name} invalid", str(exc))
        failures += 1
        cfg = None

    print("\nStrategies")
    from coensio_algo_ai.strategy import list_strategies, load_recipe_params

    folder_ids = list_strategies()
    if not folder_ids:
        _fail("no strategy folders found")
        failures += 1
    for sid in folder_ids:
        try:
            load_recipe_params(sid)
            _ok(sid)
        except Exception as exc:  # noqa: BLE001
            _fail(sid, str(exc))
            failures += 1
    if native_ids is not None:
        missing = sorted(set(folder_ids) - set(native_ids))
        extra = sorted(set(native_ids) - set(folder_ids))
        if missing:
            _fail(
                "strategies not compiled into native core",
                ", ".join(missing) + "  -> rebuild: python coensio_algo_ai/build_native.py",
            )
            failures += 1
        if extra:
            _warn("native core has strategies without folders", ", ".join(extra) + " (rebuild)")

    print("\nData")
    if cfg is not None:
        files = sorted(cfg.datasets_dir.glob("*.parquet")) if cfg.datasets_dir.is_dir() else []
        if files:
            names = ", ".join(p.stem for p in files[:8]) + (" ..." if len(files) > 8 else "")
            _ok(f"{len(files)} parquet file(s) in {cfg.datasets_dir.name}/", names)
        else:
            _warn("no datasets", "python sync_data.py --symbols BTC  (or import-csv)")
        results = Path(cfg.datasets_dir.parent) / "results"
        try:
            results.mkdir(exist_ok=True)
            probe = results / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            _ok("results/ writable")
        except OSError as exc:
            _fail("results/ not writable", str(exc))
            failures += 1

    print()
    if failures:
        print(f"doctor: {failures} problem(s). Fix the FAIL lines above.")
        return 1
    print("doctor: all good. Try: python -m coensio_algo_ai strategies")
    return 0
