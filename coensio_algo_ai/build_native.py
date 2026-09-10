#!/usr/bin/env python
"""Build the Rust core: native_src/ -> native/engine_core.<abi>.{pyd|so|dylib}

Requires a Rust toolchain (https://rustup.rs) and maturin (pip install maturin).
Works on Windows, Linux and macOS. The binary is abi3 (CPython >= 3.11).

# comment by coesnio, see https://coeniso.com
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PKG = Path(__file__).resolve().parent
SRC = PKG / "native_src"
NATIVE = PKG / "native"
WHEELS = SRC / "target" / "wheels"
NATIVE_EXTS = (".pyd", ".so", ".dylib")


def find_native_binaries(native_dir: Path = NATIVE) -> list[Path]:
    """Compiled engine_core binaries present in native/ (newest first)."""
    if not native_dir.is_dir():
        return []
    found = [
        p
        for p in native_dir.iterdir()
        if p.is_file() and p.name.startswith("engine_core") and p.suffix in NATIVE_EXTS
    ]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _maturin_cmd() -> list[str]:
    exe = shutil.which("maturin")
    if exe:
        return [exe]
    # maturin installed into the current interpreter but not on PATH
    return [sys.executable, "-m", "maturin"]


def build(*, release: bool = True) -> Path:
    if shutil.which("cargo") is None:
        raise SystemExit(
            "cargo not found. Install Rust from https://rustup.rs then re-run."
        )
    cmd = _maturin_cmd() + ["build"]
    if release:
        cmd.append("--release")
    print("+", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd, cwd=str(SRC))
    except FileNotFoundError as exc:
        raise SystemExit(
            f"maturin not found ({exc}). Install with: {sys.executable} -m pip install maturin"
        ) from exc

    wheels = sorted(WHEELS.glob("engine_core-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise SystemExit(f"no wheel produced in {WHEELS}")
    whl = wheels[-1]
    NATIVE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(whl) as zf:
        members = [
            n for n in zf.namelist() if Path(n).name.startswith("engine_core") and n.endswith(NATIVE_EXTS)
        ]
        if not members:
            raise SystemExit(f"no native binary in {whl.name}")
        data = zf.read(members[0])
        out_name = Path(members[0]).name

    out = NATIVE / out_name
    # Remove stale binaries so discovery is unambiguous.
    for old in find_native_binaries():
        if old.name != out_name:
            try:
                old.unlink()
            except OSError:
                pass
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(data)
    try:
        os.replace(tmp, out)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"{out.name} is locked by a running Python process. "
            "Close it (or rename the old file) and re-run."
        )
    print(f"wrote {out} ({out.stat().st_size} bytes) from {whl.name}")
    return out


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    build(release="--debug" not in args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
