"""`check --strategy X`: static + runtime consistency gate for one strategy.

1. recipe.toml params load (name/type/min/max)
2. genome_fmt.toml template matches recipe param order and required fields
3. rust/mod.rs reads every param with the helper matching its type
   (param_i32 for int, param_f64 for float) and reads nothing undeclared
4. strategy is compiled into the native core
5. forward parity (causal bar-by-bar == batch) on a mid-range genome

Exit 0 = PASS. Agents must run this after adding or editing a strategy.
"""

from __future__ import annotations

import re
from pathlib import Path

from coensio_algo_ai.genome_fmt import check_genome_template
from coensio_algo_ai.strategy import load_recipe_params, load_recipe_raw, strategy_dir

_I32_RE = re.compile(r'param_i32\s*\([^,]+,[^,]+,\s*"(\w+)"')
_F64_RE = re.compile(r'param_f64\s*\([^,]+,[^,]+,\s*"(\w+)"')
_RAW_RE = re.compile(r'(?:try_)?param_value\s*\([^,]+,\s*"(\w+)"\s*\)')


def rust_param_reads(strategy_id: str) -> tuple[set[str], set[str], set[str]]:
    """(int_reads, float_reads, untyped_reads) across all .rs files in the strategy."""
    rs_dir = strategy_dir(strategy_id) / "rust"
    text = "\n".join(p.read_text(encoding="utf-8") for p in rs_dir.glob("*.rs"))
    return set(_I32_RE.findall(text)), set(_F64_RE.findall(text)), set(_RAW_RE.findall(text))


def static_problems(strategy_id: str) -> list[str]:
    problems: list[str] = []
    raw = load_recipe_raw(strategy_id)
    if raw.get("id") != strategy_id:
        problems.append(f"recipe id={raw.get('id')!r} must equal folder name {strategy_id!r}")
    plugin = raw.get("plugin") or {}
    if plugin.get("type") != strategy_id:
        problems.append(f"[plugin].type={plugin.get('type')!r} must equal {strategy_id!r}")
    try:
        params = load_recipe_params(strategy_id)
    except Exception as exc:  # noqa: BLE001
        return problems + [f"recipe.toml: {exc}"]

    problems.extend(check_genome_template(strategy_id))

    ints, floats, untyped = rust_param_reads(strategy_id)
    for p in params:
        name, typ = p["name"], p["type"]
        if name in untyped:
            continue  # raw read: type responsibility is on the author
        if typ == "int" and name not in ints:
            problems.append(
                f"param {name!r} is int in recipe but rust/mod.rs has no param_i32(.., \"{name}\")"
                + (" (found param_f64)" if name in floats else "")
            )
        if typ == "float" and name not in floats:
            problems.append(
                f"param {name!r} is float in recipe but rust/mod.rs has no param_f64(.., \"{name}\")"
                + (" (found param_i32)" if name in ints else "")
            )
    declared = {p["name"] for p in params}
    for name in sorted((ints | floats | untyped) - declared):
        problems.append(f"rust reads param {name!r} that is not declared in recipe.toml")
    return problems


def run_check(
    strategy_id: str,
    *,
    datafile: str | None,
    max_bars: int,
    sessions: str | None,
    skip_forward: bool = False,
) -> int:
    print("=" * 72)
    print(f"CHECK {strategy_id}")
    print("=" * 72)
    failed = 0

    problems = static_problems(strategy_id)
    if problems:
        failed += 1
        print("static: FAIL")
        for p in problems:
            print(f"  - {p}")
    else:
        print("static: PASS (recipe, genome template, rust param reads)")

    from coensio_algo_ai.fills import _import_native, native_binary_path

    if native_binary_path() is None:
        print("native: FAIL (not built: python coensio_algo_ai/build_native.py)")
        return 2
    native_ids = set(_import_native().list_strategies())
    if strategy_id not in native_ids:
        print(
            "native: FAIL strategy not compiled into the native core "
            "-> python coensio_algo_ai/build_native.py"
        )
        return 2
    print("native: PASS (compiled in)")

    if skip_forward or failed:
        return 2 if failed else 0

    from coensio_algo_ai.data import resolve_data_file
    from coensio_algo_ai.forward import DEFAULT_CASES_FILE, run_forward_cli
    from coensio_algo_ai.sessions import session_mode_default

    file = datafile or DEFAULT_CASES_FILE
    try:
        resolve_data_file(Path(file))
    except FileNotFoundError:
        print(f"forward: SKIP (no data file {file}; pass --file or run sync_data.py)")
        return 0
    rc = run_forward_cli(
        strategy=strategy_id,
        datafile=file,
        genome=None,
        sessions=sessions,
        session_mode=session_mode_default(),
        max_bars=max_bars,
        fixed_bet_size=None,
        cases=True,
        json_out=None,
        only_strategy=strategy_id,
    )
    return 0 if rc == 0 else 2
