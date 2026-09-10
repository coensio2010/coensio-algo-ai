"""CLI: python -m coensio_algo_ai <command> ...

coensio-algo-ai - powered by coensio.com
Copyright coesnio.com
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from coensio_algo_ai.backtest import format_result, run_genome_backtest
from coensio_algo_ai.brand import POWERED_BY, TAGLINE
from coensio_algo_ai.config import (
    DEFAULT_CFG_NAME,
    cfg_path,
    load_engine_cfg,
    resolve_fixed_bet_size,
    set_cfg_path,
)
from coensio_algo_ai.data import load_ohlcv_file
from coensio_algo_ai.fills import EngineConfig
from coensio_algo_ai.optimize import run_ga
from coensio_algo_ai.sessions import parse_sessions_arg, session_default, session_mode_default
from coensio_algo_ai.strategy import list_strategies, load_strategy

PROG = "python -m coensio_algo_ai"

COMMANDS = {
    "doctor": "check Python, Rust, native core, config and data",
    "strategies": "list strategies and their genome templates",
    "backtest": "run one genome on one dataset, print metrics + chart + reports",
    "optimize": "GA search on one dataset (and one or more sessions)",
    "sweep": "GA search over [SWEEP] tickers x sessions x cycles",
    "backtest-forward": "look-ahead check: causal bar-by-bar signals must equal batch",
    "validate": "Monte Carlo permutation test (MCPT) for one genome",
    "check": "static + parity gate for a strategy folder (run after editing a strategy)",
    "new-strategy": "scaffold strategies/<id>/ with a compiling example",
    "import-csv": "convert a CSV to datasets/<SYMBOL_TF>.parquet",
}
ALIASES = {"forward": "backtest-forward", "mcpt": "validate", "optimize_sweep": "sweep"}


def _add_cfg_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--cfg",
        "-cfg",
        default=None,
        help=f"config TOML (default: {DEFAULT_CFG_NAME} in the repo root)",
    )


def _apply_cfg_arg(args: argparse.Namespace) -> None:
    if getattr(args, "cfg", None):
        set_cfg_path(args.cfg)


def _peel_global_cfg(argv: list[str]) -> list[str]:
    """Allow `python -m coensio_algo_ai --cfg FILE <cmd> ...` before the subcommand."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--cfg", "-cfg") and i + 1 < len(argv):
            set_cfg_path(argv[i + 1])
            i += 2
            continue
        if a.startswith("--cfg=") or a.startswith("-cfg="):
            set_cfg_path(a.split("=", 1)[1])
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _engine_config(args: argparse.Namespace, datafile: str | None = None) -> EngineConfig:
    cfg = load_engine_cfg()
    bet = (
        float(args.bet)
        if args.bet is not None
        else resolve_fixed_bet_size(cfg.fixed_bet_sizes, datafile)
    )
    return EngineConfig(
        initial_capital=args.capital if args.capital is not None else cfg.initial_capital,
        fixed_bet_size=bet,
        commission_rate=(
            args.commission if args.commission is not None else cfg.commission_rate
        ),
        slippage_rate=args.slippage if args.slippage is not None else cfg.slippage_rate,
        bet_mode=cfg.bet_mode,
        price_bet_frac=cfg.price_bet_frac,
    )


def _add_engine_args(ap: argparse.ArgumentParser) -> None:
    _add_cfg_arg(ap)
    ap.add_argument("--capital", type=float, default=None, help="override [ENGINE].initial_capital")
    ap.add_argument("--bet", type=float, default=None, help="override fixed bet size ($ notional)")
    ap.add_argument("--commission", type=float, default=None, help="override commission_rate")
    ap.add_argument("--slippage", type=float, default=None, help="override slippage_rate")


def _add_range_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--session-mode",
        choices=("wall", "utc"),
        default=None,
        help="wall | utc. Default: [SESSION].mode",
    )
    ap.add_argument(
        "--is-start",
        "--is_start",
        dest="is_start",
        default=None,
        help="inclusive start date YYYY.MM.DD (also YYYY-MM-DD); drop bars before this day",
    )
    ap.add_argument(
        "--oos-cutoff",
        "--oos_cutoff",
        dest="oos_cutoff",
        default=None,
        help="inclusive end date YYYY.MM.DD (also YYYY-MM-DD); drop bars after this day",
    )


def _add_session_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--sessions",
        default=None,
        help="session preset(s): new_york, london, asia, none (comma-separated). "
        "Default: [SESSION].default",
    )
    _add_range_args(ap)


def _parser(name: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=f"{PROG} {name}", description=COMMANDS[name])


def cmd_strategies(argv: list[str]) -> int:
    ap = _parser("strategies")
    _add_cfg_arg(ap)
    args = ap.parse_args(argv)
    _apply_cfg_arg(args)
    from coensio_algo_ai.genome_fmt import genome_template

    ids = list_strategies()
    if not ids:
        print("(no strategies)")
        return 0
    for sid in ids:
        mod = load_strategy(sid)
        tmpl = genome_template(sid)
        if tmpl is not None:
            print(f"{sid}  genome={tmpl}")
        else:
            fields = "|".join(p["name"] for p in mod.PARAMS)
            print(f"{sid}  genome={fields}")
    return 0


def cmd_backtest(argv: list[str]) -> int:
    ap = _parser("backtest")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--file", required=True, help="dataset name (BTC_1h.parquet) or path")
    ap.add_argument("--genome", required=True, help="genome string (see `strategies`)")
    _add_session_args(ap)
    _add_engine_args(ap)
    args = ap.parse_args(argv)
    _apply_cfg_arg(args)

    from coensio_algo_ai.config import load_raw_cfg
    from coensio_algo_ai.reporting import finalize_backtest_report

    mode = args.session_mode if args.session_mode is not None else session_mode_default()
    sessions = parse_sessions_arg(args.sessions, default=session_default())
    data_label = Path(args.file).name
    raw_cfg = load_raw_cfg()
    for sess in sessions:
        eng = _engine_config(args, data_label)
        df = load_ohlcv_file(
            args.file,
            session=sess,
            session_mode=mode,
            is_start=args.is_start,
            oos_cutoff=args.oos_cutoff,
        )
        datafile = Path(df.attrs.get("datafile") or args.file).name
        finalize_backtest_report(
            strategy_id=args.strategy,
            genome_str=args.genome,
            df=df,
            datafile=datafile,
            session=sess,
            config=eng,
            raw_cfg=raw_cfg,
            title="Full-range results",
        )
    return 0


def _add_ga_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--population", type=int, default=None, help="default [GA_CONFIG].population_size")
    ap.add_argument("--generations", type=int, default=None, help="default [GA_CONFIG].num_generations")
    ap.add_argument("--seed", type=int, default=None, help="default [GA_CONFIG].seed (0 = random)")
    ap.add_argument("--min-trades", type=int, default=None, help="default [GA_CONFIG].min_num_trades")
    ap.add_argument("--workers", type=int, default=None, help="Rust threads; default [GA_CONFIG].workers")


def cmd_optimize(argv: list[str]) -> int:
    ap = _parser("optimize")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--file", required=True, help="dataset name (BTC_1h.parquet) or path")
    _add_ga_args(ap)
    _add_session_args(ap)
    _add_engine_args(ap)
    args = ap.parse_args(argv)
    _apply_cfg_arg(args)
    cfg = load_engine_cfg()

    mode = args.session_mode if args.session_mode is not None else session_mode_default()
    sessions = parse_sessions_arg(args.sessions, default=session_default())
    mod = load_strategy(args.strategy)

    for sess in sessions:
        df = load_ohlcv_file(
            args.file,
            session=sess,
            session_mode=mode,
            is_start=args.is_start,
            oos_cutoff=args.oos_cutoff,
        )
        datafile = Path(df.attrs.get("datafile") or args.file).name
        eng = _engine_config(args, datafile)
        ranked = run_ga(
            df,
            mod,
            strategy_id=args.strategy,
            datafile=datafile,
            population=(
                args.population if args.population is not None else cfg.population_size
            ),
            generations=(
                args.generations if args.generations is not None else cfg.num_generations
            ),
            seed=args.seed if args.seed is not None else cfg.seed,
            min_trades=(
                args.min_trades if args.min_trades is not None else cfg.min_num_trades
            ),
            config=eng,
            session_name=sess,
            workers=args.workers if args.workers is not None else cfg.workers,
        )
        if ranked:
            best = ranked[0].genome
            result = run_genome_backtest(
                args.strategy,
                args.file,
                best,
                config=eng,
                session=sess,
                session_mode=mode,
            )
            print(
                format_result(
                    f"{args.strategy} {datafile} sess={sess} [{best}]",
                    result,
                    session=sess,
                    config=eng,
                )
            )
    return 0


def cmd_sweep(argv: list[str]) -> int:
    ap = _parser("sweep")
    ap.add_argument("--strategy", required=True)
    _add_ga_args(ap)
    _add_range_args(ap)
    _add_engine_args(ap)
    args = ap.parse_args(argv)
    _apply_cfg_arg(args)
    cfg = load_engine_cfg()

    from coensio_algo_ai.sweep import run_optimize_sweep

    run_optimize_sweep(
        strategy_id=args.strategy,
        population=args.population if args.population is not None else cfg.population_size,
        generations=(
            args.generations if args.generations is not None else cfg.num_generations
        ),
        workers=args.workers if args.workers is not None else cfg.workers,
        min_trades=args.min_trades if args.min_trades is not None else cfg.min_num_trades,
        seed=args.seed if args.seed is not None else cfg.seed,
        session_mode=args.session_mode,
        is_start=args.is_start,
        oos_cutoff=args.oos_cutoff,
    )
    return 0


def cmd_validate(argv: list[str]) -> int:
    from coensio_algo_ai.config import load_validation_cfg
    from coensio_algo_ai.genome_fmt import parse_genome_str
    from coensio_algo_ai.validate import validate_genome

    ap = _parser("validate")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--genome", required=True)
    _add_session_args(ap)
    _add_engine_args(ap)
    ap.add_argument("--n-perm", type=int, default=None, help="default [VALIDATION].n_perm")
    ap.add_argument("--alpha", type=float, default=None, help="default [VALIDATION].alpha")
    ap.add_argument("--metric", default=None, help="default [VALIDATION].metric")
    ap.add_argument("--seed", type=int, default=None, help="default [VALIDATION].seed")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    _apply_cfg_arg(args)
    vcfg = load_validation_cfg()

    import json

    mode = args.session_mode if args.session_mode is not None else session_mode_default()
    sessions = parse_sessions_arg(args.sessions, default=session_default())
    if len(sessions) != 1:
        print("FAIL: validate uses one session (pass --sessions london)")
        return 1
    sess = sessions[0]
    data_label = Path(args.file).name
    eng = _engine_config(args, data_label)
    mod = load_strategy(args.strategy)
    parsed = parse_genome_str(args.strategy, args.genome, mod.PARAMS)
    df = load_ohlcv_file(
        args.file,
        session=sess,
        session_mode=mode,
        is_start=args.is_start,
        oos_cutoff=args.oos_cutoff,
    )

    n_perm = int(args.n_perm if args.n_perm is not None else vcfg.n_perm)
    alpha = float(args.alpha if args.alpha is not None else vcfg.alpha)
    metric = str(args.metric if args.metric is not None else vcfg.metric)
    seed = int(args.seed if args.seed is not None else vcfg.seed)

    print("=" * 78)
    print(f"VALIDATE  strategy={args.strategy}  session={sess}")
    print(f"cfg={cfg_path().name}  data={Path(args.file).name}  bars={len(df)}")
    print(f"genome={args.genome}")
    print(f"n_perm={n_perm}  alpha={alpha}  metric={metric}  seed={seed}")
    print("=" * 78)

    result = validate_genome(
        df,
        strategy_id=args.strategy,
        genome_str=args.genome,
        genome_values=list(parsed.genome),
        engine=eng,
        metric=metric,
        n_perm=n_perm,
        alpha=alpha,
        seed=seed,
    )

    print()
    print("-" * 78)
    print(
        f"real_{metric}={result.real_metric:.4f}  "
        f"p={result.p_value:.4f}  "
        f"z_score={result.z_score:.2f}  "
        f"perm_mean={result.perm_mean:.4f}  "
        f"perm_std={result.perm_std:.4f}  "
        f"trades={result.real_trades}"
    )
    if result.passed:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        for reason in result.fail_reasons:
            print(f"  - {reason}")
    print("-" * 78)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if result.passed else 2


def cmd_backtest_forward(argv: list[str]) -> int:
    ap = _parser("backtest-forward")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--file", dest="datafile", default=None)
    ap.add_argument("--genome", default=None, help="omit with --cases to use mid-range genomes")
    ap.add_argument("--sessions", default=None)
    _add_range_args(ap)
    ap.add_argument(
        "--max-bars",
        type=int,
        default=0,
        help="use only the last N bars (forward mode is O(N^2)); 0 = full series",
    )
    ap.add_argument("--bet", type=float, default=None, help="override fixed bet size")
    ap.add_argument(
        "--cases",
        action="store_true",
        help="run every installed strategy with a mid-range genome on --file",
    )
    ap.add_argument("--json-out", default=None)
    _add_cfg_arg(ap)
    args = ap.parse_args(argv)
    _apply_cfg_arg(args)

    from coensio_algo_ai.forward import run_forward_cli

    mode = args.session_mode if args.session_mode is not None else session_mode_default()
    return run_forward_cli(
        strategy=args.strategy,
        datafile=args.datafile,
        genome=args.genome,
        sessions=args.sessions,
        session_mode=mode,
        max_bars=int(args.max_bars),
        fixed_bet_size=args.bet,
        cases=bool(args.cases),
        json_out=args.json_out,
        is_start=args.is_start,
        oos_cutoff=args.oos_cutoff,
    )


def cmd_doctor(argv: list[str]) -> int:
    ap = _parser("doctor")
    _add_cfg_arg(ap)
    args = ap.parse_args(argv)
    _apply_cfg_arg(args)
    from coensio_algo_ai.doctor import run_doctor

    return run_doctor()


def cmd_check(argv: list[str]) -> int:
    ap = _parser("check")
    ap.add_argument("--strategy", default=None, help="strategy id; default = all")
    ap.add_argument("--file", dest="datafile", default=None, help="dataset for forward parity")
    ap.add_argument("--sessions", default=None, help="one session for parity (default none)")
    ap.add_argument("--max-bars", type=int, default=3000, help="bars for forward parity")
    ap.add_argument("--no-forward", action="store_true", help="static + native checks only")
    _add_cfg_arg(ap)
    args = ap.parse_args(argv)
    _apply_cfg_arg(args)
    from coensio_algo_ai.check import run_check

    ids = [args.strategy] if args.strategy else list_strategies()
    worst = 0
    for sid in ids:
        rc = run_check(
            sid,
            datafile=args.datafile,
            max_bars=int(args.max_bars),
            sessions=args.sessions,
            skip_forward=bool(args.no_forward),
        )
        worst = max(worst, rc)
    print("=" * 72)
    print("CHECK RESULT:", "PASS" if worst == 0 else "FAIL")
    return worst


def cmd_new_strategy(argv: list[str]) -> int:
    ap = _parser("new-strategy")
    ap.add_argument("id", help="snake_case strategy id, e.g. my_breakout")
    ap.add_argument("--force", action="store_true", help="overwrite an existing folder")
    _add_cfg_arg(ap)
    args = ap.parse_args(argv)
    _apply_cfg_arg(args)
    from coensio_algo_ai.scaffold import scaffold_strategy
    from coensio_algo_ai.strategy import strategies_dir

    files = scaffold_strategy(strategies_dir(), args.id, force=bool(args.force))
    for f in files:
        print(f"wrote {f}")
    print()
    print("Next steps:")
    print(f"  1. edit {files[0].parent / 'rust' / 'mod.rs'} and recipe.toml")
    print("  2. python coensio_algo_ai/build_native.py")
    print(f"  3. {PROG} check --strategy {args.id}")
    print(f"  4. {PROG} optimize --strategy {args.id} --file BTC_1h.parquet --population 30 --generations 20")
    return 0


def cmd_import_csv(argv: list[str]) -> int:
    ap = _parser("import-csv")
    ap.add_argument("src", help="CSV file with date/time + open/high/low/close/volume columns")
    ap.add_argument("out", help="dataset stem SYMBOL_TF, e.g. EURUSD_1h")
    ap.add_argument("--tz", default=None, help="timezone of naive timestamps (default by asset class)")
    ap.add_argument("--sep", default=",", help="CSV separator")
    _add_cfg_arg(ap)
    args = ap.parse_args(argv)
    _apply_cfg_arg(args)
    from coensio_algo_ai.data import import_csv

    path = import_csv(args.src, args.out, tz=args.tz, sep=args.sep)
    df = load_ohlcv_file(path)
    print(f"wrote {path}  bars={len(df):,}  {df.index.min()} -> {df.index.max()}")
    return 0


HANDLERS = {
    "doctor": cmd_doctor,
    "strategies": cmd_strategies,
    "backtest": cmd_backtest,
    "optimize": cmd_optimize,
    "sweep": cmd_sweep,
    "backtest-forward": cmd_backtest_forward,
    "validate": cmd_validate,
    "check": cmd_check,
    "new-strategy": cmd_new_strategy,
    "import-csv": cmd_import_csv,
}


def print_help() -> None:
    print(TAGLINE)
    print(f"  {POWERED_BY}  |  https://coensio.com")
    print()
    print(f"usage: {PROG} [--cfg FILE] <command> [options]   (run from the repo root)")
    print()
    width = max(len(k) for k in COMMANDS)
    for name, desc in COMMANDS.items():
        print(f"  {name.ljust(width)}  {desc}")
    print()
    print(f"Default cfg: {DEFAULT_CFG_NAME}. Use `{PROG} <command> -h` for options.")
    print("Examples:")
    print(f"  {PROG} doctor")
    print(f"  {PROG} optimize --strategy donchian_atr --file BTC_1h.parquet --sessions none --population 50 --generations 30")
    print(f"  {PROG} backtest --strategy donchian_atr --file BTC_1h.parquet --genome \"donchian_atr|both|...\"")
    print(f"  {PROG} check --strategy donchian_atr")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _peel_global_cfg(argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print_help()
        return 0
    cmd, rest = argv[0], argv[1:]
    cmd = ALIASES.get(cmd, cmd)
    handler = HANDLERS.get(cmd)
    if handler is None:
        print(f"unknown command: {cmd}\n", file=sys.stderr)
        print_help()
        return 2
    try:
        return handler(rest)
    except (ValueError, KeyError, FileNotFoundError, NotADirectoryError) as exc:
        # user-input problems (bad genome, unknown strategy, missing file, bad cfg):
        # short message, no traceback. Set COENSIO_DEBUG=1 to see the full trace.
        if os.environ.get("COENSIO_DEBUG"):
            raise
        msg = exc.args[0] if exc.args else str(exc)
        print(f"error: {msg}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
