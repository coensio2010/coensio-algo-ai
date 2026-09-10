"""Backtest-forward: causal bar-by-bar signals vs batch signals. Must match 1:1.

The batch path computes all signals from the full series. The forward path
re-runs the plugin on a growing prefix [0..i] and keeps only the signal for bar
i. Any difference means the strategy reads future bars (look-ahead bug).

see coesnio

Usage:
  python -m coensio_algo_ai backtest-forward --strategy donchian_atr --file BTC_1h.parquet --genome "..." --sessions none
  python -m coensio_algo_ai backtest-forward --cases --file BTC_1h.parquet --max-bars 3000   # every strategy, mid genome
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coensio_algo_ai.config import load_engine_cfg, resolve_fixed_bet_size
from coensio_algo_ai.data import load_ohlcv_file
from coensio_algo_ai.fills import EngineConfig, Metrics, evaluate_recipe_detailed, evaluate_recipe_forward
from coensio_algo_ai.genome_fmt import format_genome_str, parse_genome_str
from coensio_algo_ai.strategy import load_strategy


def _n_trades(row: dict) -> int:
    t = row.get("trades")
    if isinstance(t, list):
        return len(t)
    m = row.get("metrics")
    if isinstance(m, Metrics):
        return int(m.trades)
    return int(t or 0)


def _cmp_metrics(batch: dict, fwd: dict, tol: float = 1e-6) -> list[str]:
    keys = ("rdd", "net_pnl", "max_dd", "net_avg", "stability", "recent_year_pnl", "half2_pnl")
    diffs: list[str] = []
    bt, ft = _n_trades(batch), _n_trades(fwd)
    if bt != ft:
        diffs.append(f"trades batch={bt} forward={ft}")
    bm, fm = batch.get("metrics"), fwd.get("metrics")
    if not isinstance(bm, Metrics) or not isinstance(fm, Metrics):
        return diffs
    for k in keys:
        a = getattr(bm, k)
        b = getattr(fm, k)
        if abs(float(a) - float(b)) > max(tol, tol * abs(float(a))):
            diffs.append(f"{k} batch={a} forward={b}")
    return diffs


def _cmp_trades(batch: dict, fwd: dict) -> list[str]:
    bt = batch.get("trades") or []
    ft = fwd.get("trades") or []
    if len(bt) != len(ft):
        return [f"trade_count batch={len(bt)} forward={len(ft)}"]
    diffs: list[str] = []
    for i, (a, b) in enumerate(zip(bt, ft)):
        for k in ("entry_bar", "exit_bar", "direction", "exit_reason"):
            if a.get(k) != b.get(k):
                diffs.append(f"trade[{i}].{k} batch={a.get(k)} forward={b.get(k)}")
                break
        for k in ("net_pnl", "entry_price", "exit_price"):
            if abs(float(a.get(k, 0)) - float(b.get(k, 0))) > 1e-6:
                diffs.append(f"trade[{i}].{k} batch={a.get(k)} forward={b.get(k)}")
                break
        if len(diffs) >= 5:
            diffs.append("...")
            break
    return diffs


def _mid_genome(strategy_id: str, session: str, bet: float, price_bet_frac: float) -> str:
    mod = load_strategy(strategy_id)
    params: dict[str, float] = {}
    for p in mod.PARAMS:
        mid = (float(p["start"]) + float(p["end"])) * 0.5
        if p["type"] == "int":
            mid = float(int(round(mid)))
        params[p["name"]] = mid
    return format_genome_str(
        strategy_id,
        mod.PARAMS,
        params,
        session=session or "none",
        bet_mode="fixed",
        fixed_bet_size=bet,
        price_bet_frac=price_bet_frac,
    )


def run_one(
    *,
    strategy: str,
    datafile: str | Path,
    genome_str: str,
    sessions: str | None,
    session_mode: str,
    max_bars: int | None,
    fixed_bet_size: float | None,
    oos_cutoff: str | None = None,
    is_start: str | None = None,
) -> dict[str, Any]:
    ecfg = load_engine_cfg()
    mod = load_strategy(strategy)
    parsed = parse_genome_str(strategy, genome_str, mod.PARAMS)
    sess = sessions
    if sess is None and parsed.session and parsed.session != "none":
        sess = parsed.session
    sess = sess or "none"

    df = load_ohlcv_file(
        datafile,
        session=sess,
        session_mode=session_mode,
        is_start=is_start,
        oos_cutoff=oos_cutoff,
    )
    data_label = Path(df.attrs.get("datafile") or datafile).name
    if max_bars and max_bars > 0 and len(df) > max_bars:
        df = df.iloc[-max_bars:].copy()

    bet = (
        float(fixed_bet_size)
        if fixed_bet_size is not None
        else (
            float(parsed.fixed_bet_size)
            if parsed.bet_mode == "fixed"
            else resolve_fixed_bet_size(ecfg.fixed_bet_sizes, data_label)
        )
    )
    eng = EngineConfig(
        initial_capital=ecfg.initial_capital,
        fixed_bet_size=bet,
        commission_rate=ecfg.commission_rate,
        slippage_rate=ecfg.slippage_rate,
        bet_mode=ecfg.bet_mode,
        price_bet_frac=ecfg.price_bet_frac,
    )

    vec = list(parsed.genome)
    batch = evaluate_recipe_detailed(df, strategy, vec, config=eng)
    bm = batch["metrics"]
    print(
        f"  batch   rdd={bm.rdd} pnl={bm.net_pnl} trades={bm.trades}",
        flush=True,
    )
    n_bars = len(df)
    print(
        f"  forward building causal signals ({n_bars:,} bars)... "
        f"[O(N^2) look-ahead check; full series can take many minutes. "
        f"Smoke: --max-bars 2000]",
        flush=True,
    )
    fwd = evaluate_recipe_forward(df, strategy, vec, config=eng)
    fm = fwd["metrics"]
    mismatch = fwd.get("first_signal_mismatch_bar")
    print(
        f"  forward rdd={fm.rdd} pnl={fm.net_pnl} trades={fm.trades} "
        f"mismatch_bar={mismatch}",
        flush=True,
    )

    m_diff = _cmp_metrics(batch, fwd)
    t_diff = _cmp_trades(batch, fwd)
    ok = not m_diff and not t_diff and mismatch is None
    return {
        "strategy": strategy,
        "run": f"{Path(data_label).stem}:{sess}",
        "bars": len(df),
        "ok": ok,
        "mismatch_bar": mismatch,
        "metric_diffs": m_diff,
        "trade_diffs": t_diff,
        "batch": {
            "rdd": bm.rdd,
            "net_pnl": bm.net_pnl,
            "max_dd": bm.max_dd,
            "trades": bm.trades,
        },
        "forward": {
            "rdd": fm.rdd,
            "net_pnl": fm.net_pnl,
            "max_dd": fm.max_dd,
            "trades": fm.trades,
        },
        "genome": genome_str,
    }


DEFAULT_CASES_FILE = "BTC_1h.parquet"
DEFAULT_CASES_MAX_BARS = 3000


def default_cases(datafile: str | None = None, sessions: str | None = None) -> list[dict]:
    """Smoke panel: every installed strategy, mid-range genome, one datafile."""
    from coensio_algo_ai.strategy import list_strategies

    return [
        {
            "strategy": sid,
            "datafile": datafile or DEFAULT_CASES_FILE,
            "sessions": sessions or "none",
            "genome": None,
            "fixed_bet_size": None,
        }
        for sid in list_strategies()
    ]


def run_forward_cli(
    *,
    strategy: str | None,
    datafile: str | None,
    genome: str | None,
    sessions: str | None,
    session_mode: str,
    max_bars: int,
    fixed_bet_size: float | None,
    cases: bool,
    json_out: str | None,
    oos_cutoff: str | None = None,
    is_start: str | None = None,
    only_strategy: str | None = None,
) -> int:
    ecfg = load_engine_cfg()
    if cases or not strategy:
        case_list = default_cases(datafile, sessions)
        if only_strategy:
            case_list = [c for c in case_list if c["strategy"] == only_strategy]
        if max_bars <= 0:
            max_bars = DEFAULT_CASES_MAX_BARS
    else:
        if not datafile or not genome:
            print("Need --file and --genome (or --cases)")
            return 1
        case_list = [
            {
                "strategy": strategy,
                "datafile": datafile,
                "sessions": sessions,
                "genome": genome,
                "fixed_bet_size": fixed_bet_size,
            }
        ]

    results = []
    failed = 0
    for case in case_list:
        sid = case["strategy"]
        path = Path(case["datafile"])
        try:
            from coensio_algo_ai.data import resolve_data_file

            resolve_data_file(path)
        except FileNotFoundError:
            print(f"FAIL missing data file {path} (run sync_data.py or pass --file)")
            failed += 1
            continue
        sess = case.get("sessions")
        gstr = case.get("genome")
        bet = case.get("fixed_bet_size")
        if bet is None:
            bet = resolve_fixed_bet_size(ecfg.fixed_bet_sizes, path.name)
        if not gstr:
            gstr = _mid_genome(sid, sess or "none", float(bet), ecfg.price_bet_frac)
        print("=" * 72, flush=True)
        print(f"FORWARD PARITY  {sid}  {path.name}  sess={sess}", flush=True)
        row = run_one(
            strategy=sid,
            datafile=path,
            genome_str=gstr,
            sessions=sess,
            session_mode=session_mode,
            max_bars=max_bars if max_bars > 0 else None,
            fixed_bet_size=float(bet) if bet is not None else None,
            is_start=is_start,
            oos_cutoff=oos_cutoff,
        )
        results.append(row)
        if row["ok"]:
            print("RESULT: PASS (batch == forward)", flush=True)
        else:
            failed += 1
            print("RESULT: FAIL", flush=True)
            for d in row["metric_diffs"]:
                print(f"  - {d}", flush=True)
            for d in row["trade_diffs"]:
                print(f"  - {d}", flush=True)
            if row["mismatch_bar"] is not None:
                print(f"  - first_signal_mismatch_bar={row['mismatch_bar']}", flush=True)

    if json_out:
        Path(json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {json_out}")

    print("=" * 72)
    print(f"done: {len(case_list) - failed}/{len(case_list)} PASS")
    return 0 if failed == 0 else 2
