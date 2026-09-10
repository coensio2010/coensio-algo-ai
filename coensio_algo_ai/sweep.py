"""Multi-ticker / multi-session GA sweep.

Copyright coesnio.com, see https://coensio.com
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from coensio_algo_ai.config import (
    PROJECT_ROOT,
    cfg_path,
    load_engine_cfg,
    load_raw_cfg,
    load_sweep_cfg,
    resolve_fixed_bet_size,
)
from coensio_algo_ai.data import load_ohlcv_file, resolve_data_file
from coensio_algo_ai.fills import EngineConfig
from coensio_algo_ai.optimize import GenomeResult, run_ga
from coensio_algo_ai.progress import print_strategy_table, top_unique_genome_rows
from coensio_algo_ai.reporting import reporting_settings
from coensio_algo_ai.strategy import load_strategy


SWEEP_SESSIONS_DEFAULT = ("none", "new_york", "london", "asia")


def _results_sweep_dir(raw_cfg: dict, strategy_id: str, *, stamp: str) -> Path:
    rep = reporting_settings(raw_cfg)
    path = Path(rep["results_dir"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    safe = str(strategy_id).replace("/", "-").replace("\\", "-").replace(" ", "_")
    out = path / f"sweep_{safe}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _row_from_genome_result(row: GenomeResult, *, datafile: str) -> dict[str, Any]:
    return {
        **row.metrics,
        "genome": row.genome,
        "datafile": datafile,
        "fitness": float(row.fitness),
    }


def _merge_unique(global_rows: list[dict], new_rows: list[dict]) -> tuple[float, float]:
    """Append unique genomes (by genome string); keep higher fitness on clash.

    Returns (merge_ms, sort_ms).
    """
    t0 = time.perf_counter()
    by_genome = {str(r.get("genome", "")): r for r in global_rows if r.get("genome")}
    for row in new_rows:
        g = str(row.get("genome", "") or "")
        if not g:
            continue
        prev = by_genome.get(g)
        if prev is None or float(row.get("fitness", row.get("ret_dd_ratio", -1e18))) > float(
            prev.get("fitness", prev.get("ret_dd_ratio", -1e18))
        ):
            by_genome[g] = row
    merge_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    global_rows.clear()
    global_rows.extend(
        sorted(
            by_genome.values(),
            key=lambda r: float(r.get("ret_dd_ratio", 0) or 0),
            reverse=True,
        )
    )
    sort_ms = (time.perf_counter() - t1) * 1000.0
    return merge_ms, sort_ms


def _best_per_datafile(global_rows: list[dict]) -> list[dict]:
    """One best genome per datafile (highest ret_dd_ratio), sorted by RDD desc."""
    best: dict[str, dict] = {}
    for row in global_rows:
        df = str(row.get("datafile", "") or "").strip()
        if not df:
            continue
        key = Path(df).name
        prev = best.get(key)
        score = float(row.get("ret_dd_ratio", 0) or 0)
        if prev is None or score > float(prev.get("ret_dd_ratio", 0) or 0):
            best[key] = row
    return sorted(
        best.values(),
        key=lambda r: float(r.get("ret_dd_ratio", 0) or 0),
        reverse=True,
    )


def _write_text_safe(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Atomic-ish write; on Windows lock (Errno 22 / PermissionError) fall back to stamped name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
        return path
    except OSError as exc:
        try:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        except OSError:
            pass
        alt = path.with_name(f"{path.stem}_retry{path.suffix}")
        try:
            alt.write_text(text, encoding=encoding)
            print(
                f"SWEEP WARN: could not write {path.name} ({exc}); wrote {alt.name}",
                flush=True,
            )
            return alt
        except OSError as exc2:
            print(f"SWEEP WARN: skip write {path.name}: {exc2}", flush=True)
            return path


def _save_global(out_dir: Path, rows: list[dict], *, stamp: str) -> tuple[Path, Path]:
    json_path = out_dir / f"global_all_runs_{stamp}.json"
    csv_path = out_dir / f"global_all_runs_{stamp}.csv"
    jsonl_path = out_dir / "global_rows.jsonl"

    payload = []
    for r in rows:
        payload.append(
            {
                "ret_dd_ratio": float(r.get("ret_dd_ratio", 0) or 0),
                "total_profit": float(r.get("total_profit", 0) or 0),
                "max_dd_usd": float(r.get("max_dd_usd", 0) or 0),
                "avg_trade_profit": float(r.get("avg_trade_profit", 0) or 0),
                "total_trades": int(r.get("total_trades", 0) or 0),
                "stability_r2": float(r.get("stability_r2", 0) or 0),
                "recent_year_pnl": float(r.get("recent_year_pnl", 0) or 0),
                "avg_roundtrip_fee": float(r.get("avg_roundtrip_fee", 0) or 0),
                "total_fees": float(r.get("total_fees", 0) or 0),
                "genome": str(r.get("genome", "")),
                "datafile": str(r.get("datafile", "")),
                "fitness": float(r.get("fitness", r.get("ret_dd_ratio", 0)) or 0),
            }
        )
    json_body = json.dumps(payload, indent=2)
    json_path = _write_text_safe(json_path, json_body)

    # Full snapshot (not append-all, which bloated jsonl every job)
    jsonl_body = "".join(json.dumps(row) + "\n" for row in payload)
    _write_text_safe(jsonl_path, jsonl_body)

    cols = (
        "rdd",
        "net_pnl",
        "DD",
        "net_avg",
        "#",
        "stab",
        "ryp",
        "avg_fee",
        "total_fees",
        "genome",
        "datafile",
    )
    lines = [" | ".join(cols)]
    for r in payload:
        lines.append(
            " | ".join(
                [
                    f"{r['ret_dd_ratio']:.2f}",
                    f"{r['total_profit']:.0f}",
                    f"{r['max_dd_usd']:.0f}",
                    f"{r['avg_trade_profit']:.2f}",
                    str(r["total_trades"]),
                    f"{r['stability_r2']:.3f}",
                    f"{r['recent_year_pnl']:.0f}",
                    f"{r['avg_roundtrip_fee']:.2f}",
                    f"{r['total_fees']:.2f}",
                    r["genome"],
                    Path(r["datafile"]).name,
                ]
            )
        )
    csv_body = "\n".join(lines) + "\n"
    csv_path = _write_text_safe(csv_path, csv_body)

    # Latest aliases (may be locked in Excel, never crash the sweep)
    _write_text_safe(out_dir / "global_all_runs.json", json_body)
    _write_text_safe(out_dir / "global_all_runs.csv", csv_body)
    return json_path, csv_path


def run_optimize_sweep(
    *,
    strategy_id: str,
    population: int | None = None,
    generations: int | None = None,
    workers: int | None = None,
    min_trades: int | None = None,
    seed: int | None = None,
    session_mode: str | None = None,
    oos_cutoff: str | None = None,
    is_start: str | None = None,
) -> list[dict[str, Any]]:
    """Sweep optimize: cycles x tickers x sessions; collect unique genomes globally."""
    ecfg = load_engine_cfg()
    raw_cfg = load_raw_cfg()
    sweep = load_sweep_cfg()
    mod = load_strategy(strategy_id)

    pop = int(population if population is not None else ecfg.population_size)
    gens = int(generations if generations is not None else ecfg.num_generations)
    n_workers = int(workers if workers is not None else ecfg.workers)
    trades_min = int(min_trades if min_trades is not None else ecfg.min_num_trades)
    seed_val = int(seed if seed is not None else ecfg.seed)
    mode = session_mode if session_mode is not None else str(
        (raw_cfg.get("SESSION") or {}).get("mode", "wall")
    )

    jobs: list[tuple[int, str, str]] = []
    for cycle in range(1, sweep.cycles + 1):
        for ticker in sweep.tickers:
            for sess in sweep.sessions:
                jobs.append((cycle, ticker, sess))

    total = len(jobs)
    range_bits = []
    if is_start:
        range_bits.append(f"is_start={is_start}")
    if oos_cutoff:
        range_bits.append(f"oos_cutoff={oos_cutoff}")
    cut_label = (" " + " ".join(range_bits)) if range_bits else ""
    print(
        f"SWEEP | strategy={strategy_id} | cfg={cfg_path().name} | cycles={sweep.cycles} | "
        f"tickers={len(sweep.tickers)} | sessions={list(sweep.sessions)} | "
        f"jobs={total} | pop={pop} gen={gens} workers={n_workers}{cut_label} | "
        f"powered by ceonsio.com",
        flush=True,
    )

    global_rows: list[dict[str, Any]] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _results_sweep_dir(raw_cfg, strategy_id, stamp=stamp)
    print(f"SWEEP out: {out_dir}", flush=True)
    failures: list[str] = []

    for job_i, (cycle, ticker, sess) in enumerate(jobs, start=1):
        datafile = f"{ticker}.parquet"
        print(
            f"\n=== SWEEP {job_i}/{total} cycle={cycle} ticker={ticker} session={sess} ===",
            flush=True,
        )
        try:
            path = resolve_data_file(datafile)
            df = load_ohlcv_file(
                path,
                session=sess,
                session_mode=mode,
                is_start=is_start,
                oos_cutoff=oos_cutoff,
            )
            if df.empty:
                raise ValueError(f"empty dataset after session={sess!r}")
            # de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
            bet = resolve_fixed_bet_size(ecfg.fixed_bet_sizes, datafile)
            eng = EngineConfig(
                initial_capital=ecfg.initial_capital,
                fixed_bet_size=bet,
                commission_rate=ecfg.commission_rate,
                slippage_rate=ecfg.slippage_rate,
                bet_mode=ecfg.bet_mode,
                price_bet_frac=ecfg.price_bet_frac,
            )
            ranked = run_ga(
                df,
                mod,
                strategy_id=strategy_id,
                datafile=datafile,
                population=pop,
                generations=gens,
                seed=seed_val,
                min_trades=trades_min,
                config=eng,
                session_name=sess,
                workers=n_workers,
                report=False,
            )
            run_rows = top_unique_genome_rows(
                ranked,
                n=max(sweep.table_top_n, ecfg.table_top_n),
                min_trades=trades_min,
                datafile=datafile,
            )
            for r in run_rows:
                r["fitness"] = float(r.get("ret_dd_ratio", 0) or 0)
            merge_ms, sort_ms = _merge_unique(global_rows, run_rows)
        except Exception as exc:
            msg = f"{ticker} sess={sess} cycle={cycle}: {exc}"
            failures.append(msg)
            print(f"SWEEP SKIP: {msg}", flush=True)
            continue

        # Refresh global snapshot after each job
        t_write = time.perf_counter()
        _save_global(out_dir, global_rows, stamp=stamp)
        write_ms = (time.perf_counter() - t_write) * 1000.0
        market_best = _best_per_datafile(global_rows)
        print_strategy_table(
            market_best,
            title=f"Top market genomes ({len(market_best)} markets so far)",
            min_trades=trades_min,
        )
        print(
            f"post-job timing: merge={merge_ms:.1f}ms  sort={sort_ms:.1f}ms  "
            f"write={write_ms:.1f}ms  global_rows={len(global_rows)}",
            flush=True,
        )

    print("\n" + "=" * 130)
    print("SWEEP COMPLETE")
    print("=" * 130)
    market_best = _best_per_datafile(global_rows)
    print_strategy_table(
        market_best,
        title=f"Top market genomes ({len(market_best)} markets)",
        min_trades=trades_min,
    )
    top_n = int(sweep.table_top_n)
    print_strategy_table(
        global_rows[:top_n],
        title=f"Top {top_n} unique genomes (of {len(global_rows)}; full list in CSV)",
        min_trades=trades_min,
    )
    json_path, csv_path = _save_global(out_dir, global_rows, stamp=stamp)
    print(f"\nGlobal JSON: {json_path}")
    print(f"Global CSV:  {csv_path}")
    if failures:
        print(f"\nSkipped {len(failures)} job(s):")
        for msg in failures:
            print(f"  - {msg}")
    return global_rows
