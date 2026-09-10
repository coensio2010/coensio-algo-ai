"""GA progress / table formatting (GA).

see coinsio.com
"""

from __future__ import annotations

from pathlib import Path

from coensio_algo_ai.fills import Metrics


def metrics_to_fitness_dict(m: Metrics) -> dict:
    """Map Rust compact metrics to GA fitness keys."""
    trades = int(m.trades)
    avg_roundtrip_fee = float(m.avg_roundtrip_fee)
    return {
        "ret_dd_ratio": float(m.rdd),
        "total_profit": float(m.net_pnl),
        "max_dd_usd": float(m.max_dd),
        "avg_trade_profit": float(m.net_avg),
        "total_trades": trades,
        "stability_r2": float(m.stability),
        "recent_year_pnl": float(m.recent_year_pnl),
        "half2_pnl": float(m.half2_pnl),
        "avg_roundtrip_fee": avg_roundtrip_fee,
        "total_fees": avg_roundtrip_fee * trades,
        "sum_fees": avg_roundtrip_fee,
    }


def format_progress_line(
    metrics: dict | None,
    *,
    best_fit: float,
    ms_per_str: float,
    session: str = "none",
    bet_label: str = "$2000 fixed",
    prefix: str = "",
) -> str:
    """Format GA progress with net results and explicit average/total fees."""
    m = metrics or {}
    rdd = float(m.get("ret_dd_ratio", 0.0))
    pnl = float(m.get("total_profit", 0.0))
    dd = float(m.get("max_dd_usd", 0.0))
    avg = float(m.get("avg_trade_profit", 0.0))
    ntrades = int(m.get("total_trades", 0))
    stab = float(m.get("stability_r2", 0.0))
    ryp = float(m.get("recent_year_pnl", 0.0))
    avg_roundtrip_fee = float(m.get("avg_roundtrip_fee", m.get("sum_fees", 0.0)))
    total_fees = float(m.get("total_fees", avg_roundtrip_fee * ntrades))
    return (
        f"{prefix}"
        f"rdd:{rdd:.2f} net_pnl:${pnl:.2f} DD:${dd:.2f} net_avg:${avg:.2f} "
        f"#{ntrades} stab:{stab:.3f} ryp:${ryp:.0f} "
        f"avg_roundtrip_fee:${avg_roundtrip_fee:.2f} total_fees:${total_fees:.2f} "
        f"sess:{session} bet:{bet_label} best_fit:{best_fit:.3f} {ms_per_str:.2f}ms/str"
    )


def print_progress_line(line: str) -> None:
    """Carriage-return progress update, plus clear-to-EOL so shorter lines do not leave junk (e.g. strr)."""
    print(f"\r{line}\033[K", end="", flush=True)


def print_completion_banner(best_genome_str: str, best_fitness: float) -> None:
    print("=" * 130)
    print("GA OPTIMIZATION COMPLETED!  (powered by coesnio.com)")
    print("=" * 130)
    print(f"-> Best genome: {best_genome_str}")
    print(f"-> Best fitness: ({best_fitness:.2f},)")


TABLE_COLS = (
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
_TABLE_LEFT_ALIGN = frozenset({"genome", "datafile"})


def _genome_identity(genome: object) -> str:
    """unique key for a genome row."""
    if isinstance(genome, str):
        return genome.strip()
    return str(genome or "").strip()


def _metrics_fingerprint(row: dict) -> tuple:
    """Collapse unused-param twins that print identical Top-10 metric columns."""
    return (
        round(float(row.get("ret_dd_ratio", 0) or 0), 2),
        round(float(row.get("total_profit", 0) or 0), 0),
        round(float(row.get("max_dd_usd", 0) or 0), 0),
        round(float(row.get("avg_trade_profit", 0) or 0), 2),
        int(row.get("total_trades", 0) or 0),
        round(float(row.get("stability_r2", 0) or 0), 3),
        round(float(row.get("recent_year_pnl", 0) or 0), 0),
        round(float(row.get("avg_roundtrip_fee", 0) or 0), 2),
        round(float(row.get("total_fees", 0) or 0), 2),
    )


def top_unique_genome_rows(
    ranked: list,
    n: int,
    *,
    min_trades: int,
    datafile: str = "",
) -> list[dict]:
    """Top N rows: unique genome string AND unique printed metrics (no twin spam)."""
    seen_genome: set[str] = set()
    seen_metrics: set[tuple] = set()
    out: list[dict] = []
    for row in ranked:
        metrics = getattr(row, "metrics", None) or {}
        if int(metrics.get("total_trades", 0) or 0) < min_trades:
            continue
        g = _genome_identity(getattr(row, "genome", "") or metrics.get("genome", ""))
        if not g or g in seen_genome:
            continue
        table_row = {
            **metrics,
            "genome": g,
            "datafile": datafile or str(metrics.get("datafile", "")),
        }
        fp = _metrics_fingerprint(table_row)
        if fp in seen_metrics:
            continue
        seen_genome.add(g)
        seen_metrics.add(fp)
        out.append(table_row)
        if len(out) >= n:
            break
    return out


def print_strategy_table(rows: list[dict], *, title: str | None = None, min_trades: int) -> None:
    rows = [r for r in rows if int(r.get("total_trades", 0) or 0) >= min_trades]
    if not rows:
        return
    if title:
        print(f"\n{title}")
    formatted = []
    for row in rows:
        datafile = str(row.get("datafile", "") or "")
        formatted.append(
            {
                "rdd": f"{float(row.get('ret_dd_ratio', 0) or 0):.2f}",
                "net_pnl": f"{float(row.get('total_profit', 0) or 0):.0f}",
                "DD": f"{float(row.get('max_dd_usd', 0) or 0):.0f}",
                "net_avg": f"{float(row.get('avg_trade_profit', 0) or 0):.2f}",
                "#": str(int(row.get("total_trades", 0) or 0)),
                "stab": f"{float(row.get('stability_r2', 0) or 0):.3f}",
                "ryp": f"{float(row.get('recent_year_pnl', 0) or 0):.0f}",
                "avg_fee": f"{float(row.get('avg_roundtrip_fee', 0) or 0):.2f}",
                "total_fees": f"{float(row.get('total_fees', 0) or 0):.2f}",
                "genome": str(row.get("genome", "")),
                "datafile": Path(datafile).name if datafile else "",
            }
        )
    cols = TABLE_COLS
    widths = {c: max(len(c), *(len(r[c]) for r in formatted)) for c in cols}
    sep = " | "
    header = sep.join(
        c.rjust(widths[c]) if c not in _TABLE_LEFT_ALIGN else c.ljust(widths[c]) for c in cols
    )
    divider = sep.join("-" * widths[c] for c in cols)
    print(header)
    print(divider)
    for r in formatted:
        print(
            sep.join(
                r[c].rjust(widths[c]) if c not in _TABLE_LEFT_ALIGN else r[c].ljust(widths[c])
                for c in cols
            )
        )
