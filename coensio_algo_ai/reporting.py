"""REPORTING: terminal chart, HTML equity, QA CSV.

# comment by coesnio, see coensoi.com
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coensio_algo_ai.config import PROJECT_ROOT, load_raw_cfg
from coensio_algo_ai.data import bar_timestamps_utc_ns
from coensio_algo_ai.html_report import write_html_equity_report
from coensio_algo_ai.metrics_extra import extended_metrics
from coensio_algo_ai.fills import EngineConfig, evaluate_recipe_detailed
from coensio_algo_ai.strategy import load_strategy, params_to_genome_vec


METRIC_KEYS = (
    "total_return_pct",
    "total_profit",
    "max_dd_usd",
    "max_dd_pct",
    "ret_dd_ratio",
    "total_trades",
    "win_rate",
    "avg_trade_profit",
    "avg_roundtrip_fee",
    "total_fees",
    "total_slippage",
    "total_transaction_cost",
    "profit_factor",
    "sharpe",
    "sortino",
    "stability_r2",
    "recent_year_pnl",
    "half2_pnl",
)

METRIC_LABELS = {
    "total_profit": "net_pnl",
    "avg_trade_profit": "net_avg",
}

METRIC_FMT = {
    "total_return_pct": lambda v: f"{v:.2f}",
    "total_profit": lambda v: f"{v:.2f}",
    "max_dd_usd": lambda v: f"{v:.2f}",
    "max_dd_pct": lambda v: f"{v:.2f}",
    "ret_dd_ratio": lambda v: f"{v:.2f}",
    "total_trades": lambda v: f"{int(v)}",
    "win_rate": lambda v: f"{v:.2f}",
    "avg_trade_profit": lambda v: f"{v:.2f}",
    "avg_roundtrip_fee": lambda v: f"{v:.2f}",
    "total_fees": lambda v: f"{v:.2f}",
    "total_slippage": lambda v: f"{v:.2f}",
    "total_transaction_cost": lambda v: f"{v:.2f}",
    "profit_factor": lambda v: f"{v:.2f}",
    "sharpe": lambda v: f"{v:.3f}",
    "sortino": lambda v: f"{v:.3f}",
    "stability_r2": lambda v: f"{v:.4f}",
    "recent_year_pnl": lambda v: f"{v:.2f}",
    "half2_pnl": lambda v: f"{v:.2f}",
}


def metrics_summary_rows(
    metrics: dict | None,
    *,
    session: str,
    bet_label: str,
) -> list[tuple[str, str]]:
    if not metrics:
        return []
    rows: list[tuple[str, str]] = [
        ("session", session),
        ("bet_size", bet_label),
    ]
    for key in METRIC_KEYS:
        if key not in metrics:
            continue
        label = METRIC_LABELS.get(key, key)
        val = metrics[key]
        try:
            text = METRIC_FMT.get(key, str)(val)
        except (TypeError, ValueError):
            text = str(val)
        rows.append((label, text))
    return rows


def print_metrics_summary(
    metrics: dict | None, *, title: str, session: str, bet_label: str
) -> None:
    if not metrics:
        return
    print(f"\n{title}")
    for label, text in metrics_summary_rows(metrics, session=session, bet_label=bet_label):
        print(f"  {label}: {text}")


def reporting_settings(config: dict | None = None) -> dict:
    cfg = config if config is not None else load_raw_cfg()
    rep = cfg.get("REPORTING") or cfg.get("reporting") or {}
    required = (
        "terminal_chart",
        "terminal_chart_size",
        "terminal_chart_width",
        "terminal_chart_combined",
        "terminal_chart_side_by_side",
        "html_report",
        "qa_csv",
        "results_dir",
    )
    for key in required:
        if key not in rep:
            raise KeyError(f"engine_cfg.toml: missing [REPORTING].{key}")
    return {
        "terminal_chart": bool(rep["terminal_chart"]),
        "terminal_chart_size": int(rep["terminal_chart_size"]),
        "terminal_chart_width": int(rep["terminal_chart_width"]),
        "terminal_chart_combined": bool(rep["terminal_chart_combined"]),
        "terminal_chart_side_by_side": bool(rep["terminal_chart_side_by_side"]),
        "html_report": bool(rep["html_report"]),
        "qa_csv": bool(rep["qa_csv"]),
        "results_dir": str(rep["results_dir"]),
    }


def _results_dir(raw_cfg: dict) -> Path:
    rep = reporting_settings(raw_cfg)
    path = Path(rep["results_dir"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pnl_drawdown_per_trade(trades: list[dict]):
    if not trades:
        return None, None
    pnls = np.asarray([float(t.get("net_pnl", 0.0)) for t in trades], dtype=np.float64)
    if pnls.size < 1:
        return None, None
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return cum, dd


def _emit(text: str) -> None:
    try:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
    except UnicodeEncodeError:
        buf = getattr(sys.stdout, "buffer", None)
        if buf is not None:
            buf.write((text + "\n").encode("utf-8", errors="replace"))
            buf.flush()


def _plot_side_by_side(plt, x_values, pnl_values, dd_values, base, size, width_override=0):
    if len(pnl_values) < 2:
        return False
    x = [float(v) for v in x_values]
    yp = [float(v) for v in pnl_values]
    yd = [float(v) for v in dd_values]
    height = int(size)
    if width_override and width_override > 0:
        width = int(width_override) * 2
    else:
        term_cols = shutil.get_terminal_size((120, 30)).columns
        square_w = int(round(size * 2.1)) * 2
        # de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
        width = max(square_w, term_cols - 2)
    plt.clear_figure()
    plt.subplots(1, 2)
    plt.plotsize(width, height)
    plt.subplot(1, 1)
    plt.theme("clear")
    plt.scatter(x, yp, color="green", marker="braille")
    plt.title(f"{base} | Equity PnL ($)")
    plt.xlabel("trade #")
    plt.subplot(1, 2)
    plt.theme("clear")
    plt.scatter(x, yd, color="red", marker="braille")
    plt.title(f"{base} | Drawdown ($)")
    plt.xlabel("trade #")
    _emit(plt.build())
    return True


def show_strategy_chart(detailed: dict, metrics: dict, genome_str: str, raw_cfg: dict) -> bool:
    try:
        import plotext as plt
    except ImportError:
        print("\nTerminal chart: install plotext: pip install plotext")
        return False

    trades = detailed.get("trades") or []
    pnl, dd = _pnl_drawdown_per_trade(trades)
    if pnl is None:
        print("\nTerminal chart: no trades in backtest results.")
        return False

    rep = reporting_settings(raw_cfg)
    size = int(rep["terminal_chart_size"])
    width = int(rep["terminal_chart_width"])
    side_by_side = bool(rep["terminal_chart_side_by_side"])

    n_trades = pnl.size
    x_full = np.arange(1, n_trades + 1, dtype=np.float64)
    rdd = metrics.get("ret_dd_ratio", 0)
    max_dd = metrics.get("max_dd_usd", float(np.max(dd)) if dd.size else 0.0)
    avg = metrics.get("avg_trade_profit", 0)
    ryp = metrics.get("recent_year_pnl", 0)
    base = (
        f"net_pnl ${metrics.get('total_profit', 0):.0f} DD:${max_dd:.0f} RDD={rdd} "
        f"net_avg:${avg:.2f} #{n_trades} ryp:${ryp:.0f}"
    )
    if width > 0:
        chart_dims = f"{width}x{size}"
    else:
        term_cols = shutil.get_terminal_size((120, 30)).columns
        chart_dims = f"{max(int(round(size * 2.1)), term_cols - 2)}x{size}"

    print(f"\n--- Terminal chart (plotext, {chart_dims}) ---")
    print(base)
    print(f"Genome: {genome_str}")
    if side_by_side:
        return _plot_side_by_side(plt, x_full, pnl, (-dd), base, size, width)
    return False


QA_HEADER = (
    "Ticket,OpenTime,Action,Size,Symbol,OpenPrice,Unused,Unused,"
    "CloseTime,ClosePrice,Unused,Unused,PL,Unused,Unused"
)


def _fmt_time(ts) -> str:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return ""
    if hasattr(ts, "strftime"):
        return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)


def trades_to_trade_list(trades: list[dict], index: pd.DatetimeIndex, genome_str: str) -> list[list]:
    rows: list[list] = []
    n = len(index)
    for t in trades:
        ei = int(t.get("entry_bar", 0))
        xi = int(t.get("exit_bar", 0))
        if ei < 0 or xi < 0 or ei >= n or xi >= n:
            continue
        rows.append(
            [
                index[ei],
                index[xi],
                float(t.get("entry_price", 0.0)),
                float(t.get("exit_price", 0.0)),
                str(t.get("exit_reason", "")),
                float(t.get("net_pnl", 0.0)),
                genome_str,
                int(t.get("direction", 1) or 1),
            ]
        )
    return rows


def write_qa_csv(filepath: Path, trade_rows: list[list], strategy_id: str, direction: int = 1) -> None:
    from coensio_algo_ai.html_report import write_qa_csv as _write

    _write(filepath, trade_rows, strategy_id, direction=direction)


def build_strategy_name(genome_str: str, strategy_type: str, symbol: str, timeframe: str) -> str:
    """Report file name: [SYMBOL_TF]_strategy_(genome)_Ddate."""
    from datetime import datetime

    gen_date = datetime.now().strftime("%Y%m%d")
    safe_type = str(strategy_type).replace(" ", "_")
    safe_genome = str(genome_str).replace("|", "_").replace("/", "-").replace(":", "-")
    return f"[{symbol}_{timeframe}]_{safe_type}_({safe_genome})_D{gen_date}"


def infer_symbol_timeframe(data_path: Path, session: str) -> tuple[str, str]:
    stem = data_path.stem
    parts = stem.replace("-", "_").split("_")
    symbol = parts[0] if parts else "SYMBOL"
    tf = "1d"
    for p in parts[1:]:
        pl = p.lower()
        if pl in ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"):
            tf = pl
            break
    if session not in ("none", "24h"):
        return symbol, f"{tf}_{session}"
    return symbol, tf


def finalize_ga_report(
    *,
    strategy_id: str,
    genome_str: str,
    params: dict[str, Any],
    df: pd.DataFrame,
    datafile: str,
    session: str,
    config: EngineConfig,
    raw_cfg: dict,
) -> None:
    finalize_backtest_report(
        strategy_id=strategy_id,
        genome_str=genome_str,
        params=params,
        df=df,
        datafile=datafile,
        session=session,
        config=config,
        raw_cfg=raw_cfg,
        title="Full-range results",
    )


def finalize_backtest_report(
    *,
    strategy_id: str,
    genome_str: str,
    params: dict[str, Any] | None = None,
    df: pd.DataFrame,
    datafile: str,
    session: str,
    config: EngineConfig,
    raw_cfg: dict | None = None,
    title: str = "Full-range results",
) -> dict[str, Any]:
    """Metrics block + terminal chart + optional CSV/HTML."""
    raw = raw_cfg if raw_cfg is not None else load_raw_cfg()
    rep = reporting_settings(raw)
    mod = load_strategy(strategy_id)
    if params is None:
        from coensio_algo_ai.genome_fmt import parse_genome_str

        parsed = parse_genome_str(strategy_id, genome_str, mod.PARAMS)
        params = parsed.params
        vec = list(parsed.genome)
    else:
        vec = params_to_genome_vec(mod.PARAMS, params)

    detailed = evaluate_recipe_detailed(df, strategy_id, vec, config=config)
    ts = bar_timestamps_utc_ns(pd.DatetimeIndex(df.index))
    eq = detailed.get("equity_curve")
    eq_list = eq.tolist() if hasattr(eq, "tolist") else list(eq or [])
    metrics = extended_metrics(
        detailed["metrics"],
        detailed.get("trades") or [],
        eq_list,
        initial_capital=config.initial_capital,
        timestamps=ts,
    )
    bet_label = f"${config.fixed_bet_size:.0f} {config.bet_mode}"
    print_metrics_summary(metrics, title=title, session=session, bet_label=bet_label)

    trade_list = trades_to_trade_list(detailed.get("trades") or [], df.index, genome_str)
    symbol, timeframe = infer_symbol_timeframe(Path(datafile), session)
    strategy_name = build_strategy_name(genome_str, strategy_id, symbol, timeframe)
    out_dir = _results_dir(raw)

    report_metrics = {
        **metrics,
        "strategy_name": strategy_name,
        "trade_list": trade_list,
        "session": session,
        "bet_label": bet_label,
    }

    if rep["terminal_chart"]:
        show_strategy_chart(detailed, metrics, genome_str, raw)

    if rep["qa_csv"] and trade_list:
        csv_path = out_dir / f"{strategy_name}.csv"
        write_qa_csv(csv_path, trade_list, f"{symbol}_{timeframe}")
        print(f"QA CSV: {csv_path}")

    if rep["html_report"] and trade_list:
        html_path = out_dir / f"{strategy_name}.html"
        write_html_equity_report(html_path, report_metrics)
        print(f"HTML report: {html_path}")

    return {"metrics": metrics, "detailed": detailed, "report_metrics": report_metrics}
