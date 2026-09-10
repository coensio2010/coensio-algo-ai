"""Plotly HTML + QuantAnalyzer CSV.

powered by coesnio.com
"""

from __future__ import annotations

import os
from pathlib import Path

QA_HEADER = (
    "Ticket,OpenTime,Action,Size,Symbol,OpenPrice,Unused,Unused,"
    "CloseTime,ClosePrice,Unused,Unused,PL,Unused,Unused"
)

CHART_INSET_METRICS = (
    ("total_profit", "net_pnl"),
    ("max_dd_usd", "max_dd_usd"),
    ("ret_dd_ratio", "ret_dd_ratio"),
    ("total_trades", "total_trades"),
    ("avg_trade_profit", "net_avg"),
    ("recent_year_pnl", "recent_year_pnl"),
)


def _fmt_time(ts):
    import pandas as pd

    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return ""
    if hasattr(ts, "strftime"):
        return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)


def write_qa_csv(filepath, trade_rows, strategy_id, direction=1):
    lines = [QA_HEADER]
    for idx, row in enumerate(trade_rows):
        # Prefer per-trade direction when present (index 7); else report-level default.
        trade_dir = int(row[7]) if len(row) > 7 else int(direction)
        action = "Buy" if trade_dir >= 0 else "Sell"
        # Size = units traded (notional / fill price), index 8 when present.
        size = float(row[8]) if len(row) > 8 else 1.0
        line = (
            f"{idx + 1},"
            f"{_fmt_time(row[0])},"
            f"{action},"
            f"{size:.6f},"
            f"{strategy_id},"
            f"${float(row[2]):.2f},"
            f",,"
            f"{_fmt_time(row[1])},"
            f"${float(row[3]):.2f},"
            f",,"
            f"${float(row[5]):.2f},"
            f",,"
        )
        lines.append(line)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _yearly_pnl_and_dd(df_trades) -> tuple[list[int], list[float], list[float]]:
    """Sum PnL by exit year; max running equity DD observed in each exit year."""
    yearly_pnl = df_trades.groupby(df_trades["ExitTime"].dt.year)["PnL"].sum()
    yearly_dd: dict[int, float] = {}
    peak = 0.0
    cum = 0.0
    for row in df_trades.sort_values("ExitTime").itertuples(index=False):
        cum += float(row.PnL)
        peak = max(peak, cum)
        dd = peak - cum
        year = int(row.ExitTime.year)
        yearly_dd[year] = max(yearly_dd.get(year, 0.0), dd)
    years = sorted(set(yearly_pnl.index.astype(int)) | set(yearly_dd.keys()))
    pnl_vals = [float(yearly_pnl.get(y, 0.0)) for y in years]
    dd_vals = [float(yearly_dd.get(y, 0.0)) for y in years]
    return years, pnl_vals, dd_vals


def _long_short_stats(df_trades) -> tuple[list[str], list[float], list[int]]:
    """Return (labels, pnl_by_side, count_by_side) for LONG / SHORT."""
    labels = ["LONG", "SHORT"]
    if "Direction" not in df_trades.columns or df_trades.empty:
        return labels, [0.0, 0.0], [0, 0]
    long_mask = df_trades["Direction"].astype(int) > 0
    short_mask = ~long_mask
    pnl = [
        float(df_trades.loc[long_mask, "PnL"].sum()) if long_mask.any() else 0.0,
        float(df_trades.loc[short_mask, "PnL"].sum()) if short_mask.any() else 0.0,
    ]
    counts = [int(long_mask.sum()), int(short_mask.sum())]
    return labels, pnl, counts


def _weekday_stats(df_trades) -> tuple[list[str], list[float], list[int]]:
    """PnL and trade count by entry weekday (Mon..Sun)."""
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if df_trades.empty:
        return labels, [0.0] * 7, [0] * 7
    wd = df_trades["EntryTime"].dt.dayofweek  # Mon=0 .. Sun=6
    pnl = []
    counts = []
    for i in range(7):
        mask = wd == i
        pnl.append(float(df_trades.loc[mask, "PnL"].sum()) if mask.any() else 0.0)
        counts.append(int(mask.sum()))
    return labels, pnl, counts


def _chart_inset_annotation(metrics: dict) -> dict:
    from coensio_algo_ai.reporting import METRIC_FMT

    lines: list[str] = []
    for key, label in CHART_INSET_METRICS:
        if key not in metrics:
            continue
        try:
            text = METRIC_FMT.get(key, str)(metrics[key])
        except (TypeError, ValueError):
            text = str(metrics[key])
        lines.append(f"{label}: {text}")
    return dict(
        text="<br>".join(lines),
        xref="x domain",
        yref="y domain",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        font=dict(size=10, family="Consolas, monospace"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cccccc",
        borderwidth=1,
        borderpad=6,
    )


def _fig_div(fig, width: int, height: int, *, include_plotlyjs: bool) -> str:
    html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        config={"responsive": False},
    )
    return html.replace(
        'style="height:100%; width:100%;"',
        f'style="height:{height}px; width:{width}px;"',
    )


def _metrics_table_html(metric_rows: list[tuple[str, str]], html_lib) -> str:
    if not metric_rows:
        return ""
    mid = (len(metric_rows) + 1) // 2
    chunks = [metric_rows[:mid], metric_rows[mid:]]
    tables: list[str] = []
    for chunk in chunks:
        labels = "".join(f"<th>{html_lib.escape(label)}</th>" for label, _ in chunk)
        values = "".join(f"<td>{html_lib.escape(text)}</td>" for _, text in chunk)
        tables.append(
            '  <table class="metrics">\n'
            f"    <tr>{labels}</tr>\n"
            f"    <tr>{values}</tr>\n"
            "  </table>"
        )
    return '<div class="metrics-wrap">\n' + "\n".join(tables) + "\n</div>"


def write_html_equity_report(filepath, metrics, oos_window_boundaries=None):
    """HTML: metrics tables + Equity/DD + yearly Profit/DD charts."""
    import html as html_lib

    try:
        import pandas as pd
        import plotly.graph_objects as go
    except ImportError as exc:
        print(f"HTML report: install plotly/pandas ({exc})")
        return

    from coensio_algo_ai.reporting import metrics_summary_rows

    trade_list = metrics.get("trade_list") or []
    if not trade_list:
        return

    # Backward compatible: older 7-col lists have no Direction (treat as LONG).
    n_cols = len(trade_list[0]) if trade_list else 0
    if n_cols >= 8:
        columns = [
            "EntryTime",
            "ExitTime",
            "EntryPrice",
            "ExitPrice",
            "ExitType",
            "PnL",
            "Params",
            "Direction",
        ]
        if n_cols >= 9:
            columns.append("Size")  # units traded = notional / fill price
    else:
        columns = [
            "EntryTime",
            "ExitTime",
            "EntryPrice",
            "ExitPrice",
            "ExitType",
            "PnL",
            "Params",
        ]
    df_trades = pd.DataFrame(trade_list, columns=columns)
    if "Direction" not in df_trades.columns:
        df_trades["Direction"] = 1
    df_trades["EntryTime"] = pd.to_datetime(df_trades["EntryTime"])
    df_trades["ExitTime"] = pd.to_datetime(df_trades["ExitTime"])
    df_trades = df_trades.sort_values("EntryTime")
    df_trades["CumulativePnL"] = df_trades["PnL"].cumsum()
    df_trades["PeakPnL"] = df_trades["CumulativePnL"].cummax()
    # de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    df_trades["Drawdown"] = df_trades["PeakPnL"] - df_trades["CumulativePnL"]

    date_range = ""
    if not df_trades.empty:
        date_range = (
            f"{df_trades['EntryTime'].iloc[0].strftime('%Y %m %d')} to "
            f"{df_trades['ExitTime'].iloc[-1].strftime('%Y %m %d')}"
        )

    chart_px = 420  # 30% smaller than prior 600px
    chart_gap = 20
    charts_per_row = 3
    report_w = charts_per_row * chart_px + (charts_per_row - 1) * chart_gap
    years, yearly_pnl, yearly_dd = _yearly_pnl_and_dd(df_trades)
    pnl_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in yearly_pnl]
    year_min = int(df_trades["EntryTime"].dt.year.min())
    year_max = int(df_trades["ExitTime"].dt.year.max())
    year_ticks = list(range(year_min, year_max + 1))
    year_tickvals = [pd.Timestamp(f"{y}-07-01") for y in year_ticks]
    year_ticktext = [str(y) for y in year_ticks]
    chart_layout = dict(
        width=chart_px + 20,
        height=chart_px,
        autosize=False,
        showlegend=False,
        hovermode="x unified",
        margin=dict(t=36, b=52, l=56, r=16),
    )
    year_axis = dict(
        tickvals=year_tickvals,
        ticktext=year_ticktext,
        tickangle=0,
        type="date",
    )

    fig_eq = go.Figure()
    fig_eq.add_trace(
        go.Scatter(
            x=df_trades["EntryTime"],
            y=df_trades["CumulativePnL"],
            mode="lines",
            name="Equity",
            line=dict(color="#2563eb", width=2.5),
        )
    )
    fig_eq.add_trace(
        go.Scatter(
            x=df_trades["EntryTime"],
            y=-df_trades["Drawdown"],
            mode="lines",
            name="Drawdown",
            line=dict(color="#dc2626", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(220,38,38,0.25)",
        )
    )
    if oos_window_boundaries:
        x_min = df_trades["EntryTime"].iloc[0]
        x_max = df_trades["ExitTime"].iloc[-1]
        eq_ymin = float(-df_trades["Drawdown"].max())
        eq_ymax = float(df_trades["CumulativePnL"].max())
        for b in oos_window_boundaries:
            ts = pd.to_datetime(b)
            if x_min <= ts <= x_max:
                fig_eq.add_shape(
                    type="line",
                    x0=ts,
                    x1=ts,
                    y0=eq_ymin,
                    y1=eq_ymax,
                    line=dict(color="gray", width=1, dash="dot"),
                )
    fig_eq.update_layout(
        title="Equity + Drawdown",
        annotations=[_chart_inset_annotation(metrics)],
        **chart_layout,
    )
    fig_eq.add_hline(y=0, line_width=1, line_color="#999")
    fig_eq.update_xaxes(**year_axis)
    fig_eq.update_yaxes(title_text="PnL / DD ($)", zeroline=True)

    fig_yr = go.Figure()
    fig_yr.add_trace(
        go.Bar(
            x=years,
            y=yearly_pnl,
            name="Profit / Year",
            marker_color=pnl_colors,
            opacity=0.9,
        )
    )
    fig_yr.add_trace(
        go.Bar(
            x=years,
            y=[-v for v in yearly_dd],
            name="Max DD / Year",
            marker_color="#dc2626",
            opacity=0.75,
        )
    )
    fig_yr.update_layout(
        title="Profit + Max DD / Year",
        barmode="overlay",
        **chart_layout,
    )
    fig_yr.add_hline(y=0, line_width=1, line_color="#999")
    fig_yr.update_xaxes(type="category", tickangle=0)
    fig_yr.update_yaxes(title_text="PnL / DD ($)", zeroline=True)

    side_labels, side_pnl, side_counts = _long_short_stats(df_trades)
    side_pnl_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in side_pnl]
    side_count_colors = ["#2563eb", "#f59e0b"]

    fig_ls_pnl = go.Figure()
    fig_ls_pnl.add_trace(
        go.Bar(
            x=side_labels,
            y=side_pnl,
            name="PnL",
            marker_color=side_pnl_colors,
            opacity=0.9,
            text=[f"${v:,.0f}" for v in side_pnl],
            textposition="outside",
        )
    )
    fig_ls_pnl.update_layout(
        title="LONG vs SHORT PnL",
        **{**chart_layout, "margin": dict(t=48, b=52, l=56, r=16)},
    )
    fig_ls_pnl.add_hline(y=0, line_width=1, line_color="#999")
    fig_ls_pnl.update_xaxes(type="category", tickangle=0)
    fig_ls_pnl.update_yaxes(title_text="PnL ($)", zeroline=True)

    fig_ls_n = go.Figure()
    fig_ls_n.add_trace(
        go.Bar(
            x=side_labels,
            y=side_counts,
            name="Trades",
            marker_color=side_count_colors,
            opacity=0.9,
            text=[str(v) for v in side_counts],
            textposition="outside",
        )
    )
    fig_ls_n.update_layout(
        title="LONG vs SHORT #",
        **{**chart_layout, "margin": dict(t=48, b=52, l=56, r=16)},
    )
    fig_ls_n.update_xaxes(type="category", tickangle=0)
    fig_ls_n.update_yaxes(title_text="Trades", zeroline=True)

    wd_labels, wd_pnl, wd_counts = _weekday_stats(df_trades)
    wd_pnl_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in wd_pnl]
    wd_count_colors = ["#2563eb"] * len(wd_labels)

    fig_wd_pnl = go.Figure()
    fig_wd_pnl.add_trace(
        go.Bar(
            x=wd_labels,
            y=wd_pnl,
            name="PnL",
            marker_color=wd_pnl_colors,
            opacity=0.9,
            text=[f"${v:,.0f}" for v in wd_pnl],
            textposition="outside",
        )
    )
    fig_wd_pnl.update_layout(
        title="PnL per WeekDay",
        **{**chart_layout, "margin": dict(t=48, b=52, l=56, r=16)},
    )
    fig_wd_pnl.add_hline(y=0, line_width=1, line_color="#999")
    fig_wd_pnl.update_xaxes(type="category", tickangle=0)
    fig_wd_pnl.update_yaxes(title_text="PnL ($)", zeroline=True)

    fig_wd_n = go.Figure()
    fig_wd_n.add_trace(
        go.Bar(
            x=wd_labels,
            y=wd_counts,
            name="Trades",
            marker_color=wd_count_colors,
            opacity=0.9,
            text=[str(v) for v in wd_counts],
            textposition="outside",
        )
    )
    fig_wd_n.update_layout(
        title="Trades per WeekDay #",
        **{**chart_layout, "margin": dict(t=48, b=52, l=56, r=16)},
    )
    fig_wd_n.update_xaxes(type="category", tickangle=0)
    fig_wd_n.update_yaxes(title_text="Trades", zeroline=True)

    strategy_name = metrics.get("strategy_name", "Strategy")
    session = str(metrics.get("session", "none"))
    bet_label = str(metrics.get("bet_label", ""))
    metric_rows = metrics_summary_rows(metrics, session=session, bet_label=bet_label)
    table_html = _metrics_table_html(metric_rows, html_lib)

    eq_chart_html = _fig_div(fig_eq, chart_px, chart_px, include_plotlyjs=True)
    yr_chart_html = _fig_div(fig_yr, chart_px, chart_px, include_plotlyjs=False)
    ls_pnl_html = _fig_div(fig_ls_pnl, chart_px, chart_px, include_plotlyjs=False)
    ls_n_html = _fig_div(fig_ls_n, chart_px, chart_px, include_plotlyjs=False)
    wd_pnl_html = _fig_div(fig_wd_pnl, chart_px, chart_px, include_plotlyjs=False)
    wd_n_html = _fig_div(fig_wd_n, chart_px, chart_px, include_plotlyjs=False)

    page = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html_lib.escape(strategy_name)} | coesnio</title>
  <style>
    body {{
      font-family: Consolas, "Courier New", monospace;
      font-size: 13px;
      margin: 20px;
      color: #111;
    }}
    .report {{
      width: {report_w}px;
      margin: 0 auto;
    }}
    h2 {{
      font-size: 13px;
      font-weight: 600;
      margin: 0 0 6px 0;
      line-height: 1.35;
      word-break: break-all;
      text-align: center;
    }}
    .period {{
      color: #666;
      margin-bottom: 10px;
      text-align: center;
    }}
    .charts-row {{
      display: flex;
      align-items: stretch;
      height: {chart_px}px;
      gap: {chart_gap}px;
      justify-content: center;
      margin-bottom: {chart_gap}px;
    }}
    .chart-panel {{
      width: {chart_px}px;
      height: {chart_px}px;
      flex-shrink: 0;
    }}
    .metrics-wrap {{
      margin: 8px 0 14px 0;
      text-align: center;
    }}
    .metrics {{
      border-collapse: collapse;
      margin: 0 auto 10px auto;
      font-size: 12px;
    }}
    .metrics th {{
      color: #555;
      font-weight: 600;
      padding: 5px 12px;
      text-align: center;
      border-bottom: 1px solid #ddd;
      white-space: nowrap;
    }}
    .metrics td {{
      padding: 5px 12px;
      text-align: center;
      white-space: nowrap;
    }}
    .brand {{
      margin-top: 16px;
      text-align: center;
      color: #888;
      font-size: 11px;
    }}
  </style>
</head>
<body>
  <div class="report">
    <h2>{html_lib.escape(strategy_name)}</h2>
    <div class="period">{html_lib.escape(date_range)}</div>
    {table_html}
    <div class="charts-row">
      <div class="chart-panel">{eq_chart_html}</div>
      <div class="chart-panel">{yr_chart_html}</div>
      <div class="chart-panel">{ls_pnl_html}</div>
    </div>
    <div class="charts-row">
      <div class="chart-panel">{ls_n_html}</div>
      <div class="chart-panel">{wd_pnl_html}</div>
      <div class="chart-panel">{wd_n_html}</div>
    </div>
    <div class="brand">powered by <a href="https://coensio.com">coensio.com</a> &middot; copyright coesnio.com &middot; see coesnio</div>
  </div>
</body>
</html>
"""

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
