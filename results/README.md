# results/

Every `backtest`, `optimize`, `sweep` and `validate` run writes here (HTML report, trade CSV, sweep JSON/CSV). The folder is git-ignored except for one bundled example so you can see what a report looks like before running anything.

Example: `zlema_retrace` on ETH 1h, London session, 2016-2026. Open the `.html` in a browser (equity, drawdown, trade table; plotly loaded from CDN).

Reproduce it, then change the genome or try another market:

```text
python -m coensio_algo_ai backtest --strategy zlema_retrace --file ETH_1h.parquet --sessions london --genome "zlema_retrace|both|69|35|1.948|20|0.384|0|2.229|1.956|10|16|0|1|zlema_stop_target_time|fixed|2000.0|0.1|london"
```

Expected: 220 trades, net_pnl 4846.03, max_dd_usd 118.24, ret_dd_ratio 40.98. That genome came out of a GA sweep on this exact data, so it is in-sample; run `validate` (MCPT) and `backtest-forward` on it before drawing conclusions.

Trade CSV columns: Ticket, OpenTime, Action (Buy/Sell), Size (units = notional / fill price), Symbol, OpenPrice, CloseTime, ClosePrice, PL (net, USD). Fill prices include slippage; PL includes commissions.
