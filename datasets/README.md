# datasets/

One parquet file per market and timeframe: `<SYMBOL>_<tf>.parquet` (`BTC_1h`, `SPY_15m`, `AAPL_1d`).

Contract:

- `DatetimeIndex`, tz-aware UTC preferred. Naive timestamps are localized by asset class: crypto stems (BTC, ETH, SOL, ...) as UTC, everything else as America/New_York. Override with `[SESSION].data_tz`.
- float64 columns `open`, `high`, `low`, `close`, `volume`.
- Bars sorted, no duplicate timestamps. Malformed bars are repaired or dropped on load (`data_clean.py`).

`BTC_1h.parquet` is bundled (Coinbase BTC-USD, hourly, 2016-01-01 onward, 92k bars) so the quickstart works offline. Everything else is ignored by git.

Get data:

```text
python sync_data.py --symbols BTC,ETH            # Coinbase, full history from 2016
python sync_data.py --symbols SPY,AAPL --tf 1d   # yfinance (free); 1h is limited to ~2 years by Yahoo
python -m coensio_algo_ai import-csv my_export.csv EURUSD_1h
```
