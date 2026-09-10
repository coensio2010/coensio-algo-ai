# Command examples

Copy-paste cheat sheet. Run everything from the repo root. Dataset names are stems
in `datasets/` (`BTC_1h.parquet` ships with the repo); a full path works too.
Dates use `YYYY.MM.DD` (or `YYYY-MM-DD`).

Other docs: `../README.md` (overview), `NEW_STRATEGY.md` (writing strategies), `../AGENTS.md` (AI agents).

## Setup and health

```text
python install.py
python -m coensio_algo_ai doctor
python -m coensio_algo_ai strategies
python -m coensio_algo_ai <command> -h
```

## Data

```text
python sync_data.py --symbols BTC,ETH                     # Coinbase 1h from 2016 (resumable)
python sync_data.py --symbols BTC,ETH,SOL --tf 15m
python sync_data.py --symbols SPY,QQQ,AAPL --tf 1d        # yfinance, free
python sync_data.py --symbols AAPL --tf 1h                # yfinance caps 1h at ~2 years
python sync_data.py --symbols AVAX-USD --tf 1h            # any Coinbase product
python sync_data.py --symbols BTC --tf 1d --start 2024-01-01 --data-dir D:\data
python -m coensio_algo_ai import-csv export.csv EURUSD_1h
python -m coensio_algo_ai import-csv export.csv EURUSD_1h --tz Europe/London --sep ";"
```

## Optimize (GA)

```text
python -m coensio_algo_ai optimize --strategy donchian_atr --file BTC_1h.parquet --sessions none --population 50 --generations 100 --workers 8
python -m coensio_algo_ai optimize --strategy BS1_breakout --file BTC_1h.parquet --sessions london --population 100 --generations 250 --workers 8
python -m coensio_algo_ai optimize --strategy dual_thrust --file BTC_1h.parquet --population 100 --generations 250 --workers 8
```

In-sample / out-of-sample windows (optimize on data up to a cutoff, then backtest the rest):

```text
python -m coensio_algo_ai optimize --strategy dual_thrust --file BTC_1h.parquet --population 100 --generations 250 --oos-cutoff 2025.01.01
python -m coensio_algo_ai optimize --strategy dual_thrust --file BTC_1h.parquet --population 100 --generations 250 --is-start 2018.01.01 --oos-cutoff 2025.01.01
python -m coensio_algo_ai backtest --strategy dual_thrust --file BTC_1h.parquet --genome "<best genome>" --is-start 2025.01.01
```

Several sessions in one run, reproducible seed, override bet size and fees:

```text
python -m coensio_algo_ai optimize --strategy donchian_atr --file BTC_1h.parquet --sessions none,london,new_york --population 60 --generations 60
python -m coensio_algo_ai optimize --strategy donchian_atr --file BTC_1h.parquet --seed 7 --population 60 --generations 40
python -m coensio_algo_ai optimize --strategy donchian_atr --file BTC_1h.parquet --bet 5000 --commission 0.0002 --slippage 0.0001
```

Defaults for population, generations, seed, workers, min trades come from `engine_cfg.toml` `[GA_CONFIG]`.

## Backtest a genome

Genome strings are printed by `optimize` and by `strategies` (template). Always quote them.

```text
python -m coensio_algo_ai backtest --strategy donchian_atr --file BTC_1h.parquet --sessions none --genome "donchian_atr|both|40|20|10|50|1.2|2.0|1|48|donchian_atr_exp|fixed|2000.0|0.1|none"
python -m coensio_algo_ai backtest --strategy BS1_breakout --file BTC_1h.parquet --sessions london --genome "BS1_breakout|long|7|17|1.358|opposite_donchian|26|filter1|2|17|13|filter2|22|8|5|fixed|2000.0|0.1|london"
python -m coensio_algo_ai backtest --strategy dual_thrust --file BTC_1h.parquet --sessions london --genome "dual_thrust|both|25|0.431|0.994|31|2|9|26|2.0|flip_trail_time|11|fixed|2000.0|0.1|london"
python -m coensio_algo_ai backtest --strategy zlema_retrace --file BTC_1h.parquet --sessions london --genome "zlema_retrace|both|69|35|1.948|20|0.384|0|2.229|1.956|10|16|0|1|zlema_stop_target_time|fixed|2000.0|0.1|london"
python -m coensio_algo_ai backtest --strategy donchian_atr --file BTC_1h.parquet --sessions asia --genome "donchian_atr|both|49|21|9|59|1.059|1.0|1|27|donchian_atr_exp|fixed|2000.0|0.1|asia" --is-start 2023.01.01 --oos-cutoff 2025.01.01
```

Short form with only the parameter values (order = `[[params]]` in `recipe.toml`):

```text
python -m coensio_algo_ai backtest --strategy donchian_atr --file BTC_1h.parquet --genome "40|20|10|50|1.2|2.0|1|48"
```

Outputs: metrics block, terminal equity/drawdown chart, `results/<name>.html`, `results/<name>.csv` (toggle in `[REPORTING]`).

## Look-ahead check (forward parity)

```text
python -m coensio_algo_ai backtest-forward --strategy BS1_breakout --file BTC_1h.parquet --sessions london --genome "BS1_breakout|long|7|17|1.358|opposite_donchian|26|filter1|2|17|13|filter2|22|8|5|fixed|2000.0|0.1|london" --max-bars 5000
python -m coensio_algo_ai backtest-forward --cases --file BTC_1h.parquet --max-bars 3000      # every strategy, mid-range genome
```

Full-series forward is O(N^2); use `--max-bars` on long histories. Expect `RESULT: PASS (batch == forward)`.

## MCPT validation

```text
python -m coensio_algo_ai validate --strategy BS1_breakout --file BTC_1h.parquet --sessions london --genome "BS1_breakout|long|7|17|1.358|opposite_donchian|26|filter1|2|17|13|filter2|22|8|5|fixed|2000.0|0.1|london"
python -m coensio_algo_ai validate --strategy donchian_atr --file BTC_1h.parquet --sessions none --genome "<genome>" --n-perm 200 --alpha 0.05 --json-out results/mcpt_donchian.json
```

Defaults (`n_perm`, `alpha`, `metric`, `seed`) come from `[VALIDATION]`. Exit code 2 = FAIL.

## Sweep (many markets x sessions)

Edit `[SWEEP]` in `engine_cfg.toml` (`sweep_tickers`, `sweep_sessions`, `sweep_cycles`), then:

```text
python -m coensio_algo_ai sweep --strategy donchian_atr --population 100 --generations 250 --workers 8
python -m coensio_algo_ai sweep --strategy zlema_retrace --population 100 --generations 250 --workers 8 --oos-cutoff 2025.01.01
python -m coensio_algo_ai --cfg my_sweep.toml sweep --strategy ttm_squeeze
```

Results: `results/sweep_<strategy>_<stamp>/global_all_runs.csv|json` (unique genomes, best per market).

## Strategy development

```text
python -m coensio_algo_ai new-strategy my_breakout
python coensio_algo_ai/build_native.py
python -m coensio_algo_ai check --strategy my_breakout
python -m coensio_algo_ai check                              # all strategies
python -m coensio_algo_ai optimize --strategy my_breakout --file BTC_1h.parquet --population 30 --generations 20
```

## Alternative config

```text
copy engine_cfg.toml my_cfg.toml
python -m coensio_algo_ai --cfg my_cfg.toml optimize --strategy donchian_atr --file BTC_1h.parquet
python -m coensio_algo_ai optimize --cfg my_cfg.toml --strategy donchian_atr --file BTC_1h.parquet
```

## Tests

```text
python -m pytest
cd coensio_algo_ai/native_src && cargo test --release
```
