# coensio-algo-ai

Ultra fast backtesting and strategy research for bar-based trading systems.
Python CLI, Rust core, genetic-algorithm optimizer, built so that an AI coding
agent can install it, design a strategy, optimize it and prove it has no
look-ahead, all from the command line.

Powered by [coensio.com](https://coensio.com).

## Why

- **Speed.** Signals and fills run in Rust with Rayon. 125 genomes on 40k hourly bars evaluate in about 40 ms on an 8-core laptop, so a 250-generation GA finishes in seconds, not hours.
- **Honest fills.** Signal on the close of bar `i`, fill at the open of bar `i+1`, per-side commission and slippage, fixed notional bet. No same-bar magic.
- **Look-ahead gate.** `backtest-forward` recomputes signals bar by bar on a growing prefix and requires a 1:1 match with the batch run. `check` runs it for you.
- **Agent friendly.** One config file, one CLI, explicit contracts, `AGENTS.md`, a scaffold command and a consistency checker. Tell your agent "install coensio-algo-ai from GitHub and build me a Keltner retest strategy" and it can.
- **Strategies as small Rust plugins.** One folder: `recipe.toml` (params), `genome_fmt.toml` (genome string), `rust/mod.rs` (signals). The build scans the folder; there is no registry to edit.

## Quickstart

Requirements: Python 3.11+, git. Rust (https://rustup.rs) only if you need to rebuild the core (Windows x64 binary is included).

```text
git clone https://github.com/coensio2010/coensio-algo-ai.git
cd coensio-algo-ai
python -m venv .venv              # optional but recommended; then activate it
python install.py                 # pip deps, native build if needed, doctor (--skip-native, --rebuild, --data)
python -m coensio_algo_ai doctor  # everything OK?
python -m coensio_algo_ai strategies
```

BTC and ETH hourly data (Coinbase, 2016 onward, ~90k bars each) ship in `datasets/`, so this works offline:

```text
python -m coensio_algo_ai optimize --strategy donchian_atr --file BTC_1h.parquet --sessions none --population 60 --generations 40 --seed 7
```

Runs in about 3 seconds on 8 cores. One progress line per generation, then the result table:

```text
GA | donchian_atr | BTC_1h.parquet | session=none | 92,402 bars | pop=60 gen=40 | workers=8 (Rust Rayon) | seed=7 | coesnio
C:01G:040T:00040/00040 rdd:17.63 net_pnl:$8377.62 DD:$475.28 net_avg:$13.13 #638 stab:0.754 ryp:$-360 avg_roundtrip_fee:$1.81 total_fees:$1154.78 sess:none bet:$2000 fixed best_fit:17.627 0.59ms/str

Top 10 unique genomes
  rdd | net_pnl |  DD | net_avg |   # |  stab |  ryp | avg_fee | total_fees | genome                                                                               | datafile
17.63 |    8378 | 475 |   13.13 | 638 | 0.754 | -360 |    1.81 |    1154.78 | donchian_atr|both|35|19|8|30|1.093|2.327|0|20|donchian_atr_exp|fixed|2000.0|0.1|none | BTC_1h.parquet
17.61 |    8368 | 475 |   13.12 | 638 | 0.754 | -360 |    1.81 |    1154.78 | donchian_atr|both|35|20|8|30|1.093|2.327|0|20|donchian_atr_exp|fixed|2000.0|0.1|none | BTC_1h.parquet
```

followed by the full metrics block, a terminal equity / drawdown chart, an HTML report and a trade CSV in `results/`. (Yes, that genome is over-fitted to 10 years of BTC, and it lost money in the most recent year. Read the disclaimer.)

Backtest a genome (copy it from the optimize output):

```text
python -m coensio_algo_ai backtest --strategy donchian_atr --file BTC_1h.parquet --genome "donchian_atr|both|40|20|10|50|1.2|2.0|1|48|donchian_atr_exp|fixed|2000.0|0.1|none"
```

```text
Full-range results
  session: none
  bet_size: $2000 fixed
  net_pnl: 1124.81
  max_dd_usd: 2138.84
  ret_dd_ratio: 0.53
  total_trades: 699
  win_rate: 39.06
  net_avg: 1.61
  total_fees: 1259.37
  profit_factor: 1.05
  sharpe: 0.168
  stability_r2: 0.2430
  ...
```

A GA-found genome on the bundled ETH data, London session (its report ships in `results/`, open the `.html`):

```text
python -m coensio_algo_ai backtest --strategy zlema_retrace --file ETH_1h.parquet --sessions london --genome "zlema_retrace|both|69|35|1.948|20|0.384|0|2.229|1.956|10|16|0|1|zlema_stop_target_time|fixed|2000.0|0.1|london"
```

220 trades, net_pnl 4846.03, max_dd_usd 118.24 over 2016-2026. In-sample; treat it as a demo of the tooling, not a trading recommendation.

## Commands

| Command | Purpose |
|---|---|
| `doctor` | verify Python, Rust, native core, config, data |
| `strategies` | list strategies with their genome templates |
| `optimize` | GA on one dataset, one or more sessions |
| `sweep` | GA over `[SWEEP]` tickers x sessions x cycles, global unique genome table |
| `backtest` | one genome: metrics, terminal chart, HTML + CSV report |
| `backtest-forward` | look-ahead check: causal bar-by-bar == batch |
| `validate` | Monte Carlo permutation test (MCPT) p-value for a genome |
| `check` | static + parity gate for a strategy folder |
| `new-strategy` | scaffold a compiling strategy folder |
| `import-csv` | convert a CSV export into `datasets/` |

Run `python -m coensio_algo_ai <command> -h` for options. Global `--cfg FILE` selects another config.

Common options: `--sessions none|new_york|london|asia` (session chart filter), `--is-start YYYY.MM.DD` / `--oos-cutoff YYYY.MM.DD` (date window for in-sample / out-of-sample splits), `--bet`, `--commission`, `--slippage` overrides.

## Data

```text
python sync_data.py --symbols BTC,ETH,SOL              # Coinbase, hourly, from 2016
python sync_data.py --symbols SPY,QQQ,AAPL --tf 1d     # yfinance (free)
python sync_data.py --symbols AAPL --tf 1h             # yfinance: Yahoo caps 1h at ~2 years, 15m at ~60 days
python -m coensio_algo_ai import-csv export.csv EURUSD_1h
```

Files are `datasets/<SYMBOL>_<tf>.parquet` with a UTC `DatetimeIndex` and `open, high, low, close, volume`. Syncs are resumable. Details in `datasets/README.md`.

## Configuration

Everything lives in `engine_cfg.toml` and every key is required (no hidden code defaults):

- `[ENGINE]` capital, bet mode, `fixed_bet_size` (`default` + per-dataset overrides), commission, slippage
- `[GA_CONFIG]` population, generations, restarts, crossover / mutation, workers, seed
- `[GA_FITNESS]` weighted metrics with caps and gates (default: return / drawdown ratio)
- `[SESSION]` wall-clock presets (DST aware) or fixed UTC windows
- `[SWEEP]`, `[VALIDATION]`, `[REPORTING]`

Copy the file and pass `--cfg my.toml` to keep several setups.

## Writing a strategy

```text
python -m coensio_algo_ai new-strategy my_breakout
# edit strategies/my_breakout/recipe.toml, genome_fmt.toml, rust/mod.rs
python coensio_algo_ai/build_native.py
python -m coensio_algo_ai check --strategy my_breakout      # must PASS
python -m coensio_algo_ai optimize --strategy my_breakout --file BTC_1h.parquet --population 30 --generations 20
```

Read `docs/NEW_STRATEGY.md` before writing signal code. It lists the causality rules the gate enforces. Agents: `AGENTS.md`.

Included strategies (all long/short capable unless noted): `BS1_breakout` (POI breakout with switchable filters, long), `donchian_atr`, `donchian_chan`, `dual_thrust`, `ib_retrace`, `keltner_retrace`, `nr_expand`, `orb_retrace`, `poc_retrace`, `ttm_squeeze`, `zlema_retrace`. They are research examples, not trading advice.

## Repository layout

```text
coensio_algo_ai/          Python package (CLI, GA, reporting, data)
coensio_algo_ai/native_src/  Rust crate (fills, batch evaluation, recipe registry)
coensio_algo_ai/native/   compiled core (engine_core.pyd / .so / .dylib)
strategies/<id>/          recipe.toml, genome_fmt.toml, rust/mod.rs
datasets/                 parquet OHLCV (BTC_1h and ETH_1h 2016+ included)
results/                  reports (git-ignored, one example report bundled)
engine_cfg.toml           all settings
tests/                    pytest
docs/NEW_STRATEGY.md      strategy authoring guide
docs/README.md            command cheat sheet
AGENTS.md                 instructions for AI agents
```

## Development

```text
python install.py                           # installs pytest and maturin too
python -m pytest
python -m coensio_algo_ai check --max-bars 1500   # look-ahead gate, all strategies
cd coensio_algo_ai/native_src
cargo test --release
```

## Disclaimer

This is research software. Backtests are hypothetical, ignore market impact, borrow costs and many real-world constraints, and are trivially over-fitted by a GA. Nothing here is investment advice. Use `--oos-cutoff`, `validate` and `sweep` before believing any number.

## License

MIT. See `LICENSE`.
