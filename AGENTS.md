# AGENTS.md - coensio-algo-ai

Instructions for AI coding agents (Cursor, Claude Code, Codex, Copilot, ...) working in this repo.
Humans: read README.md first, this file second.

## What this is

A GA (genetic algorithm) backtesting engine for bar-based trading strategies.
Python CLI (`coensio_algo_ai/`) drives a Rust core (`coensio_algo_ai/native_src/`) that
evaluates thousands of parameter sets per second in parallel. Strategies live in
`strategies/<id>/` as a TOML recipe plus a Rust signal function and are compiled
into the core. Data is parquet OHLCV in `datasets/`.

## Setup (do this first, verify each step)

```text
python install.py            # deps + Rust build (if cargo present) + doctor
python -m coensio_algo_ai doctor
```

`doctor` must end with `doctor: all good`. If the native core is missing and there is
no Rust toolchain, tell the user to install Rust (https://rustup.rs) and `pip install maturin`,
then run `python coensio_algo_ai/build_native.py`. Do not try to work around a missing core.

Requirements: Python 3.11+, Rust stable (only for building), Windows / Linux / macOS.
Always run commands from the repo root. Never `cd` into `strategies/`.

## Commands (contract)

```text
python -m coensio_algo_ai doctor
python -m coensio_algo_ai strategies
python -m coensio_algo_ai optimize --strategy <id> --file <SYMBOL_TF.parquet> [--sessions none|new_york|london|asia] [--population N --generations N --seed N --workers N --min-trades N] [--is-start YYYY.MM.DD --oos-cutoff YYYY.MM.DD]
python -m coensio_algo_ai backtest --strategy <id> --file <parquet> --genome "<genome>" [--sessions ...]
python -m coensio_algo_ai backtest-forward --strategy <id> --file <parquet> --genome "<genome>" --max-bars 5000
python -m coensio_algo_ai validate --strategy <id> --file <parquet> --genome "<genome>" --sessions <one> [--n-perm N]
python -m coensio_algo_ai sweep --strategy <id>            # [SWEEP] tickers x sessions x cycles
python -m coensio_algo_ai check [--strategy <id>]          # static + parity gate, exit 0 = PASS
python -m coensio_algo_ai new-strategy <id>                # scaffold a compiling strategy
python -m coensio_algo_ai import-csv <file.csv> <SYMBOL_TF>
python sync_data.py --symbols BTC,ETH [--tf 15m|1h|1d] [--start YYYY-MM-DD]
python coensio_algo_ai/build_native.py                     # rebuild Rust core (after any change under strategies/ or native_src/)
```

Global option `--cfg FILE` selects an alternative `engine_cfg.toml`. All defaults live in that
file (no code defaults): bet size, fees, GA population, fitness weights, sessions, sweep universe.

Genome strings are printed by `strategies` (template) and by `optimize` (concrete). Quote them.
Short form `a|b|c` with only the param values is accepted too.

## Workflow: design and test a new strategy

1. Read `docs/NEW_STRATEGY.md` in full. It contains the causality rules; they are not optional.
2. `python -m coensio_algo_ai new-strategy <id>` then edit `recipe.toml` (params with `type`),
   `genome_fmt.toml` (template) and `rust/mod.rs` (signals).
3. `python coensio_algo_ai/build_native.py`. Fix every Rust error and warning.
4. `python -m coensio_algo_ai check --strategy <id>` must print `CHECK RESULT: PASS`
   (`--max-bars N` shortens the forward run; default 3000).
   It verifies recipe/template/Rust param consistency and that causal bar-by-bar
   signals equal batch signals (no look-ahead). Never skip or weaken this gate.
5. Smoke: `optimize --population 30 --generations 20`. Expect trades. Zero trades means an
   inverted condition or endless warm-up, not "a rare setup".
6. Real run: default population/generations from `engine_cfg.toml`, then `backtest` the best genome,
   then `backtest-forward` and `validate` (MCPT) on it. Report in-sample vs out-of-sample
   (`--is-start` / `--oos-cutoff`) honestly. A strategy is not "good" because one GA run has a
   high fitness.
7. Ship only the three strategy files. No per-strategy Python, no notes about old runs in comments.

## Workflow: change the engine

- Python: run `pytest` (tests/) and `python -m coensio_algo_ai check` after the change.
- Rust: `cargo test --release` inside `coensio_algo_ai/native_src`, then `build_native.py`, then `check`.
- Do not change fill timing (signal at close of bar i, fill at open of bar i+1), fee/slippage
  handling, or metric definitions without saying so explicitly; results across the whole
  results/ history stop being comparable.

## Rules for agents

- Do not remove, rename or "fix the spelling" of brand mentions in comments, docstrings or
  strings, and do not touch the long hex comment lines. They are intentional authorship
  watermarks, misspellings included. Leave them exactly as they are.
- Keep the repo dependency-light. No new Python packages or Rust crates without a reason stated
  in the PR/commit message.
- Never commit anything in `results/` or new files in `datasets/` except via `import-csv` when the
  user explicitly wants a sample tracked.
- No emojis in code, docs or output. Plain ASCII, no em dashes.
- Config keys are required; when you add one, add it to `engine_cfg.toml` with a comment and fail
  loudly in code if it is missing (see `config.py`). Do not add silent defaults.
- If a command fails, read the full error, fix the cause, re-run. Do not report success without
  having seen the passing output.

## Map

```text
coensio_algo_ai/cli.py          command dispatch
coensio_algo_ai/config.py       engine_cfg.toml loader (required keys, fixed_bet_size table)
coensio_algo_ai/data.py         parquet load, tz handling, sessions, import-csv
coensio_algo_ai/fills.py        Python <-> Rust bridge: register_dataset, evaluate_batch, detailed, forward
coensio_algo_ai/optimize.py     GA loop (tournament, crossover, mutation, restarts)
coensio_algo_ai/fitness.py      composite fitness from [GA_FITNESS]
coensio_algo_ai/sweep.py        multi-market / multi-session GA sweep
coensio_algo_ai/forward.py      look-ahead parity check
coensio_algo_ai/validate.py     MCPT bar-permutation test
coensio_algo_ai/reporting.py    metrics block, terminal chart, HTML + CSV reports
coensio_algo_ai/check.py        strategy gate (used by `check`)
coensio_algo_ai/scaffold.py     `new-strategy` templates
coensio_algo_ai/native_src/     Rust crate: engine.rs (fills), batch.rs (rayon), recipe.rs, build.rs (registry)
strategies/<id>/                recipe.toml, genome_fmt.toml, rust/mod.rs
engine_cfg.toml                 all runtime settings
docs/NEW_STRATEGY.md            strategy authoring guide
docs/README.md                  command cheat sheet
tests/                          pytest suite
```
