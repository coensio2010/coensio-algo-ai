# New Strategy Guide

How to design and add a trading strategy to coensio-algo-ai. Written for AI
agents and humans alike. Keep ids stable, files minimal, signals causal.

The fastest path:

```text
python -m coensio_algo_ai new-strategy my_breakout      # scaffold (compiles as-is)
# edit strategies/my_breakout/{recipe.toml, genome_fmt.toml, rust/mod.rs}
python coensio_algo_ai/build_native.py                  # rebuild Rust core
python -m coensio_algo_ai check --strategy my_breakout  # static + look-ahead gate, must PASS
python -m coensio_algo_ai optimize --strategy my_breakout --file BTC_1h.parquet --population 30 --generations 20
```

---

## 1. Layout

One folder under `strategies/`:

```text
strategies/<strategy_id>/
  recipe.toml        identity + GA params (name, type, min, max)
  genome_fmt.toml    human genome string template
  rust/mod.rs        signal logic (Rust), exports build_signals()
```

Rules:

- `<strategy_id>` is snake_case (lowercase letters, digits, underscores, starts with a letter). It becomes a Rust module name.
- `id` in `recipe.toml` == `[plugin].type` == folder name.
- Do not start the folder name with `_` (ignored).
- Never hand-edit a registry: `build.rs` scans `strategies/` on every build.

---

## 2. Design first, then code

Write 5 lines before touching Rust:

1. **Structure**: the levels / indicators that define the setup (channel, MA, volume profile, opening range).
2. **Arm / filter** (optional): when a setup becomes eligible.
3. **Entry**: the exact close-of-bar condition for long and/or short.
4. **Exit**: stop, target, flip, time, session reset.
5. **Params**: every tunable number with an honest min/max for the timeframe you will optimize.

Put those lines as the comment header of `recipe.toml` and `rust/mod.rs`. No marketing, no citations of blogs, no history of previous versions.

### Engine contract

| Signal decided | Fill executed |
|---|---|
| on the **close** of bar `i` | at bar `i+1` **open** (plus slippage, commission) |

- One position at a time. Inside the loop check exits before entries; never enter on an exit bar.
- `buffers.entries[i] = true` opens, `buffers.exits[i] = true` closes, `buffers.directions[i] = 1 | -1` sets long/short for the entry.
- Skip bars with non-finite OHLC. Warm up indicators before arming.

---

## 3. Look-ahead rules (hard requirements)

Every one of these came from a real bug that looked profitable until fixed.

**Levels must be prior-window.** When testing `close[i]` against a level, the level must not contain bar `i`:

- Channel: `max(high[i-N .. i-1])`, never `..= i`.
- Rolling profile / VWAP / POC: bars `[i-N .. i-1]`, stored so it is usable on bar `i`.
- Fast adaptive MAs that hug the close: compare with `ma[i-1]`.

**Arm / retest logic:** arm on a close-confirmed break of a prior level, never enter on the arm bar, freeze the armed level and ATR at arm time.

**Intrabar:** use close-based stops/targets for signal flags. Do not assume you know the path of the bar (which came first, high or low).

**Sessions:** on day change clear pending state (and flatten if the design says so) using only data known at that bar. `ctx.session_days[i]` gives the session index; `spec.session_utc_start` is the configured session open.

**Prove it:**

```text
python -m coensio_algo_ai check --strategy <id>                  # mid-range genome, last 3000 bars
python -m coensio_algo_ai backtest-forward --strategy <id> --file BTC_1h.parquet --genome "..." --max-bars 5000
```

The forward path re-runs your plugin on a growing prefix `[0..i]` and keeps only the signal for bar `i`. Any mismatch with the batch run means the code reads future bars. Fix it before trusting any GA result. Full-series forward is O(N^2); use `--max-bars` on long histories.

Scope of the gate: it detects reads of bars `> i` (e.g. `close[i + 1]`, a window ending after `i`). Reading bar `i` itself is causal (the signal is acted on at bar `i+1` open) and passes; whether your logic should use `x[i]` or `x[i-1]` is a design choice, not a look-ahead question.

---

## 4. `recipe.toml`

```toml
# my_breakout - prior 20..60 bar channel break with ATR stop/target and time exit.
id = "my_breakout"

[plugin]
type = "my_breakout"
direction = "both"            # long / short decided inside rust/mod.rs
exit_method = "atr_stop_time" # label only, shows in genome strings
session_utc_start = [14, 30]  # session open (UTC) for strategies that reset per day

[[params]]
name = "lookback"
type = "int"
min = 10
max = 60

[[params]]
name = "stop_mult"
type = "float"
min = 1.0
max = 3.0
```

Rules:

- Every gene: `name`, `type` (`"int"` or `"float"`), `min`, `max`. Optional `step`, `default`.
- `type` is the single source of truth: `int` genes are sampled and mutated as integers, `float` genes continuously. It must match the Rust read helper (`param_i32` for int, `param_f64` for float). `check` verifies this.
- Flags are `int` with `min = 0`, `max = 1` (for example `allow_short`).
- Param order in `[[params]]` = genome vector order = template order.
- Fewer params optimize better. 6 to 12 is the sweet spot. Do not add a gene the Rust code never reads.

---

## 5. `genome_fmt.toml`

```toml
template = "my_breakout|{direction}|{lookback}|{stop_mult}|atr_stop_time|{bet_mode}|{fixed_bet_size}|{price_bet_frac}|{session}"
```

Rules:

- Starts with the literal strategy id, then `{direction}`.
- Contains every recipe param, same order as `[[params]]`.
- Then the exit label (any literal), then always `{bet_mode}|{fixed_bet_size}|{price_bet_frac}|{session}`.
- Integer formatting comes from the recipe `type`; there is no separate integer list.

---

## 6. `rust/mod.rs`

Required export and imports:

```rust
use crate::data::MarketData;
use crate::recipe::{PluginSpec, Recipe};
use crate::runtime::{BatchContext, SignalBuffers};

pub fn build_signals(
    data: &MarketData,
    recipe: &Recipe,
    genome: &[f64],
    spec: &PluginSpec,
    ctx: &BatchContext,
    buffers: &mut SignalBuffers,
) {
    let n = data.len();
    buffers.resize(n);
    buffers.entries.fill(false);
    buffers.exits.fill(false);
    buffers.directions.fill(1);

    // 1) params           let lookback = param_i32(recipe, genome, "lookback", 20).max(2) as usize;
    // 2) indicators       causal arrays, value at i uses bars <= i
    // 3) bar loop         exits first, then entries; single position state
}
```

Param helpers (copy verbatim into every plugin, `check` greps for them):

```rust
fn param_i32(recipe: &Recipe, genome: &[f64], name: &str, default: i32) -> i32 {
    recipe.try_param_value(genome, name).map(|v| v.round() as i32).unwrap_or(default)
}

fn param_f64(recipe: &Recipe, genome: &[f64], name: &str, default: f64) -> f64 {
    recipe.try_param_value(genome, name).unwrap_or(default)
}
```

Available data: `data.open/high/low/close/volume/timestamp` (`Vec<f64>`, timestamp in ns or s), `data.len()`, `ctx.session_days` (`Option<Arc<Vec<i32>>>`), `spec.session_utc_start`.

Keep indicator helpers private in the same file (ATR, SMA, EMA, ...). No external crates. Add at least one `#[cfg(test)]` unit test that feeds a synthetic series and asserts an entry fires where expected. Test helpers: `MarketData::new(ts, open, high, low, close, volume)`, `crate::recipe::get_recipe("<id>")`, `SignalBuffers::default()`, `BatchContext::default()`; the scaffold's `mod tests` shows the exact call. Run it with `cargo test --release <id>` from `coensio_algo_ai/native_src`.

---

## 7. Build, check, optimize

```text
python coensio_algo_ai/build_native.py             # needs Rust + maturin, ~20 s incremental
python -m coensio_algo_ai strategies               # your id must be listed with its template
python -m coensio_algo_ai check --strategy <id>    # must print CHECK RESULT: PASS
python -m coensio_algo_ai optimize --strategy <id> --file BTC_1h.parquet --sessions none --population 30 --generations 20
python -m coensio_algo_ai backtest --strategy <id> --file BTC_1h.parquet --genome "<best genome from optimize>"
```

If the build says the binary is locked (Windows), close the Python process that imported it and rebuild.

Expect a sensible mid-range genome to produce trades on BTC_1h. Zero trades usually means an inverted condition or a warm-up that never ends.

---

## 8. Checklist

- [ ] `strategies/<id>/` has exactly `recipe.toml`, `genome_fmt.toml`, `rust/mod.rs`
- [ ] `id` / `[plugin].type` / folder name identical
- [ ] every param has `type`, is read in Rust with the matching helper, nothing unread
- [ ] template order == `[[params]]` order, ends with the four fixed fields
- [ ] prior-window levels, no entry on arm bar, exits before entries
- [ ] native rebuilt, `strategies` lists it, `check` PASS
- [ ] smoke `optimize` produces trades without a panic

## 9. Do not

- Add per-strategy Python wrappers or edit generated registry files.
- Put absolute machine paths, dates of previous runs, or "champion genome" notes in strategy files.
- Invent engine APIs. Stick to `MarketData`, `Recipe`, `PluginSpec`, `BatchContext`, `SignalBuffers`.
- Judge a strategy on one in-sample GA run. Use `--is-start/--oos-cutoff` splits, `validate` (MCPT) and `sweep` across markets.
