"""`new-strategy`: scaffold strategies/<id>/{recipe.toml, genome_fmt.toml, rust/mod.rs}.

The generated strategy is a complete, causal, compiling example (prior-window
channel breakout with ATR stop / target / time exit). Edit the Rust loop and the
params, then rebuild with `python coensio_algo_ai/build_native.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")

RECIPE_TOML = '''# {id} - prior-window channel breakout with ATR stop / target / time exit.
# TODO: describe the setup, entry, exit in 2-3 lines. Keep it honest and short.
id = "{id}"

[plugin]
type = "{id}"
direction = "both"            # long / short decided inside rust/mod.rs
exit_method = "{id}_exit"     # label only (shows in genome strings)
session_utc_start = [14, 30]  # only used by strategies that reset state per session

# Every gene: name, type (int|float), min, max. Order here = genome vector order
# = order in genome_fmt.toml template = names read in rust/mod.rs.
# type must match the Rust read helper: int -> param_i32, float -> param_f64.
[[params]]
name = "lookback"
type = "int"
min = 10
max = 60

[[params]]
name = "atr_period"
type = "int"
min = 7
max = 30

[[params]]
name = "stop_mult"
type = "float"
min = 1.0
max = 3.0

[[params]]
name = "target_mult"
type = "float"
min = 1.0
max = 5.0

[[params]]
name = "allow_short"
type = "int"
min = 0
max = 1

[[params]]
name = "exit_after_bars"
type = "int"
min = 5
max = 60
'''

GENOME_FMT_TOML = (
    'template = "{id}|{{direction}}|{{lookback}}|{{atr_period}}|{{stop_mult}}|{{target_mult}}'
    '|{{allow_short}}|{{exit_after_bars}}|{id}_exit|{{bet_mode}}|{{fixed_bet_size}}'
    '|{{price_bet_frac}}|{{session}}"\n'
)

MOD_RS = '''// {id}: prior-window channel breakout with ATR stop / target / time exit.
//
// Causality contract (see docs/NEW_STRATEGY.md):
//   * signals are decided on the CLOSE of bar i, the engine fills at bar i+1 OPEN
//   * every level used on bar i is built from bars < i (never includes bar i)
//   * one position at a time; exits are checked before entries
//
// Channel  = highest high / lowest low of bars [i-lookback .. i-1]
// Entry    = close[i] > channel_high (long) or close[i] < channel_low (short)
// Exit     = close crosses ATR stop or ATR target (frozen at entry), or time

use crate::data::MarketData;
use crate::recipe::{{PluginSpec, Recipe}};
use crate::runtime::{{BatchContext, SignalBuffers}};

pub fn build_signals(
    data: &MarketData,
    recipe: &Recipe,
    genome: &[f64],
    _spec: &PluginSpec,
    _ctx: &BatchContext,
    buffers: &mut SignalBuffers,
) {{
    let n = data.len();
    buffers.resize(n);
    buffers.entries.fill(false);
    buffers.exits.fill(false);
    buffers.directions.fill(1);

    // 1) params (names + int/float must match recipe.toml)
    let lookback = param_i32(recipe, genome, "lookback", 20).max(2) as usize;
    let atr_period = param_i32(recipe, genome, "atr_period", 14).max(2) as usize;
    let stop_mult = param_f64(recipe, genome, "stop_mult", 2.0).max(0.1);
    let target_mult = param_f64(recipe, genome, "target_mult", 3.0).max(0.1);
    let allow_short = param_i32(recipe, genome, "allow_short", 1) > 0;
    let exit_after_bars = param_i32(recipe, genome, "exit_after_bars", 20).max(1) as usize;

    // 2) indicators (all causal: value at i uses bars <= i; we then read i-1 where needed)
    let atr = wilder_atr(data, atr_period);
    let warm = lookback.max(atr_period) + 1;

    // 3) bar loop with single-position state
    let mut in_pos = false;
    let mut pos_dir: i8 = 1;
    let mut entry_i = 0usize;
    let mut stop_level = f64::NAN;
    let mut target_level = f64::NAN;

    for i in 0..n {{
        let c = data.close[i];
        if !c.is_finite() {{
            continue;
        }}

        if in_pos {{
            let held = i.saturating_sub(entry_i);
            let stop_hit = if pos_dir > 0 {{ c <= stop_level }} else {{ c >= stop_level }};
            let target_hit = if pos_dir > 0 {{ c >= target_level }} else {{ c <= target_level }};
            if stop_hit || target_hit || held >= exit_after_bars {{
                buffers.exits[i] = true;
                in_pos = false;
            }}
            continue; // never enter on the exit bar
        }}

        if i < warm {{
            continue;
        }}
        let a = atr[i];
        if !(a.is_finite() && a > 0.0) {{
            continue;
        }}

        // prior window [i-lookback .. i-1], excludes bar i
        let mut ch_high = f64::MIN;
        let mut ch_low = f64::MAX;
        for j in (i - lookback)..i {{
            let h = data.high[j];
            let l = data.low[j];
            if !(h.is_finite() && l.is_finite()) {{
                ch_high = f64::NAN;
                break;
            }}
            ch_high = ch_high.max(h);
            ch_low = ch_low.min(l);
        }}
        if !ch_high.is_finite() {{
            continue;
        }}

        if c > ch_high {{
            buffers.entries[i] = true;
            buffers.directions[i] = 1;
            in_pos = true;
            pos_dir = 1;
            entry_i = i;
            stop_level = c - stop_mult * a;
            target_level = c + target_mult * a;
        }} else if allow_short && c < ch_low {{
            buffers.entries[i] = true;
            buffers.directions[i] = -1;
            in_pos = true;
            pos_dir = -1;
            entry_i = i;
            stop_level = c + stop_mult * a;
            target_level = c - target_mult * a;
        }}
    }}
}}

fn wilder_atr(data: &MarketData, period: usize) -> Vec<f64> {{
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period < 2 || n < period {{
        return out;
    }}
    let mut tr = vec![f64::NAN; n];
    tr[0] = data.high[0] - data.low[0];
    for i in 1..n {{
        let hl = data.high[i] - data.low[i];
        let hc = (data.high[i] - data.close[i - 1]).abs();
        let lc = (data.low[i] - data.close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }}
    let mut sum = 0.0;
    for i in 0..period {{
        sum += tr[i];
    }}
    let mut atr = sum / period as f64;
    out[period - 1] = atr;
    for i in period..n {{
        atr = (atr * (period as f64 - 1.0) + tr[i]) / period as f64;
        out[i] = atr;
    }}
    out
}}

fn param_i32(recipe: &Recipe, genome: &[f64], name: &str, default: i32) -> i32 {{
    recipe
        .try_param_value(genome, name)
        .map(|v| v.round() as i32)
        .unwrap_or(default)
}}

fn param_f64(recipe: &Recipe, genome: &[f64], name: &str, default: f64) -> f64 {{
    recipe.try_param_value(genome, name).unwrap_or(default)
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn breakout_enters_long_once() {{
        let n = 40usize;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 * 3_600e9).collect();
        let mut close = vec![100.0; n];
        let mut high = vec![101.0; n];
        let mut low = vec![99.0; n];
        // bar 30 breaks the prior 20-bar channel
        close[30] = 105.0;
        high[30] = 106.0;
        low[30] = 100.0;
        let open = close.clone();
        let data = MarketData::new(ts, open, high, low, close, vec![1.0; n]).unwrap();
        let recipe = crate::recipe::get_recipe("{id}").unwrap();
        // lookback, atr_period, stop_mult, target_mult, allow_short, exit_after_bars
        let genome = vec![20.0, 14.0, 2.0, 3.0, 1.0, 20.0];
        let mut bufs = SignalBuffers::default();
        build_signals(&data, &recipe, &genome, recipe.plugin.as_ref().unwrap(), &BatchContext::default(), &mut bufs);
        assert!(bufs.entries[30]);
        assert_eq!(bufs.entries.iter().filter(|&&e| e).count(), 1);
    }}
}}
'''


def validate_strategy_id(sid: str) -> None:
    if not ID_RE.match(sid):
        raise ValueError(
            f"strategy id {sid!r} must be snake_case: lowercase letters, digits, underscores, "
            "starting with a letter (it becomes a Rust module name)"
        )


def scaffold_strategy(strategies_root: Path, sid: str, *, force: bool = False) -> list[Path]:
    validate_strategy_id(sid)
    folder = strategies_root / sid
    if folder.exists() and not force:
        raise FileExistsError(f"{folder} already exists (use --force to overwrite)")
    (folder / "rust").mkdir(parents=True, exist_ok=True)
    files = {
        folder / "recipe.toml": RECIPE_TOML.format(id=sid),
        folder / "genome_fmt.toml": GENOME_FMT_TOML.format(id=sid),
        folder / "rust" / "mod.rs": MOD_RS.format(id=sid),
    }
    for path, text in files.items():
        path.write_text(text, encoding="utf-8", newline="\n")
    return list(files)
