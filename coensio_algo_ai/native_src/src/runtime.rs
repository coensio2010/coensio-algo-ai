use crate::data::MarketData;
use crate::indicators::FeatureBank;
use crate::recipe::{EntryRule, ExitRule, IndicatorDef, Recipe};
pub use crate::session::BatchContext;

#[derive(Clone, Default)]
pub struct SignalBuffers {
    pub entries: Vec<bool>,
    pub exits: Vec<bool>,
    pub directions: Vec<i8>,
}

/// Mirror the engine's single-position signal state.
///
/// An exit from an existing position wins over an entry on the same bar, so a
/// signal builder cannot create a position that the engine never opened.
pub fn update_signal_position(in_position: bool, entry: bool, exit: bool) -> bool {
    if in_position {
        !exit
    } else {
        entry
    }
}

impl SignalBuffers {
    pub fn resize(&mut self, n: usize) {
        self.entries.resize(n, false);
        self.exits.resize(n, false);
        self.directions.resize(n, 1);
    }
}

/// Build entry/exit/direction signals for one genome using a precomputed [`FeatureBank`].
pub fn build_signals(
    data: &MarketData,
    recipe: &Recipe,
    genome: &[f64],
    bank: &FeatureBank,
    ctx: &BatchContext,
    buffers: &mut SignalBuffers,
) {
    if recipe.is_plugin() {
        if let Some(spec) = &recipe.plugin {
            crate::strategies_registry::build_plugin_signals(
                &spec.plugin_type,
                data,
                recipe,
                genome,
                spec,
                ctx,
                buffers,
            )
            .expect("plugin signal build");
            return;
        }
    }

    let n = data.len();
    buffers.resize(n);
    buffers.entries.fill(false);
    buffers.exits.fill(false);
    buffers.directions.fill(1);

    let indicator = primary_indicator(recipe, genome, bank);
    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    apply_entry_rule(recipe, genome, indicator, &mut buffers.entries);
    apply_exit_rule(
        data,
        recipe,
        genome,
        indicator,
        bank,
        &buffers.entries,
        &mut buffers.exits,
    );

    if recipe.has_direction_param() {
        let dir_val = recipe.param_value(genome, "direction");
        let dir = if dir_val < 0.5 { 1i8 } else { -1i8 };
        buffers.directions.fill(dir);
    }
}

fn primary_indicator<'a>(recipe: &Recipe, genome: &[f64], bank: &'a FeatureBank) -> &'a [f64] {
    match recipe.indicator.as_ref().expect("indicator recipe") {
        IndicatorDef::Mfi { period_param } => {
            let period = clamp_usize(recipe.param_value(genome, period_param) as i64, 2, 200);
            bank.mfi_series(period).unwrap_or(&[])
        }
        IndicatorDef::UltimateOscillator {
            p1_param,
            p2_param,
            p3_param,
        } => {
            let p1 = clamp_usize(recipe.param_value(genome, p1_param) as i64, 2, 200);
            let p2 = clamp_usize(recipe.param_value(genome, p2_param) as i64, p1 + 1, 300);
            let p3 = clamp_usize(recipe.param_value(genome, p3_param) as i64, p2 + 1, 400);
            bank.uo_series(p1, p2, p3).unwrap_or(&[])
        }
    }
}

fn apply_entry_rule(recipe: &Recipe, genome: &[f64], indicator: &[f64], entries: &mut [bool]) {
    match recipe.entry.as_ref().expect("indicator recipe") {
        EntryRule::Below { threshold_param } => {
            let threshold = recipe.param_value(genome, threshold_param);
            for (i, &value) in indicator.iter().enumerate().take(entries.len()) {
                entries[i] = value.is_finite() && value < threshold;
            }
        }
        EntryRule::CrossBelow { level_param } => {
            let level = recipe.param_value(genome, level_param);
            for i in 1..entries.len() {
                let cur = indicator[i];
                let prev = indicator[i - 1];
                entries[i] = cur.is_finite() && prev.is_finite() && cur < level && prev >= level;
            }
        }
    }
}

fn apply_exit_rule(
    data: &MarketData,
    recipe: &Recipe,
    genome: &[f64],
    indicator: &[f64],
    bank: &FeatureBank,
    entries: &[bool],
    exits: &mut [bool],
) {
    match recipe.exit.as_ref().expect("indicator recipe") {
        ExitRule::CrossAbove { level_param } => {
            let level = recipe.param_value(genome, level_param);
            for i in 1..exits.len() {
                let cur = indicator[i];
                let prev = indicator[i - 1];
                exits[i] = cur.is_finite() && prev.is_finite() && cur > level && prev <= level;
            }
        }
        ExitRule::PrevHighOrTime { max_bars_param } => {
            let max_bars = recipe.param_value(genome, max_bars_param).max(0.0) as usize;
            build_prev_high_or_time_exits(data, entries, max_bars, bank.prev_high(), exits);
        }
    }
}

fn build_prev_high_or_time_exits(
    data: &MarketData,
    entries: &[bool],
    max_bars: usize,
    prev_high: &[f64],
    exits: &mut [bool],
) {
    let n = data.len();
    let mut in_pos = false;
    let mut entry_i = 0usize;
    for i in 0..n {
        let was_in_pos = in_pos;
        if in_pos {
            let held = i - entry_i;
            let prev = prev_high[i];
            if (prev.is_finite() && data.close[i] > prev) || (max_bars > 0 && held >= max_bars) {
                exits[i] = true;
            }
        }
        in_pos = update_signal_position(was_in_pos, entries[i], exits[i]);
        if !was_in_pos && in_pos {
            entry_i = i;
        }
    }
}

fn clamp_usize(v: i64, min: usize, max: usize) -> usize {
    v.max(min as i64).min(max as i64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_prevents_same_bar_simulated_reentry() {
        assert!(!update_signal_position(true, true, true));
    }

    #[test]
    fn flat_state_accepts_entry() {
        assert!(update_signal_position(false, true, false));
    }
}
