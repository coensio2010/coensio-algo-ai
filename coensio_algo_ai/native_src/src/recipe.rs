use std::collections::HashMap;
use std::fmt;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ParamRange {
    pub name: String,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IndicatorDef {
    Mfi {
        period_param: String,
    },
    UltimateOscillator {
        p1_param: String,
        p2_param: String,
        p3_param: String,
    },
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EntryRule {
    Below { threshold_param: String },
    CrossBelow { level_param: String },
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExitRule {
    PrevHighOrTime { max_bars_param: String },
    CrossAbove { level_param: String },
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct SideRules {
    pub long: EntryRule,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ExitSideRules {
    pub long: ExitRule,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct PluginSpec {
    #[serde(rename = "type")]
    pub plugin_type: String,
    pub direction: String,
    pub exit_method: String,
    #[serde(default)]
    pub session_utc_start: Vec<i32>,
    #[serde(default)]
    pub session_utc_end: Vec<i32>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct RecipeFile {
    id: String,
    params: Vec<ParamRange>,
    plugin: Option<PluginSpec>,
    indicator: Option<IndicatorDef>,
    entry: Option<SideRules>,
    exit: Option<ExitSideRules>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Recipe {
    pub id: String,
    pub params: Vec<ParamRange>,
    pub plugin: Option<PluginSpec>,
    pub indicator: Option<IndicatorDef>,
    pub entry: Option<EntryRule>,
    pub exit: Option<ExitRule>,
    param_index: HashMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecipeError {
    Parse(String),
    UnknownStrategy(String),
    GenomeLength { expected: usize, actual: usize },
    InvalidRecipe(String),
}

impl fmt::Display for RecipeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(msg) => write!(f, "recipe parse error: {msg}"),
            Self::UnknownStrategy(id) => write!(f, "unknown strategy: {id}"),
            Self::GenomeLength { expected, actual } => {
                write!(f, "genome length {actual} != expected {expected}")
            }
            Self::InvalidRecipe(msg) => write!(f, "invalid recipe: {msg}"),
        }
    }
}

impl std::error::Error for RecipeError {}

impl Recipe {
    pub fn parse(toml_src: &str) -> Result<Self, RecipeError> {
        let file: RecipeFile =
            toml::from_str(toml_src).map_err(|e| RecipeError::Parse(e.to_string()))?;

        if file.plugin.is_none()
            && (file.indicator.is_none() || file.entry.is_none() || file.exit.is_none())
        {
            return Err(RecipeError::InvalidRecipe(
                "indicator recipes require indicator, entry, and exit sections".into(),
            ));
        }

        let mut param_index = HashMap::new();
        for (i, param) in file.params.iter().enumerate() {
            param_index.insert(param.name.clone(), i);
        }
        Ok(Self {
            id: file.id,
            params: file.params,
            plugin: file.plugin,
            indicator: file.indicator,
            entry: file.entry.map(|e| e.long),
            exit: file.exit.map(|e| e.long),
            param_index,
        })
    }

    pub fn is_plugin(&self) -> bool {
        self.plugin.is_some()
    }

    pub fn param_index(&self, name: &str) -> usize {
        *self
            .param_index
            .get(name)
            .unwrap_or_else(|| panic!("unknown param {name}"))
    }

    pub fn param_value(&self, genome: &[f64], name: &str) -> f64 {
        genome[self.param_index(name)]
    }

    pub fn try_param_value(&self, genome: &[f64], name: &str) -> Option<f64> {
        self.param_index.get(name).map(|&i| genome[i])
    }

    pub fn validate_genome(&self, genome: &[f64]) -> Result<(), RecipeError> {
        // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
        if genome.len() != self.params.len() {
            return Err(RecipeError::GenomeLength {
                expected: self.params.len(),
                actual: genome.len(),
            });
        }
        Ok(())
    }

    pub fn has_direction_param(&self) -> bool {
        self.param_index.contains_key("direction")
    }
}

fn all_recipes() -> Vec<Recipe> {
    crate::strategies_registry::all_recipe_tomls()
        .iter()
        .map(|(_, toml)| Recipe::parse(toml).expect("builtin recipe parse"))
        .collect()
}

pub fn get_recipe(strategy_id: &str) -> Result<Recipe, RecipeError> {
    all_recipes()
        .into_iter()
        .find(|r| r.id == strategy_id)
        .ok_or_else(|| RecipeError::UnknownStrategy(strategy_id.to_string()))
}

pub fn list_strategy_ids() -> Vec<String> {
    crate::strategies_registry::all_recipe_tomls()
        .iter()
        .map(|(id, _)| id.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every strategy folder under strategies/ must parse, be a plugin, and have
    /// a non-empty, duplicate-free param list whose id matches its folder name.
    #[test]
    fn all_installed_recipes_are_valid_plugins() {
        let ids = list_strategy_ids();
        assert!(!ids.is_empty(), "no strategies found under strategies/");
        for id in &ids {
            let recipe = get_recipe(id).unwrap_or_else(|e| panic!("{id}: {e}"));
            assert_eq!(&recipe.id, id, "recipe id must equal folder name");
            assert!(recipe.is_plugin(), "{id}: [plugin] section required");
            assert!(!recipe.params.is_empty(), "{id}: [[params]] is empty");
            let plugin = recipe.plugin.as_ref().unwrap();
            assert_eq!(&plugin.plugin_type, id, "{id}: [plugin].type must equal id");
            for (i, p) in recipe.params.iter().enumerate() {
                assert!(p.min <= p.max, "{id}: param {} min > max", p.name);
                assert_eq!(recipe.param_index(&p.name), i, "{id}: duplicate param {}", p.name);
            }
        }
    }

    #[test]
    fn list_strategies_is_sorted_and_unique() {
        let ids = list_strategy_ids();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(ids, sorted);
    }
}
