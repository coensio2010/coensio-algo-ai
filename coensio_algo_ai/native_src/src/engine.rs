use crate::data::MarketData;
use crate::metrics::{self, CompactMetrics, MetricsInput};
use rayon::prelude::*;


/// Bet sizing mode (fixed notional or price fraction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BetMode {
    Fixed,
    PriceFrac,
}

impl BetMode {
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "price_frac" | "pricefrac" => Self::PriceFrac,
            _ => Self::Fixed,
        }
    }
}

/// Engine execution and cost configuration.
#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    pub initial_capital: f64,
    pub bet_mode: BetMode,
    pub fixed_bet_size: f64,
    pub price_bet_frac: f64,
    pub commission_rate: f64,
    pub slippage_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    InvalidInitialCapital,
    InvalidFixedBetSize,
    InvalidPriceBetFrac,
    NegativeCommission,
    InvalidSlippage,
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidDirection {
        index: usize,
        value: i8,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInitialCapital => write!(f, "initial_capital must be finite and > 0"),
            Self::InvalidFixedBetSize => write!(f, "fixed_bet_size must be finite and > 0"),
            Self::InvalidPriceBetFrac => {
                write!(
                    f,
                    "price_bet_frac must be finite and > 0 when bet_mode=price_frac"
                )
            }
            Self::NegativeCommission => write!(f, "commission_rate must be >= 0"),
            Self::InvalidSlippage => write!(f, "slippage_rate must be finite, >= 0, and < 1"),
            Self::LengthMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "{field} length {actual} does not match data length {expected}"
            ),
            Self::InvalidDirection { index, value } => {
                write!(f, "directions[{index}] must be -1 or +1, got {value}")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

impl EngineConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !self.initial_capital.is_finite() || self.initial_capital <= 0.0 {
            return Err(ConfigError::InvalidInitialCapital);
        }
        match self.bet_mode {
            BetMode::Fixed => {
                if !self.fixed_bet_size.is_finite() || self.fixed_bet_size <= 0.0 {
                    return Err(ConfigError::InvalidFixedBetSize);
                }
            }
            BetMode::PriceFrac => {
                if !self.price_bet_frac.is_finite() || self.price_bet_frac <= 0.0 {
                    return Err(ConfigError::InvalidPriceBetFrac);
                }
            }
        }
        if !self.commission_rate.is_finite() || self.commission_rate < 0.0 {
            return Err(ConfigError::NegativeCommission);
        }
        if !self.slippage_rate.is_finite() || self.slippage_rate < 0.0 || self.slippage_rate >= 1.0
        {
            return Err(ConfigError::InvalidSlippage);
        }
        Ok(())
    }

    /// USD notional at entry: fixed $ or fraction of asset price at fill.
    pub fn entry_notional(&self, fill_price: f64) -> f64 {
        match self.bet_mode {
            BetMode::Fixed => self.fixed_bet_size,
            BetMode::PriceFrac => self.price_bet_frac * fill_price,
        }
    }
}

/// One completed round-trip.
#[derive(Debug, Clone, PartialEq)]
pub struct Trade {
    pub entry_bar: usize,
    pub exit_bar: usize,
    pub raw_entry_price: f64,
    pub raw_exit_price: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub entry_notional: f64,
    pub exit_consideration: f64,
    pub entry_commission: f64,
    pub exit_commission: f64,
    pub slippage_cost: f64,
    pub direction: i8,
    pub gross_pnl: f64,
    pub net_pnl: f64,
    pub exit_reason: &'static str,
}

/// Full engine output (metrics plus optional detail vectors).
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub metrics: CompactMetrics,
    pub equity_curve: Vec<f64>,
    pub trades: Vec<Trade>,
}

/// Pending order scheduled by a signal on the prior bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingFill {
    None,
    Entry { direction: i8 },
    Exit,
}

/// Run the Sprint 1 single-position bar engine.
///
/// Execution rules (no same-bar lookahead):
/// - Entry signal at bar `i` fills at bar `i + 1` open.
/// - Exit signal at bar `i` fills at bar `i + 1` open.
/// - Slippage moves each fill price adversely for the position direction.
/// - Requested notional is capped to positive flat equity; leverage is not modeled.
/// - Any open position is force-closed at the final bar close.
/// - Entry signals are ignored while already in a position.
pub fn run_backtest(
    data: &MarketData,
    entries: &[bool],
    exits: &[bool],
    directions: &[i8],
    config: &EngineConfig,
) -> Result<BacktestResult, ConfigError> {
    config.validate()?;

    let n = data.len();
    validate_signal_lengths(n, entries, exits, directions)?;
    validate_directions(directions)?;

    let mut cash = config.initial_capital;
    let mut equity_curve = Vec::with_capacity(n);
    let mut trades = Vec::new();

    let mut in_position = false;
    let mut position_direction: i8 = 1;
    let mut entry_bar: usize = 0;
    let mut raw_entry_price = 0.0;
    let mut entry_price = 0.0;
    let mut entry_commission = 0.0;
    let mut notional = 0.0;
    let mut quantity = 0.0;
    let mut pending = PendingFill::None;

    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    for i in 0..n {
        // Fills from prior-bar signals execute at this bar's open.
        if i > 0 {
            match pending {
                PendingFill::Entry { direction } => {
                    let raw_fill_price = data.open[i];
                    let fill_price =
                        adverse_fill_price(raw_fill_price, direction, true, config.slippage_rate);
                    let requested_notional = config.entry_notional(fill_price);
                    notional = requested_notional.min(cash.max(0.0));
                    if notional > 0.0 {
                        entry_bar = i;
                        raw_entry_price = raw_fill_price;
                        entry_price = fill_price;
                        position_direction = direction;
                        quantity = notional / fill_price;
                        entry_commission = notional * config.commission_rate;
                        cash -= entry_commission;
                        in_position = true;
                    }
                    pending = PendingFill::None;
                }
                PendingFill::Exit => {
                    close_position(
                        i,
                        data.open[i],
                        "signal",
                        config,
                        entry_commission,
                        &mut cash,
                        &mut in_position,
                        &mut notional,
                        &mut quantity,
                        entry_bar,
                        raw_entry_price,
                        entry_price,
                        position_direction,
                        &mut trades,
                    );
                    pending = PendingFill::None;
                }
                PendingFill::None => {}
            }
        }

        let mark = if in_position {
            let unrealized = position_direction as f64 * (data.close[i] - entry_price) * quantity;
            cash + unrealized
        } else {
            cash
        };
        equity_curve.push(mark);

        // Signals observed at bar close; fills occur on the next bar if one exists.
        if in_position {
            if exits[i] && i + 1 < n {
                pending = PendingFill::Exit;
            }
        } else if pending == PendingFill::None && entries[i] && i + 1 < n {
            pending = PendingFill::Entry {
                direction: directions[i],
            };
        }
    }

    if in_position {
        close_position(
            n - 1,
            data.close[n - 1],
            "force_close",
            config,
            entry_commission,
            &mut cash,
            &mut in_position,
            &mut notional,
            &mut quantity,
            entry_bar,
            raw_entry_price,
            entry_price,
            position_direction,
            &mut trades,
        );
        if let Some(last) = equity_curve.last_mut() {
            *last = cash;
        }
    }

    let metrics = metrics::compute_compact(MetricsInput {
        initial_capital: config.initial_capital,
        equity_curve: &equity_curve,
        timestamps: &data.timestamp,
        trades: &trades,
    });

    Ok(BacktestResult {
        metrics,
        equity_curve,
        trades,
    })
}

/// Parallel compact backtests over many signal sets (Rayon). Shared OHLCV.
pub fn run_backtest_batch(
    data: &MarketData,
    entries: &[Vec<bool>],
    exits: &[Vec<bool>],
    directions: &[Vec<i8>],
    config: &EngineConfig,
) -> Result<Vec<CompactMetrics>, ConfigError> {
    config.validate()?;
    if entries.len() != exits.len() || entries.len() != directions.len() {
        return Err(ConfigError::LengthMismatch {
            field: "batch_signals",
            expected: entries.len(),
            actual: exits.len().min(directions.len()),
        });
    }
    let n = entries.len();
    let mut out: Vec<Result<CompactMetrics, ConfigError>> = (0..n)
        .into_par_iter()
        .map(|i| {
            run_backtest(data, &entries[i], &exits[i], &directions[i], config)
                .map(|r| r.metrics)
        })
        .collect();
    let mut metrics = Vec::with_capacity(n);
    for row in out.drain(..) {
        metrics.push(row?);
    }
    Ok(metrics)
}

#[allow(clippy::too_many_arguments)]
fn close_position(
    exit_bar: usize,
    raw_exit_price: f64,
    reason: &'static str,
    config: &EngineConfig,
    entry_commission: f64,
    cash: &mut f64,
    in_position: &mut bool,
    notional: &mut f64,
    quantity: &mut f64,
    entry_bar: usize,
    raw_entry_price: f64,
    entry_price: f64,
    position_direction: i8,
    trades: &mut Vec<Trade>,
) {
    let exit_price = adverse_fill_price(
        raw_exit_price,
        position_direction,
        false,
        config.slippage_rate,
    );
    let exit_consideration = *quantity * exit_price;
    let gross = position_direction as f64 * (exit_price - entry_price) * *quantity;
    let exit_commission = exit_consideration * config.commission_rate;
    let slippage_cost =
        ((entry_price - raw_entry_price).abs() + (exit_price - raw_exit_price).abs()) * *quantity;
    let trade_net = gross - entry_commission - exit_commission;
    *cash += gross - exit_commission;
    trades.push(Trade {
        entry_bar,
        exit_bar,
        raw_entry_price,
        raw_exit_price,
        entry_price,
        exit_price,
        entry_notional: *notional,
        exit_consideration,
        entry_commission,
        exit_commission,
        slippage_cost,
        direction: position_direction,
        gross_pnl: gross,
        net_pnl: trade_net,
        exit_reason: reason,
    });
    *in_position = false;
    *notional = 0.0;
    *quantity = 0.0;
}

fn adverse_fill_price(raw_price: f64, direction: i8, is_entry: bool, slippage_rate: f64) -> f64 {
    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    let adverse_sign = if (direction > 0) == is_entry {
        1.0
    } else {
        -1.0
    };
    raw_price * (1.0 + adverse_sign * slippage_rate)
}

fn validate_signal_lengths(
    n: usize,
    entries: &[bool],
    exits: &[bool],
    directions: &[i8],
) -> Result<(), ConfigError> {
    let fields: [(&str, usize); 3] = [
        ("entries", entries.len()),
        ("exits", exits.len()),
        ("directions", directions.len()),
    ];
    for (field, len) in fields {
        if len != n {
            return Err(ConfigError::LengthMismatch {
                field,
                expected: n,
                actual: len,
            });
        }
    }
    Ok(())
}

fn validate_directions(directions: &[i8]) -> Result<(), ConfigError> {
    for (i, &dir) in directions.iter().enumerate() {
        if dir != -1 && dir != 1 {
            return Err(ConfigError::InvalidDirection {
                index: i,
                value: dir,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::MarketData;

    fn sample_data(opens: &[f64], highs: &[f64], lows: &[f64], closes: &[f64]) -> MarketData {
        let n = opens.len();
        MarketData::new(
            (0..n).map(|i| i as f64 * 86_400.0).collect(),
            opens.to_vec(),
            highs.to_vec(),
            lows.to_vec(),
            closes.to_vec(),
            vec![1.0; n],
        )
        .unwrap()
    }

    fn base_config() -> EngineConfig {
        EngineConfig {
            initial_capital: 10_000.0,
            bet_mode: BetMode::Fixed,
            fixed_bet_size: 1_000.0,
            price_bet_frac: 0.0,
            commission_rate: 0.0,
            slippage_rate: 0.0,
        }
    }

    fn price_frac_config(frac: f64) -> EngineConfig {
        EngineConfig {
            bet_mode: BetMode::PriceFrac,
            fixed_bet_size: 0.0,
            price_bet_frac: frac,
            ..base_config()
        }
    }

    #[test]
    fn long_profit() {
        let data = sample_data(
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
        );
        let entries = vec![true, false, false];
        let exits = vec![false, true, false];
        let directions = vec![1i8; 3];
        let result = run_backtest(&data, &entries, &exits, &directions, &base_config()).unwrap();
        assert_eq!(result.trades.len(), 1);
        assert!((result.trades[0].net_pnl - 100.0).abs() < 1e-9);
        assert!((result.metrics.net_pnl - 100.0).abs() < 1e-9);
    }

    #[test]
    fn short_profit() {
        let data = sample_data(
            &[100.0, 100.0, 90.0],
            &[100.0, 100.0, 100.0],
            &[100.0, 90.0, 90.0],
            &[100.0, 95.0, 90.0],
        );
        let entries = vec![true, false, false];
        let exits = vec![false, true, false];
        let directions = vec![-1i8; 3];
        let result = run_backtest(&data, &entries, &exits, &directions, &base_config()).unwrap();
        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].direction, -1);
        assert!((result.trades[0].net_pnl - 100.0).abs() < 1e-9);
    }

    #[test]
    fn next_bar_fill_causality() {
        let data = sample_data(
            &[50.0, 100.0, 120.0],
            &[50.0, 100.0, 120.0],
            &[50.0, 100.0, 120.0],
            &[50.0, 100.0, 120.0],
        );
        let entries = vec![true, false, false];
        let exits = vec![false, false, true];
        let directions = vec![1i8; 3];
        let result = run_backtest(&data, &entries, &exits, &directions, &base_config()).unwrap();
        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].entry_price, 100.0);
        assert_eq!(result.trades[0].exit_price, 120.0);
        assert_eq!(result.trades[0].entry_bar, 1);
        assert_eq!(result.trades[0].exit_bar, 2);
    }

    #[test]
    fn fees_and_slippage_reduce_pnl() {
        let data = sample_data(
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
        );
        let entries = vec![true, false, false];
        let exits = vec![false, true, false];
        let directions = vec![1i8; 3];
        let no_cost = run_backtest(&data, &entries, &exits, &directions, &base_config()).unwrap();
        let with_cost = run_backtest(
            &data,
            &entries,
            &exits,
            &directions,
            &EngineConfig {
                commission_rate: 0.001,
                slippage_rate: 0.001,
                ..base_config()
            },
        )
        .unwrap();
        assert!(with_cost.metrics.net_pnl < no_cost.metrics.net_pnl);
        let trade = &with_cost.trades[0];
        assert!((trade.entry_price - 100.1).abs() < 1e-9);
        assert!((trade.exit_price - 109.89).abs() < 1e-9);
        assert!((trade.entry_commission - 1.0).abs() < 1e-9);
        assert!((trade.exit_commission - trade.exit_consideration * 0.001).abs() < 1e-9);
        assert!(trade.slippage_cost > 0.0);
        assert!((trade.net_pnl - with_cost.metrics.net_pnl).abs() < 0.01);
    }

    #[test]
    fn final_force_close() {
        let data = sample_data(
            &[100.0, 105.0],
            &[100.0, 105.0],
            &[100.0, 105.0],
            &[100.0, 105.0],
        );
        let entries = vec![true, false];
        let exits = vec![false, false];
        let directions = vec![1i8; 2];
        let result = run_backtest(&data, &entries, &exits, &directions, &base_config()).unwrap();
        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].exit_reason, "force_close");
        assert_eq!(result.trades[0].exit_price, 105.0);
    }

    #[test]
    fn force_close_uses_adverse_slippage_and_actual_commission() {
        let data = sample_data(
            &[100.0, 100.0],
            &[100.0, 110.0],
            &[100.0, 99.0],
            &[100.0, 110.0],
        );
        let config = EngineConfig {
            commission_rate: 0.001,
            slippage_rate: 0.01,
            ..base_config()
        };
        let result =
            run_backtest(&data, &[true, false], &[false, false], &[1, 1], &config).unwrap();
        let trade = &result.trades[0];
        assert_eq!(trade.exit_reason, "force_close");
        assert!((trade.entry_price - 101.0).abs() < 1e-9);
        assert!((trade.exit_price - 108.9).abs() < 1e-9);
        assert!(
            (trade.exit_commission - trade.exit_consideration * config.commission_rate).abs()
                < 1e-9
        );
    }

    #[test]
    fn requested_notional_is_capped_to_positive_flat_equity() {
        let data = sample_data(
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
        );
        let config = EngineConfig {
            fixed_bet_size: 25_000.0,
            ..base_config()
        };
        let result = run_backtest(
            &data,
            &[true, false, false],
            &[false, true, false],
            &[1; 3],
            &config,
        )
        .unwrap();
        assert!((result.trades[0].entry_notional - 10_000.0).abs() < 1e-9);
        assert!((result.trades[0].gross_pnl - 1_000.0).abs() < 1e-9);
    }

    #[test]
    fn slippage_is_adverse_for_every_fill_side() {
        assert!((adverse_fill_price(100.0, 1, true, 0.01) - 101.0).abs() < 1e-9);
        assert!((adverse_fill_price(100.0, 1, false, 0.01) - 99.0).abs() < 1e-9);
        assert!((adverse_fill_price(100.0, -1, true, 0.01) - 99.0).abs() < 1e-9);
        assert!((adverse_fill_price(100.0, -1, false, 0.01) - 101.0).abs() < 1e-9);
    }

    #[test]
    fn penultimate_exit_signal_fills_at_last_open() {
        let data = sample_data(
            &[100.0, 100.0, 105.0],
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 100.0],
            &[100.0, 100.0, 110.0],
        );
        let result = run_backtest(
            &data,
            &[true, false, false],
            &[false, true, false],
            &[1; 3],
            &base_config(),
        )
        .unwrap();
        assert_eq!(result.trades[0].exit_bar, 2);
        assert_eq!(result.trades[0].exit_price, 105.0);
        assert_eq!(result.trades[0].exit_reason, "signal");
    }

    #[test]
    fn final_bar_entry_signal_is_ignored() {
        let data = sample_data(
            &[100.0, 100.0],
            &[100.0, 100.0],
            &[100.0, 100.0],
            &[100.0, 100.0],
        );
        let result = run_backtest(
            &data,
            &[false, true],
            &[false, false],
            &[1; 2],
            &base_config(),
        )
        .unwrap();
        assert!(result.trades.is_empty());
    }

    #[test]
    fn invalid_input_rejected() {
        let data = sample_data(&[100.0], &[100.0], &[100.0], &[100.0]);
        let err = run_backtest(&data, &[], &[], &[], &base_config()).unwrap_err();
        assert!(matches!(err, ConfigError::LengthMismatch { .. }));

        let bad_dir = run_backtest(&data, &[true], &[false], &[2], &base_config()).unwrap_err();
        assert!(matches!(bad_dir, ConfigError::InvalidDirection { .. }));

        let bad_cfg = EngineConfig {
            bet_mode: BetMode::Fixed,
            fixed_bet_size: 0.0,
            ..base_config()
        };
        assert!(matches!(
            bad_cfg.validate().unwrap_err(),
            ConfigError::InvalidFixedBetSize
        ));
    }

    #[test]
    fn invalid_direction_on_non_entry_bar_rejected() {
        let data = sample_data(
            &[100.0, 100.0],
            &[100.0, 100.0],
            &[100.0, 100.0],
            &[100.0, 100.0],
        );
        let err = run_backtest(
            &data,
            &[true, false],
            &[false, false],
            &[1, 0],
            &base_config(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ConfigError::InvalidDirection { index: 1, value: 0 }
        ));
    }

    #[test]
    fn deterministic_repeatability() {
        let data = sample_data(
            &[100.0, 100.0, 110.0, 105.0],
            &[100.0, 100.0, 110.0, 105.0],
            &[100.0, 100.0, 110.0, 105.0],
            &[100.0, 100.0, 110.0, 105.0],
        );
        let entries = vec![true, false, false, false];
        let exits = vec![false, true, false, false];
        let directions = vec![1i8; 4];
        let a = run_backtest(&data, &entries, &exits, &directions, &base_config()).unwrap();
        let b = run_backtest(&data, &entries, &exits, &directions, &base_config()).unwrap();
        assert_eq!(a.metrics, b.metrics);
        assert_eq!(a.trades, b.trades);
        assert_eq!(a.equity_curve, b.equity_curve);
    }

    #[test]
    fn metrics_sanity() {
        let data = sample_data(
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
        );
        let entries = vec![true, false, false];
        let exits = vec![false, true, false];
        let directions = vec![1i8; 3];
        let result = run_backtest(&data, &entries, &exits, &directions, &base_config()).unwrap();
        let m = &result.metrics;
        assert_eq!(m.trades, 1);
        assert!((m.net_avg - m.net_pnl).abs() < 1e-9);
        assert!(m.max_dd >= 0.0);
        assert!(m.rdd > 0.0);
    }

    #[test]
    fn price_frac_scales_with_asset_price() {
        let data = sample_data(
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
            &[100.0, 100.0, 110.0],
        );
        let entries = vec![true, false, false];
        let exits = vec![false, true, false];
        let directions = vec![1i8; 3];
        let fixed = run_backtest(
            &data,
            &entries,
            &exits,
            &directions,
            &EngineConfig {
                fixed_bet_size: 10.0,
                ..base_config()
            },
        )
        .unwrap();
        let scaled = run_backtest(
            &data,
            &entries,
            &exits,
            &directions,
            &price_frac_config(0.10),
        )
        .unwrap();
        // 10% of $100 entry => $10 notional, same as fixed $10.
        assert!((fixed.metrics.net_pnl - scaled.metrics.net_pnl).abs() < 1e-6);

        let data2 = sample_data(
            &[200.0, 200.0, 220.0],
            &[200.0, 200.0, 220.0],
            &[200.0, 200.0, 220.0],
            &[200.0, 200.0, 220.0],
        );
        let scaled2 = run_backtest(
            &data2,
            &entries,
            &exits,
            &directions,
            &price_frac_config(0.10),
        )
        .unwrap();
        // Double price => double notional => double dollar PnL for same % move.
        assert!((scaled2.metrics.net_pnl - 2.0 * scaled.metrics.net_pnl).abs() < 1e-6);
    }

    #[test]
    fn ignore_entry_while_in_position() {
        let data = sample_data(
            &[100.0, 100.0, 100.0, 110.0],
            &[100.0, 100.0, 100.0, 110.0],
            &[100.0, 100.0, 100.0, 110.0],
            &[100.0, 100.0, 100.0, 110.0],
        );
        let entries = vec![true, true, false, false];
        let exits = vec![false, false, true, false];
        let directions = vec![1i8; 4];
        let result = run_backtest(&data, &entries, &exits, &directions, &base_config()).unwrap();
        assert_eq!(result.trades.len(), 1);
    }
}
