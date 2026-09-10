import pytest

from coensio_algo_ai.config import (
    _parse_fixed_bet_sizes,
    load_engine_cfg,
    load_sweep_cfg,
    load_validation_cfg,
    resolve_fixed_bet_size,
)


def test_default_cfg_loads():
    cfg = load_engine_cfg()
    assert cfg.initial_capital > 0
    assert cfg.population_size > 0
    assert "default" in cfg.fixed_bet_sizes
    assert cfg.datasets_dir.name == "datasets"
    assert cfg.strategies_dir.is_dir()
    load_sweep_cfg()
    load_validation_cfg()


def test_fixed_bet_size_scalar():
    sizes = _parse_fixed_bet_sizes(1500)
    assert resolve_fixed_bet_size(sizes, "ANY_1h.parquet") == 1500.0
    assert resolve_fixed_bet_size(sizes, None) == 1500.0


def test_fixed_bet_size_table_override_and_default():
    sizes = _parse_fixed_bet_sizes({"default": 2000.0, "SPY_1h": 10000.0})
    assert resolve_fixed_bet_size(sizes, "SPY_1h.parquet") == 10000.0
    assert resolve_fixed_bet_size(sizes, "datasets/SPY_1h.parquet") == 10000.0
    assert resolve_fixed_bet_size(sizes, "NEW_1d.parquet") == 2000.0
    assert resolve_fixed_bet_size(sizes, None) == 2000.0


def test_fixed_bet_size_table_requires_default():
    with pytest.raises(KeyError):
        _parse_fixed_bet_sizes({"SPY_1h": 10000.0})
    with pytest.raises(ValueError):
        _parse_fixed_bet_sizes({"default": 0})
