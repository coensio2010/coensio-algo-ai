"""Native core: strategy registry, determinism, forward parity, regression on the sample."""

import pytest

from coensio_algo_ai.fills import EngineConfig, evaluate_batch, register_dataset, run_backtest_genome
from coensio_algo_ai.genome_fmt import parse_genome_str
from coensio_algo_ai.strategy import list_strategies, load_strategy

IDS = list_strategies()


def _mid(spec):
    return [(p["start"] + p["end"]) / 2.0 for p in spec]


def _cfg():
    return EngineConfig(
        initial_capital=10_000.0,
        fixed_bet_size=2_000.0,
        commission_rate=0.00045,
        slippage_rate=0.0002,
        bet_mode="fixed",
        price_bet_frac=0.0,
    )


def test_native_registry_matches_folders(native):
    assert sorted(native.list_strategies()) == IDS


@pytest.mark.parametrize("sid", IDS)
def test_batch_is_deterministic(native, sample_df, sid):
    df = sample_df.iloc[-4000:]
    ds = register_dataset(df)
    spec = load_strategy(sid).PARAMS
    genomes = [_mid(spec), [p["start"] for p in spec], [p["end"] for p in spec]]
    a = evaluate_batch(ds, sid, genomes, config=_cfg())
    b = evaluate_batch(ds, sid, genomes, config=_cfg())
    assert a == b
    assert all(m.trades >= 0 for m in a)


@pytest.mark.parametrize("sid", IDS)
def test_forward_parity(native, sample_df, sid):
    """Causal bar-by-bar signals must equal batch signals (no look-ahead)."""
    from coensio_algo_ai.forward import run_one
    from coensio_algo_ai.genome_fmt import format_genome_str

    spec = load_strategy(sid).PARAMS
    params = {p["name"]: v for p, v in zip(spec, _mid(spec))}
    from coensio_algo_ai import genome as G

    params = G.clip_params(spec, params)
    gstr = format_genome_str(sid, spec, params, session="none", fixed_bet_size=2000.0, price_bet_frac=0.0)
    row = run_one(
        strategy=sid,
        datafile="BTC_1h.parquet",
        genome_str=gstr,
        sessions="none",
        session_mode="wall",
        max_bars=800,
        fixed_bet_size=2000.0,
    )
    assert row["ok"], row


def test_regression_donchian_atr_sample(native, sample_df):
    """Pin the sample-dataset result so fill / fee / metric changes are noticed."""
    genome = "donchian_atr|both|40|20|10|50|1.2|2.0|1|48|donchian_atr_exp|fixed|2000.0|0.1|none"
    parsed = parse_genome_str("donchian_atr", genome, load_strategy("donchian_atr").PARAMS)
    res = run_backtest_genome(sample_df, "donchian_atr", parsed.genome, config=_cfg())
    m = res.metrics
    assert m.trades == 333
    assert round(m.net_pnl, 2) == 1062.82
    assert round(m.max_dd, 2) == 1064.65
