"""Every strategy folder must be internally consistent (recipe, template, Rust reads)."""

import numpy as np
import pytest

from coensio_algo_ai import genome as G
from coensio_algo_ai.check import static_problems
from coensio_algo_ai.genome_fmt import format_genome_str, parse_genome_str
from coensio_algo_ai.strategy import list_strategies, load_recipe_params, load_strategy

IDS = list_strategies()


def test_have_strategies():
    assert IDS, "no strategy folders found"


@pytest.mark.parametrize("sid", IDS)
def test_recipe_params_typed(sid):
    params = load_recipe_params(sid)
    assert params
    for p in params:
        assert p["type"] in ("int", "float")
        assert p["start"] <= p["end"]
        if p["type"] == "int":
            assert isinstance(p["start"], int) and isinstance(p["end"], int)


@pytest.mark.parametrize("sid", IDS)
def test_static_check_passes(sid):
    assert static_problems(sid) == []


@pytest.mark.parametrize("sid", IDS)
def test_genome_string_roundtrip(sid):
    spec = load_strategy(sid).PARAMS
    rng = np.random.default_rng(123)
    params = G.random_params(spec, rng)
    text = format_genome_str(
        sid, spec, params, session="london", bet_mode="fixed", fixed_bet_size=2500.0, price_bet_frac=0.1
    )
    assert text.startswith(f"{sid}|")
    parsed = parse_genome_str(sid, text, spec)
    assert parsed.session == "london"
    assert parsed.fixed_bet_size == 2500.0
    for p in spec:
        v = parsed.params[p["name"]]
        if p["type"] == "int":
            assert v == params[p["name"]]
        else:
            assert abs(float(v) - float(params[p["name"]])) < 1e-3


@pytest.mark.parametrize("sid", IDS)
def test_short_genome_roundtrip(sid):
    spec = load_strategy(sid).PARAMS
    rng = np.random.default_rng(7)
    params = G.random_params(spec, rng)
    short = G.encode(spec, params)
    back = G.decode(spec, short)
    for p in spec:
        assert abs(float(back[p["name"]]) - float(params[p["name"]])) < 1e-4


def test_ga_operators_respect_types_and_bounds():
    spec = load_strategy(IDS[0]).PARAMS
    rng = np.random.default_rng(0)
    a = G.random_params(spec, rng)
    b = G.random_params(spec, rng)
    for child in (*G.crossover(spec, a, b, rng), G.mutate_params(spec, a, rng)):
        for p in spec:
            v = child[p["name"]]
            assert p["start"] <= v <= p["end"]
            if p["type"] == "int":
                assert isinstance(v, int)
