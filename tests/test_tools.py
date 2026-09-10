"""Scaffold, CSV import, data loading."""

import tomllib

import pandas as pd
import pytest

from coensio_algo_ai.data import import_csv, load_ohlcv_file
from coensio_algo_ai.genome_fmt import TEMPLATE_FIELD_RE
from coensio_algo_ai.scaffold import scaffold_strategy, validate_strategy_id


def test_scaffold_writes_consistent_files(tmp_path):
    files = scaffold_strategy(tmp_path, "my_test_strat")
    assert {f.name for f in files} == {"recipe.toml", "genome_fmt.toml", "mod.rs"}
    recipe = tomllib.load(open(tmp_path / "my_test_strat" / "recipe.toml", "rb"))
    assert recipe["id"] == "my_test_strat"
    assert recipe["plugin"]["type"] == "my_test_strat"
    names = [p["name"] for p in recipe["params"]]
    assert all(p["type"] in ("int", "float") for p in recipe["params"])
    fmt = tomllib.load(open(tmp_path / "my_test_strat" / "genome_fmt.toml", "rb"))
    fields = TEMPLATE_FIELD_RE.findall(fmt["template"])
    assert fields[0] == "direction"
    assert fields[1 : 1 + len(names)] == names
    assert fields[-4:] == ["bet_mode", "fixed_bet_size", "price_bet_frac", "session"]
    rs = (tmp_path / "my_test_strat" / "rust" / "mod.rs").read_text(encoding="utf-8")
    assert "pub fn build_signals(" in rs
    for n in names:
        assert f'"{n}"' in rs
    with pytest.raises(FileExistsError):
        scaffold_strategy(tmp_path, "my_test_strat")


@pytest.mark.parametrize("bad", ["My_Strat", "1abc", "a-b", "a b", ""])
def test_scaffold_rejects_bad_ids(bad):
    with pytest.raises(ValueError):
        validate_strategy_id(bad)


def test_import_csv_roundtrip(tmp_path):
    idx = pd.date_range("2024-01-02 09:30", periods=50, freq="h")
    df = pd.DataFrame(
        {
            "Date": idx.strftime("%Y-%m-%d %H:%M:%S"),
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 1000,
        }
    )
    df.loc[3, "Date"] = df.loc[2, "Date"]  # duplicate timestamp must be dropped
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    out = import_csv(csv, "TEST_1h", tz="America/New_York", out_dir=tmp_path)
    loaded = load_ohlcv_file(out)
    assert list(loaded.columns[:5]) == ["open", "high", "low", "close", "volume"]
    assert len(loaded) == 49
    assert loaded.index.is_monotonic_increasing
    assert loaded.index.tz is not None


def test_import_csv_epoch_ms(tmp_path):
    ts = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": (ts.view("int64") // 1_000_000),
            "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10,
        }
    )
    csv = tmp_path / "e.csv"
    df.to_csv(csv, index=False)
    out = import_csv(csv, "BTC_1h", out_dir=tmp_path)
    loaded = load_ohlcv_file(out)
    assert len(loaded) == 10
    assert str(loaded.index[0]) == "2024-01-01 00:00:00+00:00"
