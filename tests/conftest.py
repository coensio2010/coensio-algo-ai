import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE = ROOT / "datasets" / "BTC_1h.parquet"


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def sample_df():
    from coensio_algo_ai.data import load_ohlcv_file

    if not SAMPLE.is_file():
        pytest.skip("bundled datasets/BTC_1h.parquet missing")
    return load_ohlcv_file(SAMPLE)


@pytest.fixture(scope="session")
def native():
    from coensio_algo_ai.fills import native_binary_path

    if native_binary_path() is None:
        pytest.skip("native core not built (python coensio_algo_ai/build_native.py)")
    from coensio_algo_ai.fills import _import_native

    return _import_native()
