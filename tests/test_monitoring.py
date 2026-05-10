import pytest
import pandas as pd
import numpy as np
from src.monitoring.monitoring import ModelMonitor, _clean, NUMERICAL_COLS, CATEGORICAL_COLS


def _make_df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    num = {col: rng.random(n) * 10 for col in NUMERICAL_COLS}
    cat = {col: rng.choice(["a", "b"], n) for col in CATEGORICAL_COLS}
    return pd.DataFrame({**num, **cat})


@pytest.fixture
def monitor():
    m = ModelMonitor.__new__(ModelMonitor)
    m.reference_data = None
    m.current_data = None
    return m


@pytest.fixture
def monitor_with_data():
    m = ModelMonitor.__new__(ModelMonitor)
    m.reference_data = _make_df(seed=0)
    m.current_data = _make_df(seed=0)
    return m


class TestClean:
    def test_replaces_inf_with_nan_then_zero(self):
        df = pd.DataFrame({"a": [np.inf, 1.0, -np.inf]})
        result = _clean(df)
        assert not np.isinf(result["a"]).any()

    def test_fills_nan_with_zero(self):
        df = pd.DataFrame({"a": [np.nan, 1.0]})
        result = _clean(df)
        assert not result["a"].isnull().any()


class TestChecksReturnErrorWithoutData:
    def test_drift_check_without_data(self, monitor):
        result = monitor.check_data_drift()
        assert "error" in result

    def test_quality_check_without_data(self, monitor):
        result = monitor.check_data_quality()
        assert "error" in result


class TestDataDrift:
    def test_returns_expected_keys(self, monitor_with_data):
        result = monitor_with_data.check_data_drift()
        assert "error" not in result
        for key in ("drift_detected", "drift_score", "drifted_columns", "total_columns", "timestamp"):
            assert key in result

    def test_drift_score_is_valid(self, monitor_with_data):
        result = monitor_with_data.check_data_drift()
        assert 0.0 <= result["drift_score"] <= 1.0

    def test_similar_data_has_low_drift(self, monitor_with_data):
        result = monitor_with_data.check_data_drift()
        assert result["drift_score"] < 0.5

    def test_shifted_data_has_higher_drift(self, monitor_with_data):
        shifted = _make_df(seed=0).copy()
        for col in NUMERICAL_COLS:
            shifted[col] = shifted[col] + 1000
        monitor_with_data.current_data = shifted
        result = monitor_with_data.check_data_drift()
        assert result["drift_score"] > 0.5
