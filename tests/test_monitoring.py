import pytest
import pandas as pd
import numpy as np
from src.monitoring.monitoring import ModelMonitor, _clean


@pytest.fixture
def monitor():
    m = ModelMonitor.__new__(ModelMonitor)
    m.reference_data = None
    m.current_data = None
    return m


class TestUpdateCurrentData:
    def test_sets_current_data(self, monitor):
        df = pd.DataFrame({"a": [1, 2, 3]})
        monitor.update_current_data(df)
        assert monitor.current_data is not None
        assert len(monitor.current_data) == 3

    def test_replaces_existing_data(self, monitor):
        monitor.update_current_data(pd.DataFrame({"a": [1]}))
        monitor.update_current_data(pd.DataFrame({"a": [1, 2, 3]}))
        assert len(monitor.current_data) == 3


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
