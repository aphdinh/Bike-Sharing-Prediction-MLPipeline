import pytest
import pandas as pd
import numpy as np
from src.monitoring.monitoring import ModelMonitor


@pytest.fixture
def monitor():
    m = ModelMonitor.__new__(ModelMonitor)
    m.reference_data = None
    m.current_data = None
    m.data_definition = None
    return m


class TestModelPerformance:
    def test_perfect_predictions(self, monitor):
        actuals = [100, 200, 300]
        result = monitor.check_model_performance(actuals, actuals)
        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["r2_score"] == pytest.approx(1.0)

    def test_mae_calculation(self, monitor):
        predictions = [100, 150, 80]
        actuals = [110, 140, 90]
        result = monitor.check_model_performance(predictions, actuals)
        expected_mae = np.mean([10, 10, 10])
        assert result["mae"] == pytest.approx(expected_mae)

    def test_rmse_calculation(self, monitor):
        predictions = [100, 200]
        actuals = [110, 190]
        result = monitor.check_model_performance(predictions, actuals)
        expected_rmse = np.sqrt(np.mean([100, 100]))
        assert result["rmse"] == pytest.approx(expected_rmse)

    def test_empty_predictions_returns_error(self, monitor):
        result = monitor.check_model_performance([], [])
        assert "error" in result

    def test_result_has_required_fields(self, monitor):
        result = monitor.check_model_performance([100], [110])
        assert "mae" in result
        assert "rmse" in result
        assert "r2_score" in result
        assert "total_predictions" in result


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


class TestDataValidation:
    def test_replaces_inf_with_zero(self, monitor):
        monitor.reference_data = pd.DataFrame({"a": [np.inf, 1.0]})
        monitor.current_data = pd.DataFrame({"a": [1.0, -np.inf]})
        monitor._validate_data()
        assert not np.isinf(monitor.reference_data["a"]).any()
        assert not np.isinf(monitor.current_data["a"]).any()

    def test_fills_nan_with_zero(self, monitor):
        monitor.reference_data = pd.DataFrame({"a": [np.nan, 1.0]})
        monitor.current_data = pd.DataFrame({"a": [1.0, np.nan]})
        monitor._validate_data()
        assert not monitor.reference_data["a"].isnull().any()
        assert not monitor.current_data["a"].isnull().any()
