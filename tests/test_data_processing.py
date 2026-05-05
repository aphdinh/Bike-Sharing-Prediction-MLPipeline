import pytest
import pandas as pd
import numpy as np
from src.data.data_processing import feature_engineering, preprocess_data, prepare_features


RAW_ROW = {
    "date": "15/06/2018",
    "hour": 8,
    "temperature_c": 20.0,
    "humidity": 60.0,
    "wind_speed": 2.0,
    "visibility_10m": 1500.0,
    "dew_point_c": 10.0,
    "solar_radiation": 1.5,
    "rainfall_mm": 0.0,
    "snowfall_cm": 0.0,
    "season": "Summer",
    "holiday": "No Holiday",
    "functioning_day": "Yes",
    "rented_bike_count": 500,
}


@pytest.fixture
def raw_df():
    return pd.DataFrame([RAW_ROW])


@pytest.fixture
def engineered_df(raw_df):
    return feature_engineering(raw_df)


class TestFeatureEngineering:
    def test_creates_cyclic_hour_features(self, engineered_df):
        assert "hour_sin" in engineered_df.columns
        assert "hour_cos" in engineered_df.columns

    def test_cyclic_encoding_values(self, engineered_df):
        expected_sin = np.sin(2 * np.pi * 8 / 24)
        expected_cos = np.cos(2 * np.pi * 8 / 24)
        assert abs(engineered_df["hour_sin"].iloc[0] - expected_sin) < 1e-6
        assert abs(engineered_df["hour_cos"].iloc[0] - expected_cos) < 1e-6

    def test_rush_hour_detection(self, raw_df):
        rush = feature_engineering(raw_df.assign(hour=8))
        non_rush = feature_engineering(raw_df.assign(hour=11))
        assert rush["is_rush_hour"].iloc[0] == 1
        assert non_rush["is_rush_hour"].iloc[0] == 0

    def test_weekend_detection(self, raw_df):
        # 15/06/2018 is a Friday (weekday)
        assert feature_engineering(raw_df)["is_weekend"].iloc[0] == 0
        # 16/06/2018 is a Saturday
        weekend = feature_engineering(raw_df.assign(date="16/06/2018"))
        assert weekend["is_weekend"].iloc[0] == 1

    def test_season_flags(self, engineered_df):
        assert engineered_df["is_summer"].iloc[0] == 1
        assert engineered_df["is_winter"].iloc[0] == 0
        assert engineered_df["is_spring"].iloc[0] == 0

    def test_rain_flag(self, raw_df):
        no_rain = feature_engineering(raw_df.assign(rainfall_mm=0.0))
        rain = feature_engineering(raw_df.assign(rainfall_mm=2.0))
        assert no_rain["has_rain"].iloc[0] == 0
        assert rain["has_rain"].iloc[0] == 1

    def test_extreme_weather_flag(self, raw_df):
        normal = feature_engineering(raw_df)
        extreme = feature_engineering(raw_df.assign(rainfall_mm=10.0))
        assert normal["extreme_weather"].iloc[0] == 0
        assert extreme["extreme_weather"].iloc[0] == 1

    def test_interaction_features_created(self, engineered_df):
        assert "temp_humidity_interaction" in engineered_df.columns
        assert "wind_rain_interaction" in engineered_df.columns
        assert "temp_solar_interaction" in engineered_df.columns

    def test_interaction_values(self, engineered_df):
        expected = 20.0 * 60.0 / 100
        assert abs(engineered_df["temp_humidity_interaction"].iloc[0] - expected) < 1e-6


class TestPreprocessData:
    def test_returns_dataframe(self, raw_df):
        result = preprocess_data(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_no_nulls_in_output(self, raw_df):
        result = preprocess_data(raw_df)
        assert not result.isnull().any().any()

    def test_expected_columns_present(self, raw_df):
        result = preprocess_data(raw_df)
        assert "hour_sin" in result.columns
        assert "is_rush_hour" in result.columns
        assert "temp_humidity_interaction" in result.columns

    def test_categorical_columns_encoded(self, raw_df):
        result = preprocess_data(raw_df)
        assert "time_of_day" not in result.columns
        assert "temp_feel" not in result.columns
        assert any("time_of_day_" in c for c in result.columns)

    def test_missing_categorical_cols_filled_with_zero(self, raw_df):
        result = preprocess_data(raw_df)
        from src.data.data_processing import EXPECTED_CATEGORICAL_COLS
        for col in EXPECTED_CATEGORICAL_COLS:
            assert col in result.columns


class TestPrepareFeatures:
    def test_returns_x_y_features(self, engineered_df):
        X, y, features = prepare_features(engineered_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(features, list)

    def test_y_is_bike_count(self, engineered_df):
        _, y, _ = prepare_features(engineered_df)
        assert y.iloc[0] == 500

    def test_no_target_in_x(self, engineered_df):
        X, _, _ = prepare_features(engineered_df)
        assert "rented_bike_count" not in X.columns
