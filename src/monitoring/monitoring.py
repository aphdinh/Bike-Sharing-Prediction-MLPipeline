import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime
from typing import Dict, Any

warnings.filterwarnings('ignore', category=RuntimeWarning)

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

logger = logging.getLogger(__name__)

NUMERICAL_COLS = [
    'hour', 'temperature_c', 'humidity', 'wind_speed', 'visibility_10m',
    'dew_point_c', 'solar_radiation', 'rainfall_mm', 'snowfall_cm',
    'year', 'month', 'day_of_week', 'hour_sin', 'hour_cos',
    'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
    'temp_humidity_interaction', 'wind_rain_interaction', 'temp_solar_interaction',
]

CATEGORICAL_COLS = [
    'season', 'holiday', 'functioning_day', 'time_of_day', 'temp_feel',
    'is_weekend', 'is_rush_hour', 'has_rain', 'has_snow', 'is_holiday',
    'is_functioning', 'is_spring', 'is_summer', 'is_autumn', 'is_winter',
    'extreme_weather',
]

DATA_DEFINITION = DataDefinition(
    numerical_columns=NUMERICAL_COLS,
    categorical_columns=CATEGORICAL_COLS,
)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df


class ModelMonitor:
    def __init__(self, reference_data_path: str = "data/reference_data.csv"):
        self.current_data = None
        try:
            self.reference_data = _clean(pd.read_csv(reference_data_path))
            logger.info(f"Loaded reference data: {self.reference_data.shape}")
        except Exception as e:
            logger.warning(f"Could not load reference data: {e}")
            self.reference_data = None

    def update_current_data(self, df: pd.DataFrame):
        self.current_data = df
        logger.info(f"Updated current data: {df.shape}")

    def check_data_drift(self) -> Dict[str, Any]:
        if self.reference_data is None or self.current_data is None:
            return {"error": "Reference or current data not available"}
        try:
            ref = Dataset.from_pandas(_clean(self.reference_data), data_definition=DATA_DEFINITION)
            cur = Dataset.from_pandas(_clean(self.current_data), data_definition=DATA_DEFINITION)
            result = Report([DataDriftPreset()]).run(cur, ref)

            metrics = result.dict().get("metrics", [])
            summary = next((m for m in metrics if "DriftedColumnsCount" in m.get("metric_name", "")), None)
            column_drifts = [m for m in metrics if m.get("metric_name", "").startswith("ValueDrift")]

            drift_share = summary["value"].get("share", 0) if summary else 0
            return {
                "drift_detected": drift_share > 0.5,
                "drift_score": round(drift_share, 4),
                "drifted_columns": int(summary["value"].get("count", 0)) if summary else 0,
                "total_columns": len(column_drifts),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Drift check failed: {e}")
            return {"error": str(e)}

    def check_data_quality(self) -> Dict[str, Any]:
        if self.current_data is None:
            return {"error": "Current data not available"}
        try:
            cur = Dataset.from_pandas(_clean(self.current_data), data_definition=DATA_DEFINITION)
            ref = Dataset.from_pandas(_clean(self.reference_data), data_definition=DATA_DEFINITION) if self.reference_data is not None else None
            Report([DataSummaryPreset()]).run(cur, ref)
            return {
                "total_rows": len(self.current_data),
                "missing_values": self.current_data.isnull().sum().to_dict(),
                "duplicate_rows": int(self.current_data.duplicated().sum()),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return {"error": str(e)}

    def run_monitoring(self) -> Dict[str, Any]:
        drift = self.check_data_drift()
        quality = self.check_data_quality()
        alerts = []
        if drift.get("drift_detected"):
            alerts.append(f"Data drift detected — score {drift.get('drift_score')}")
        if quality.get("total_rows", 1) < 10:
            alerts.append(f"Low data volume: {quality.get('total_rows')} rows")
        status = "warning" if alerts else "healthy"
        return {"timestamp": datetime.now().isoformat(), "status": status,
                "alerts": alerts, "drift": drift, "quality": quality}


_monitor: ModelMonitor = None


def get_monitor() -> ModelMonitor:
    global _monitor
    if _monitor is None:
        _monitor = ModelMonitor()
    return _monitor


def initialize_monitoring(reference_data_path: str = "data/reference_data.csv") -> ModelMonitor:
    global _monitor
    _monitor = ModelMonitor(reference_data_path)
    return _monitor
