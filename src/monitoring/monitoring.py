import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Any
import json
import os

warnings.filterwarnings('ignore', category=RuntimeWarning)

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, reference_data_path: str = "data/reference_data.csv"):
        self.reference_data_path = reference_data_path
        self.reference_data = None
        self.current_data = None
        self.data_definition = None
        
        os.makedirs("reports/monitoring", exist_ok=True)
        self._load_reference_data()
        
    def _load_reference_data(self):
        try:
            if os.path.exists(self.reference_data_path):
                self.reference_data = pd.read_csv(self.reference_data_path)
                logger.info(f"Loaded reference data: {self.reference_data.shape}")
            else:
                logger.warning(f"Reference data not found at {self.reference_data_path}")
                self.reference_data = None
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            self.reference_data = None
    
    def _create_data_definition(self):
        numerical_columns = [
            'hour', 'temperature_c', 'humidity', 'wind_speed', 'visibility_10m',
            'dew_point_c', 'solar_radiation', 'rainfall_mm', 'snowfall_cm',
            'year', 'month', 'day_of_week', 'hour_sin', 'hour_cos',
            'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
            'temp_humidity_interaction', 'wind_rain_interaction', 'temp_solar_interaction'
        ]
        
        categorical_columns = [
            'season', 'holiday', 'functioning_day', 'time_of_day', 'temp_feel',
            'is_weekend', 'is_rush_hour', 'has_rain', 'has_snow', 'is_holiday',
            'is_functioning', 'is_spring', 'is_summer', 'is_autumn', 'is_winter',
            'extreme_weather'
        ]
        
        if self.reference_data is not None and 'rented_bike_count' in self.reference_data.columns:
            numerical_columns.append('rented_bike_count')
        
        self.data_definition = DataDefinition(
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns
        )
    
    def _validate_data(self):
        try:
            if self.reference_data is not None:
                self.reference_data = self.reference_data.replace([np.inf, -np.inf], np.nan)
                numerical_cols = self.reference_data.select_dtypes(include=[np.number]).columns
                self.reference_data[numerical_cols] = self.reference_data[numerical_cols].fillna(0)
            
            if self.current_data is not None:
                self.current_data = self.current_data.replace([np.inf, -np.inf], np.nan)
                numerical_cols = self.current_data.select_dtypes(include=[np.number]).columns
                self.current_data[numerical_cols] = self.current_data[numerical_cols].fillna(0)
        except Exception as e:
            logger.warning(f"Data validation warning: {e}")
    
    def update_current_data(self, new_data: pd.DataFrame):
        self.current_data = new_data
        logger.info(f"Updated current data: {self.current_data.shape}")
    
    def check_data_drift(self) -> Dict[str, Any]:
        if self.reference_data is None or self.current_data is None:
            return {"error": "Reference or current data not available"}
        
        try:
            self._validate_data()
            self._create_data_definition()
            
            reference_dataset = Dataset.from_pandas(self.reference_data, data_definition=self.data_definition)
            current_dataset = Dataset.from_pandas(self.current_data, data_definition=self.data_definition)
            
            drift_report = Report([DataDriftPreset()])
            evaluation = drift_report.run(current_dataset, reference_dataset)
            
            drift_info = {
                "drift_detected": False,
                "drift_score": 0,
                "drifted_columns": 0,
                "total_columns": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                metrics = evaluation.dict().get("metrics", [])
                column_drifts = [m for m in metrics if m.get("metric_name", "").startswith("ValueDrift")]
                summary = next((m for m in metrics if "DriftedColumnsCount" in m.get("metric_name", "")), None)
                if summary:
                    val = summary.get("value", {})
                    drift_share = val.get("share", 0)
                    threshold = summary.get("config", {}).get("drift_share", 0.5)
                    drift_info["drift_detected"] = drift_share > threshold
                    drift_info["drift_score"] = round(drift_share, 4)
                    drift_info["drifted_columns"] = int(val.get("count", 0))
                    drift_info["total_columns"] = len(column_drifts)
            except Exception as e:
                logger.warning(f"Could not extract drift metrics: {e}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"reports/monitoring/data_drift_{timestamp}.html"
            evaluation.save_html(report_path)
            drift_info["report_path"] = report_path
            
            return drift_info
            
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            return {"error": str(e)}
    
    def check_data_quality(self) -> Dict[str, Any]:
        if self.current_data is None:
            return {"error": "Current data not available"}
        
        try:
            self._validate_data()
            self._create_data_definition()
            
            reference_dataset = Dataset.from_pandas(self.reference_data, data_definition=self.data_definition) if self.reference_data is not None else None
            current_dataset = Dataset.from_pandas(self.current_data, data_definition=self.data_definition)
            
            quality_report = Report([DataSummaryPreset()])
            evaluation = quality_report.run(current_dataset, reference_dataset)
            
            quality_info = {
                "total_rows": len(self.current_data),
                "missing_values": self.current_data.isnull().sum().to_dict(),
                "duplicate_rows": self.current_data.duplicated().sum(),
                "timestamp": datetime.now().isoformat()
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"reports/monitoring/data_quality_{timestamp}.html"
            evaluation.save_html(report_path)
            quality_info["report_path"] = report_path
            
            return quality_info
            
        except Exception as e:
            logger.error(f"Error checking data quality: {e}")
            return {"error": str(e)}
    
    def check_model_performance(self, predictions: List[float], actuals: List[float]) -> Dict[str, Any]:
        if not predictions or not actuals:
            return {"error": "Predictions or actuals not provided"}
        
        try:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            performance_metrics = {
                "mae": np.mean(np.abs(predictions - actuals)),
                "rmse": np.sqrt(np.mean((predictions - actuals)**2)),
                "r2_score": self._calculate_r2_score(predictions, actuals),
                "total_predictions": len(predictions),
                "timestamp": datetime.now().isoformat()
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"reports/monitoring/performance_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(performance_metrics, f, indent=2, default=str)
            performance_metrics["report_path"] = report_path
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
            return {"error": str(e)}
    
    def _calculate_r2_score(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        try:
            mean_actual = np.mean(actuals)
            ss_tot = np.sum((actuals - mean_actual) ** 2)
            ss_res = np.sum((actuals - predictions) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            return r2
        except Exception:
            return 0.0
    
    def run_monitoring(self) -> Dict[str, Any]:
        logger.info("Running monitoring checks...")
        
        drift_results = self.check_data_drift()
        quality_results = self.check_data_quality()
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "drift_check": drift_results,
            "quality_check": quality_results,
            "status": "healthy"
        }
        
        alerts = []
        
        if "error" not in drift_results:
            if drift_results.get("drift_detected", False):
                alerts.append("Data drift detected!")
                summary["status"] = "warning"
        
        if "error" not in quality_results:
            missing_total = sum(quality_results.get("missing_values", {}).values())
            if missing_total > 0:
                alerts.append(f"Found {missing_total} missing values")
                summary["status"] = "warning"
        
        summary["alerts"] = alerts
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/monitoring/comprehensive_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        summary["report_path"] = report_path
        
        logger.info(f"Monitoring complete. Status: {summary['status']}")
        return summary

monitor = None

def get_monitor() -> ModelMonitor:
    global monitor
    if monitor is None:
        monitor = ModelMonitor()
    return monitor

def initialize_monitoring(reference_data_path: str = "data/reference_data.csv") -> ModelMonitor:
    global monitor
    monitor = ModelMonitor(reference_data_path)
    logger.info("Monitoring system initialized")
    return monitor 