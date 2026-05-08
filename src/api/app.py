from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import json
import threading
import numpy as np
import pandas as pd
import time
import logging
from typing import List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from ..data.data_processing import preprocess_data, load_data, feature_engineering
from ..utils.mlflow_utils import load_production_model_with_tracking
from ..utils.aws_utils import load_best_model_from_s3, aws_available, S3_BUCKET_NAME, check_s3_model_completeness
from ..monitoring.monitoring import initialize_monitoring, get_monitor
from ..training.train_core import main_training_pipeline

DRIFT_THRESHOLD = 0.5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _ModelState:
    """Holds all model-related state as a single object so it can be swapped atomically."""
    __slots__ = ("model", "scaler", "model_metadata", "verification_status", "model_loaded_at")

    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_metadata = None
        self.verification_status = None
        self.model_loaded_at = None


_state = _ModelState()
_state_lock = threading.Lock()
retraining_in_progress = False


class PredictionRequest(BaseModel):
    date: str = Field(..., description="Date in DD/MM/YYYY format")
    hour: int = Field(..., ge=0, le=23)
    temperature_c: float
    humidity: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(..., ge=0)
    visibility_10m: float = Field(..., ge=0)
    dew_point_c: float
    solar_radiation: float = Field(..., ge=0)
    rainfall_mm: float = Field(..., ge=0)
    snowfall_cm: float = Field(..., ge=0)
    season: str
    holiday: str
    functioning_day: str


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    model_info: Dict[str, Any]
    prediction_timestamp: str
    processing_time_ms: float


class BatchPredictionRequest(BaseModel):
    data: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    total_processing_time_ms: float
    model_info: Dict[str, Any]


def load_production_model():
    global _state
    new_state = _ModelState()

    if aws_available:
        try:
            new_state.model, new_state.scaler, new_state.model_metadata = load_best_model_from_s3()
            if new_state.model is not None:
                new_state.verification_status = "s3_loaded"
                new_state.model_loaded_at = datetime.now().isoformat()
                with _state_lock:
                    _state = new_state
                return True
        except Exception as e:
            logger.warning(f"S3 model load failed: {e}")

    try:
        new_state.model, model_info = load_production_model_with_tracking("production")
        if new_state.model is not None:
            new_state.model_metadata = model_info
            new_state.model_loaded_at = datetime.now().isoformat()
            with _state_lock:
                _state = new_state
            return True
    except Exception as e:
        logger.warning(f"MLflow load failed: {e}")

    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Seoul Bike Sharing Prediction API...")
    try:
        initialize_monitoring()
        logger.info("Monitoring initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize monitoring: {e}")
    if load_production_model():
        logger.info("Model loaded successfully on startup")
    yield
    logger.info("Shutting down Seoul Bike Sharing Prediction API...")


app = FastAPI(
    title="Seoul Bike Sharing Prediction API",
    description="ML model API for predicting bike sharing demand",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def serialize(obj):
    return json.loads(json.dumps(obj, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x)))


def ensure_model_loaded():
    if _state.model is None and not load_production_model():
        raise HTTPException(status_code=503, detail="Model not available.")


def run_prediction(request_data: PredictionRequest) -> int:
    state = _state  # snapshot reference — safe even if retraining swaps _state mid-request
    df = pd.DataFrame([request_data.dict()])
    X = preprocess_data(df)
    X_input = state.scaler.transform(X) if state.scaler is not None else X
    prediction = max(0, int(round(state.model.predict(X_input)[0])))

    try:
        monitor = get_monitor()
        if monitor:
            row = feature_engineering(df).drop(columns=['date', 'day', 'day_name'], errors='ignore')
            existing = monitor.current_data
            monitor.update_current_data(
                pd.concat([existing, row], ignore_index=True) if existing is not None else row
            )
    except Exception as e:
        logger.debug(f"Monitor update failed: {e}")

    return prediction


def get_confidence(prediction: int) -> float:
    """Confidence based on prediction magnitude relative to model RMSE.
    Higher prediction relative to RMSE = higher confidence."""
    try:
        metadata = _state.model_metadata
        rmse = None
        if isinstance(metadata, dict):
            rmse = (metadata.get("performance_metrics") or {}).get("test_rmse")
            if rmse is None:
                rmse = (metadata.get("tags") or {}).get("test_rmse_score")
        if rmse:
            rmse = float(rmse)
            return round(min(0.99, 1 / (1 + rmse / max(prediction, 1))), 2)
    except (TypeError, ValueError):
        pass
    return 0.75


def get_model_info() -> Dict[str, Any]:
    return {"model_type": type(_state.model).__name__, "verification_status": _state.verification_status, "s3_bucket": S3_BUCKET_NAME}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    ensure_model_loaded()
    prediction = run_prediction(request)
    return PredictionResponse(
        prediction=prediction,
        confidence=get_confidence(prediction),
        model_info=get_model_info(),
        prediction_timestamp=datetime.now().isoformat(),
        processing_time_ms=(time.time() - start_time) * 1000
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    start_time = time.time()
    ensure_model_loaded()
    predictions = []
    for i, pred_request in enumerate(request.data):
        try:
            pred = run_prediction(pred_request)
            predictions.append({"index": i, "prediction": pred, "confidence": get_confidence(pred), "status": "success"})
        except Exception as e:
            predictions.append({"index": i, "prediction": None, "confidence": 0.0, "status": "error", "error": str(e)})

    info = get_model_info()
    info.update({"total_requests": len(request.data), "successful_predictions": sum(1 for p in predictions if p["status"] == "success")})
    return BatchPredictionResponse(predictions=predictions, total_processing_time_ms=(time.time() - start_time) * 1000, model_info=info)


@app.get("/check-s3-model-completeness")
async def check_s3_model_completeness_endpoint():
    if not aws_available:
        return {"status": "aws_not_available"}
    info = check_s3_model_completeness()
    if "error" in info:
        return {"status": "error", "error": info["error"]}
    return {"status": "success", "completeness_info": info, "can_load_model": info.get("can_load", False),
            "model_exists": info.get("model_exists", False), "scaler_exists": info.get("scaler_exists", False)}


def run_retraining():
    global retraining_in_progress
    retraining_in_progress = True
    try:
        logger.info("Drift detected — starting retraining...")
        main_training_pipeline()
        logger.info("Retraining complete — reloading model...")
        load_production_model()
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
    finally:
        retraining_in_progress = False


@app.post("/monitoring/data-drift")
async def generate_data_drift_report(background_tasks: BackgroundTasks):
    try:
        report = serialize(get_monitor().check_data_drift())
        drift_score = report.get("drift_score", 0)
        retraining_triggered = False

        if drift_score > DRIFT_THRESHOLD and not retraining_in_progress:
            background_tasks.add_task(run_retraining)
            retraining_triggered = True
            logger.info(f"Drift score {drift_score} > {DRIFT_THRESHOLD} — retraining triggered")

        return {"status": "success", "report": report,
                "retraining_triggered": retraining_triggered,
                "retraining_in_progress": retraining_in_progress}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/monitoring/data-quality")
async def generate_data_quality_report():
    try:
        return {"status": "success", "report": serialize(get_monitor().check_data_quality())}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/monitoring/comprehensive")
async def generate_comprehensive_monitoring_report():
    try:
        return {"status": "success", "report": serialize(get_monitor().run_monitoring())}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/monitoring/update-current-data")
async def update_current_data_for_monitoring(data: List[Dict[str, Any]] = Body(...)):
    try:
        df = pd.DataFrame(data)
        get_monitor().update_current_data(df)
        return {"status": "success", "message": f"Updated current data with {len(df)} rows"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/monitoring/status")
async def get_monitoring_status():
    try:
        monitor = get_monitor()
        return {"status": "success", "monitoring_initialized": monitor is not None,
                "reference_data_loaded": monitor.reference_data is not None if monitor else False,
                "current_data_loaded": monitor.current_data is not None if monitor else False,
                "reports_directory": "reports/monitoring"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": _state.model is not None,
            "model_loaded_at": _state.model_loaded_at, "verification_status": _state.verification_status}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
