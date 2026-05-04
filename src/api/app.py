from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import pandas as pd
import time
import logging
from typing import List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from ..data.data_processing import preprocess_data, load_data, feature_engineering
from ..utils.mlflow_utils import load_production_model_with_tracking, load_model_with_s3_verification
from ..utils.aws_utils import load_best_model_from_s3, aws_available, S3_BUCKET_NAME, check_s3_model_completeness
from ..monitoring.monitoring import initialize_monitoring, get_monitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
scaler = None
model_metadata = None
verification_status = None
model_loaded_at = None


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
    global model, scaler, model_metadata, verification_status, model_loaded_at

    if aws_available:
        model, scaler, model_metadata = load_best_model_from_s3()
        if model is not None:
            verification_status = "s3_loaded"
            model_loaded_at = datetime.now().isoformat()
            return True

    model, model_info, s3_info, verification_status = load_model_with_s3_verification("production")
    if model is not None:
        model_metadata = {"model_info": model_info, "s3_info": s3_info, "verification_status": verification_status}
        model_loaded_at = datetime.now().isoformat()
        return True

    model, model_info, s3_info = load_production_model_with_tracking("production")
    if model is not None:
        model_metadata = {"model_info": model_info, "s3_info": s3_info, "verification_status": "mlflow_only"}
        model_loaded_at = datetime.now().isoformat()
        return True

    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Seoul Bike Sharing Prediction API...")
    try:
        initialize_monitoring()
        monitor = get_monitor()
        if monitor and os.path.exists("data/SeoulBikeData.csv"):
            sample = feature_engineering(load_data("data/SeoulBikeData.csv")).sample(100, random_state=42)
            sample = sample.drop(columns=['date', 'day', 'day_name'], errors='ignore')
            monitor.update_current_data(sample)
            logger.info("Monitoring current data loaded")
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


def ensure_model_loaded():
    if model is None and not load_production_model():
        raise HTTPException(status_code=503, detail="Model not available.")


def run_prediction(request_data: PredictionRequest) -> int:
    X = preprocess_data(pd.DataFrame([request_data.dict()]))
    X_input = scaler.transform(X) if scaler is not None else X
    return max(0, int(round(model.predict(X_input)[0])))


def get_model_info() -> Dict[str, Any]:
    return {"model_type": type(model).__name__, "verification_status": verification_status, "s3_bucket": S3_BUCKET_NAME}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    ensure_model_loaded()
    prediction = run_prediction(request)
    return PredictionResponse(
        prediction=prediction,
        confidence=0.85,
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
            predictions.append({"index": i, "prediction": run_prediction(pred_request), "confidence": 0.85, "status": "success"})
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


@app.post("/monitoring/data-drift")
async def generate_data_drift_report():
    try:
        return {"status": "success", "report": get_monitor().check_data_drift()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/monitoring/data-quality")
async def generate_data_quality_report():
    try:
        return {"status": "success", "report": get_monitor().check_data_quality()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/monitoring/comprehensive")
async def generate_comprehensive_monitoring_report():
    try:
        return {"status": "success", "report": get_monitor().run_monitoring()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/monitoring/update-current-data")
async def update_current_data_for_monitoring(data: List[Dict[str, Any]]):
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
    return {"status": "healthy", "model_loaded": model is not None,
            "model_loaded_at": model_loaded_at, "verification_status": verification_status}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
