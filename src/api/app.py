from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import time
import logging
from typing import List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from ..data.data_processing import preprocess_data
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
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    temperature_c: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    wind_speed: float = Field(..., ge=0, description="Wind speed in m/s")
    visibility_10m: float = Field(..., ge=0, description="Visibility in 10m units")
    dew_point_c: float = Field(..., description="Dew point temperature in Celsius")
    solar_radiation: float = Field(..., ge=0, description="Solar radiation in MJ/m2")
    rainfall_mm: float = Field(..., ge=0, description="Rainfall in mm")
    snowfall_cm: float = Field(..., ge=0, description="Snowfall in cm")
    season: str = Field(..., description="Season: Spring, Summer, Autumn, Winter")
    holiday: str = Field(..., description="Holiday: Holiday or No Holiday")
    functioning_day: str = Field(..., description="Functioning Day: Yes or No")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted number of bikes rented")
    confidence: float = Field(..., description="Prediction confidence score")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class BatchPredictionRequest(BaseModel):
    data: List[PredictionRequest] = Field(..., description="List of prediction requests")

class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")

class MonitoringResponse(BaseModel):
    timestamp: str = Field(..., description="Monitoring timestamp")
    data_drift: Dict[str, Any] = Field(..., description="Data drift analysis")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    alerts: List[Dict[str, Any]] = Field(..., description="Generated alerts")
    summary: Dict[str, Any] = Field(..., description="Monitoring summary")

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
    global model_loaded_at
    
    logger.info("Starting Seoul Bike Sharing Prediction API...")
    
    # Initialize monitoring system
    try:
        initialize_monitoring()
        logger.info("Monitoring system initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize monitoring: {e}")
    
    if load_production_model():
        logger.info("Model loaded successfully on startup")
    else:
        pass
    
    yield
    
    logger.info("Shutting down Seoul Bike Sharing Prediction API...")

app = FastAPI(
    title="Seoul Bike Sharing Prediction API",
    description="ML model API for predicting bike sharing demand using models stored in S3",
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
    global model
    if model is None:
        if not load_production_model():
            raise HTTPException(status_code=503, detail="Model not available. Please try again later.")

def preprocess_input_data(request_data: PredictionRequest) -> pd.DataFrame:
    df = pd.DataFrame([{
        'date': request_data.date,
        'hour': request_data.hour,
        'temperature_c': request_data.temperature_c,
        'humidity': request_data.humidity,
        'wind_speed': request_data.wind_speed,
        'visibility_10m': request_data.visibility_10m,
        'dew_point_c': request_data.dew_point_c,
        'solar_radiation': request_data.solar_radiation,
        'rainfall_mm': request_data.rainfall_mm,
        'snowfall_cm': request_data.snowfall_cm,
        'season': request_data.season,
        'holiday': request_data.holiday,
        'functioning_day': request_data.functioning_day
    }])
    
    X = preprocess_data(df)
    return X

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global model, scaler
    
    start_time = time.time()
    ensure_model_loaded()
    
    X = preprocess_input_data(request)
    
    if scaler is not None:
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
    else:
        prediction = model.predict(X)[0]
    
    prediction = max(0, int(round(prediction)))
    processing_time = (time.time() - start_time) * 1000
    
    
    return PredictionResponse(
        prediction=prediction,
        confidence=0.85,
        model_info={
            "model_type": type(model).__name__,
            "verification_status": verification_status,
            "s3_bucket": S3_BUCKET_NAME
        },
        prediction_timestamp=datetime.now().isoformat(),
        processing_time_ms=processing_time
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    global model, scaler
    
    start_time = time.time()
    ensure_model_loaded()
    
    predictions = []
    
    for i, pred_request in enumerate(request.data):
        try:
            X = preprocess_input_data(pred_request)
            
            if scaler is not None:
                X_scaled = scaler.transform(X)
                prediction = model.predict(X_scaled)[0]
            else:
                prediction = model.predict(X)[0]
            
            prediction = max(0, int(round(prediction)))
            
            
            predictions.append({
                "index": i,
                "prediction": prediction,
                "confidence": 0.85,
                "status": "success"
            })
            
        except Exception as e:
            predictions.append({
                "index": i,
                "prediction": None,
                "confidence": 0.0,
                "status": "error",
                "error": str(e)
            })
    
    total_processing_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_processing_time_ms=total_processing_time,
        model_info={
            "model_type": type(model).__name__,
            "verification_status": verification_status,
            "s3_bucket": S3_BUCKET_NAME,
            "total_requests": len(request.data),
            "successful_predictions": len([p for p in predictions if p["status"] == "success"])
        }
    )

@app.get("/check-s3-model-completeness")
async def check_s3_model_completeness_endpoint():
    if not aws_available:
        return {"status": "aws_not_available", "message": "AWS S3 is not available"}
    
    completeness_info = check_s3_model_completeness()
    
    if "error" in completeness_info:
        return {"status": "error", "error": completeness_info["error"]}
    
    return {
        "status": "success",
        "completeness_info": completeness_info,
        "can_load_model": completeness_info.get("can_load", False),
        "model_exists": completeness_info.get("model_exists", False),
        "scaler_exists": completeness_info.get("scaler_exists", False)
    }

@app.post("/monitoring/data-drift")
async def generate_data_drift_report():
    """Generate data drift monitoring report"""
    try:
        monitor = get_monitor()
        report = monitor.check_data_drift()
        return {"status": "success", "report": report}
    except Exception as e:
        logger.error(f"Error generating data drift report: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/monitoring/data-quality")
async def generate_data_quality_report():
    """Generate data quality monitoring report"""
    try:
        monitor = get_monitor()
        report = monitor.check_data_quality()
        return {"status": "success", "report": report}
    except Exception as e:
        logger.error(f"Error generating data quality report: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/monitoring/comprehensive")
async def generate_comprehensive_monitoring_report():
    """Generate comprehensive monitoring report"""
    try:
        monitor = get_monitor()
        report = monitor.run_monitoring()
        return {"status": "success", "report": report}
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/monitoring/update-current-data")
async def update_current_data_for_monitoring(data: List[Dict[str, Any]]):
    """Update current data for monitoring"""
    try:
        monitor = get_monitor()
        df = pd.DataFrame(data)
        monitor.update_current_data(df)
        return {"status": "success", "message": f"Updated current data with {len(df)} rows"}
    except Exception as e:
        logger.error(f"Error updating current data: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/monitoring/status")
async def get_monitoring_status():
    """Get monitoring system status"""
    try:
        monitor = get_monitor()
        return {
            "status": "success",
            "monitoring_initialized": monitor is not None,
            "reference_data_loaded": monitor.reference_data is not None if monitor else False,
            "current_data_loaded": monitor.current_data is not None if monitor else False,
            "reports_directory": monitor.config.reports_dir if monitor else None
        }
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_loaded_at": model_loaded_at,
        "verification_status": verification_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 