import pytest
import numpy as np
import importlib
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

app_module = importlib.import_module('src.api.app')
app = app_module.app


VALID_REQUEST = {
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
    "functioning_day": "Yes"
}


@pytest.fixture
def mock_model():
    m = MagicMock()
    m.predict.return_value = np.array([500])
    return m


@pytest.fixture
def client(mock_model):
    with patch.object(app_module, 'load_best_model_from_s3', return_value=(mock_model, None, {})), \
         patch.object(app_module, 'aws_available', True), \
         patch.object(app_module, 'initialize_monitoring'):
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=VALID_REQUEST)
        assert response.status_code == 200

    def test_predict_response_shape(self, client):
        data = client.post("/predict", json=VALID_REQUEST).json()
        assert "prediction" in data
        assert "confidence" in data
        assert "processing_time_ms" in data

    def test_predict_returns_non_negative(self, client):
        data = client.post("/predict", json=VALID_REQUEST).json()
        assert data["prediction"] >= 0

    def test_predict_missing_fields_returns_422(self, client):
        response = client.post("/predict", json={"hour": 8})
        assert response.status_code == 422

    def test_predict_invalid_hour_returns_422(self, client):
        bad_request = {**VALID_REQUEST, "hour": 99}
        response = client.post("/predict", json=bad_request)
        assert response.status_code == 422

    def test_predict_invalid_humidity_returns_422(self, client):
        bad_request = {**VALID_REQUEST, "humidity": 150}
        response = client.post("/predict", json=bad_request)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, client):
        response = client.post("/predict/batch", json={"data": [VALID_REQUEST, VALID_REQUEST]})
        assert response.status_code == 200

    def test_batch_predict_response_shape(self, client):
        data = client.post("/predict/batch", json={"data": [VALID_REQUEST]}).json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["status"] == "success"


class TestMonitoringEndpoints:
    def test_monitoring_status_returns_200(self, client):
        response = client.get("/monitoring/status")
        assert response.status_code == 200

    def test_monitoring_status_fields(self, client):
        data = client.get("/monitoring/status").json()
        assert "monitoring_initialized" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
