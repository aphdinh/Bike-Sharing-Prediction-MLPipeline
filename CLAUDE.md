# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Dependencies

Required before installing Python packages (macOS):
```bash
brew install libomp        # required by LightGBM
brew install python@3.11   # Python 3.11 recommended (3.14 not supported by pyarrow)
```

## Commands

```bash
# Install dependencies
make install              # pip install -r requirements.txt

# Train models
make train                # python src/training/train.py core
make train-prefect        # python src/training/train.py prefect

# Start API server
make api                  # uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Run tests
make test                 # tests/test_data_processing.py + tests/test_api.py
make test-all             # all tests in tests/
make test-monitoring      # tests/test_monitoring.py only

# Run a single test file
python -m pytest tests/test_data_processing.py -v

# Run a single test by name
python -m pytest tests/test_api.py::test_name -v

# Lint and format
make lint                 # flake8 + pylint on src/ and tests/
make format               # black + isort on src/ and tests/

# Monitoring
make monitor              # python src/monitoring/integration_example.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `eu-north-1` | AWS region |
| `S3_BUCKET_NAME` | `seoul-bike-sharing-aphdinh` | S3 artifact bucket |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend |
| `MLFLOW_ARTIFACT_URI` | `s3://seoul-bike-sharing-aphdinh/mlflow-artifacts/` | Artifact storage |

Run `scripts/setup/server-start.sh` to set these automatically.

## Architecture

The project is an end-to-end MLOps pipeline predicting Seoul bike rental demand (hourly, regression).

### `src/` layout

- **`data/data_processing.py`** — data loading, feature engineering, preprocessing. Entry point for all data transformation; called by both training and API.
- **`models/models.py`** — model registry (`get_models()`, `create_model()`), hyperparameter tuning (`hyperparameter_comparison()`). Covers 12+ algorithms; best model is LightGBM/XGBoost.
- **`training/train_core.py`** — core training loop: data prep → MLflow setup → multi-model training/eval → best model registration → optional S3 upload.
- **`training/prefect_orchestrator.py`** — wraps `train_core` as a Prefect flow for scheduled/orchestrated runs.
- **`training/train.py`** — CLI entry point: `core` / `prefect` / `deploy` subcommands.
- **`api/app.py`** — FastAPI service; loads production model at startup (tries MLflow registry first, falls back to S3); exposes `/predict`, `/batch-predict`, `/monitoring/*`, `/health`.
- **`api/predict.py`** — prediction logic called by the API.
- **`monitoring/monitoring.py`** — Evidently-based drift detection and performance tracking.
- **`monitoring/integration.py`** — wires monitoring into the API; `integration_example.py` is a standalone smoke test.
- **`utils/mlflow_utils.py`** — MLflow helpers: experiment setup, metric logging, model registration, production model loading.
- **`utils/aws_utils.py`** — S3 helpers: upload/download models, check bucket availability (`aws_available` flag).
- **`utils/config.py`** — shared configuration constants.

### Data flow

```
Raw CSV (data/) → data_processing.py (load + feature engineering)
                → train_core.py (train 12+ models, track with MLflow)
                → MLflow registry + S3 (artifacts/)
                → api/app.py (loads production model at startup)
                → /predict endpoint → monitoring.py (drift/perf checks)
```

### Model loading priority (API startup)

1. MLflow production model (`load_production_model_with_tracking`)
2. S3 best model (`load_best_model_from_s3`)
3. Fails with logged error if neither is available

### Tests

- `test_data_processing.py` — unit tests (marked `@pytest.mark.unit`)
- `test_api.py` — integration tests (marked `@pytest.mark.integration`)
- `test_monitoring.py` — monitoring tests (marked `@pytest.mark.monitoring`)
- `conftest.py` — shared fixtures: `sample_data`, `mock_monitor`, `mock_model`, `api_client`
