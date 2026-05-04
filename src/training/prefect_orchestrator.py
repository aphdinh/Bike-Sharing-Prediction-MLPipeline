import sys
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any
import pandas as pd
import logging
import warnings

warnings.filterwarnings('ignore')

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.server.schemas.schedules import CronSchedule
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.runtime import flow_run
import prefect
import mlflow

from .train_core import (
    validate_environment_core, setup_mlflow_core, prepare_data_core,
    train_single_model_core, perform_hyperparameter_tuning_core,
    register_and_save_best_model_core, create_training_report_core,
    get_scale_sensitive_models
)
from ..models.models import get_models


@task(name="validate_environment", tags=["setup"], cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def validate_environment() -> Dict[str, Any]:
    logger = get_run_logger()
    config = validate_environment_core()
    config['prefect_flow_run_id'] = str(flow_run.id) if flow_run else None
    config['timestamp'] = datetime.now().isoformat()
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    return config


@task(name="setup_mlflow_experiment", tags=["mlflow"], cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def setup_mlflow_experiment(config: Dict[str, Any]) -> str:
    logger = get_run_logger()
    experiment_id = setup_mlflow_core()
    if mlflow.active_run():
        mlflow.set_tag("prefect_flow_run_id", config.get('prefect_flow_run_id'))
        mlflow.set_tag("prefect_flow_name", flow_run.flow_name if flow_run else "unknown")
        mlflow.log_param("prefect_version", prefect.__version__)
    logger.info(f"MLflow experiment ID: {experiment_id}")
    return experiment_id


@task(name="prepare_training_data", tags=["data"], cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=6))
def prepare_training_data() -> Tuple:
    logger = get_run_logger()
    data = prepare_data_core()
    logger.info(f"Train: {len(data[0])} | Val: {len(data[1])} | Test: {len(data[2])} | Features: {data[0].shape[1]}")
    return data


@task(name="train_single_model", tags=["training"], retries=2, retry_delay_seconds=30)
def train_single_model(model_info, X_train, X_test, y_train, y_test, scale_sensitive_models) -> Dict[str, Any]:
    logger = get_run_logger()
    model_name, model = model_info
    result = train_single_model_core(model, X_train, X_test, y_train, y_test, model_name, scale_sensitive_models)
    logger.info(f"{model_name} — R²: {result['test_r2']:.4f}, RMSE: {result['test_rmse']:.4f}")
    return result


@task(name="train_all_models", tags=["training"])
def train_all_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    logger = get_run_logger()
    futures = [
        train_single_model.submit((name, model), X_train, X_test, y_train, y_test, get_scale_sensitive_models())
        for name, model in get_models().items()
    ]
    results = [f.result() for f in futures]
    logger.info(f"Trained {len(results)} models")
    return pd.DataFrame(results)


@task(name="hyperparameter_optimization", tags=["tuning"], cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=12))
def perform_hyperparameter_optimization(best_model_name, X_train, y_train, X_val, y_val, X_test, y_test) -> Optional[Dict[str, Any]]:
    logger = get_run_logger()
    try:
        result = perform_hyperparameter_tuning_core(best_model_name, X_train, y_train, X_val, y_val, X_test, y_test)
        if result:
            logger.info(f"Tuning complete — R²: {result['test_r2']:.4f}, RMSE: {result['test_rmse']:.4f}")
        return result
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        return None


@task(name="register_best_model", tags=["model-registry"])
def register_and_save_best_model(results_df: pd.DataFrame) -> Tuple:
    logger = get_run_logger()
    result = register_and_save_best_model_core(results_df)
    logger.info(f"Registered: {result[2]}")
    return result


@task(name="create_training_report", tags=["reporting"])
def create_training_report(config, results_df, comparison_df, best_model_name, tuning_result) -> str:
    logger = get_run_logger()
    report = create_training_report_core(config, results_df, comparison_df, best_model_name, tuning_result)
    report += f"\n## Prefect Details\n- Flow Run ID: {config.get('prefect_flow_run_id', 'N/A')}\n- Version: {prefect.__version__}\n"
    create_markdown_artifact(markdown=report, key="training-report", description="Training pipeline report")
    create_table_artifact(table=results_df.round(4).to_dict('records'), key="model-results", description="Model results")
    logger.info("Report created")
    return report


@flow(name="ml-training-pipeline", version="1.0.0", persist_result=True, retries=1, retry_delay_seconds=60)
def ml_training_pipeline() -> Dict[str, Any]:
    logger = get_run_logger()

    config = validate_environment()
    experiment_id = setup_mlflow_experiment(config)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_training_data()

    results_df = train_all_models(X_train, X_test, y_train, y_test)
    best_model_name = results_df.loc[results_df['test_r2'].idxmax(), 'model_name']
    logger.info(f"Best model: {best_model_name}")

    tuning_result = perform_hyperparameter_optimization(
        best_model_name, X_train, y_train, X_val, y_val, X_test, y_test
    )
    updated_results_df, comparison_df, registered_model_name = register_and_save_best_model(results_df)
    create_training_report(config, updated_results_df, comparison_df, best_model_name, tuning_result)

    best_r2 = updated_results_df['test_r2'].max()
    best_rmse = updated_results_df.loc[updated_results_df['test_r2'].idxmax(), 'test_rmse']
    logger.info(f"Best Model: R² = {best_r2:.4f}, RMSE = {best_rmse:.2f}")

    return {
        'status': 'success',
        'best_model': best_model_name,
        'best_r2_score': best_r2,
        'best_rmse': best_rmse,
        'total_models_trained': len(updated_results_df),
        'registered_model_name': registered_model_name,
        'mlflow_experiment_id': experiment_id,
        'flow_run_id': str(flow_run.id) if flow_run else None,
        'execution_time': datetime.now().isoformat()
    }


def create_deployment():
    return {
        "name": "ml-training-pipeline-deployment",
        "schedule": CronSchedule(cron="0 2 1 * *", timezone="UTC"),
        "work_pool_name": "default-agent-pool",
        "parameters": {"retrain_models": True, "optimize_hyperparameters": True}
    }


if __name__ == "__main__":
    result = ml_training_pipeline()
    print(f"Pipeline completed: {result}")
