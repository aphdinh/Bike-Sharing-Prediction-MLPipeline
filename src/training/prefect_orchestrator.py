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
    prepare_data_core, train_all_models_core, perform_hyperparameter_tuning_core,
    register_and_save_best_model_core
)
from ..utils.mlflow_utils import setup_mlflow


def _setup_mlflow():
    try:
        mlflow.end_run()
    except mlflow.exceptions.MlflowException:
        pass
    return setup_mlflow()


@task(name="prepare_training_data", tags=["data"], cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=6))
def prepare_training_data() -> Tuple:
    logger = get_run_logger()
    data = prepare_data_core()
    X_train = data[2]
    logger.info(f"Train: {len(X_train)} | Features: {X_train.shape[1]}")
    return data


@task(name="train_all_models", tags=["training"])
def train_all_models(X, y, X_train, X_test, y_train, y_test) -> Tuple[pd.DataFrame, Dict]:
    logger = get_run_logger()
    results_df, trained_models = train_all_models_core(X, y, X_train, X_test, y_train, y_test)
    logger.info(f"Trained {len(results_df)} models")
    return results_df, trained_models


@task(name="hyperparameter_optimization", tags=["tuning"], cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=12))
def perform_hyperparameter_optimization(best_model_name, X, y, X_train, y_train, X_val, y_val, X_test, y_test) -> Tuple[Optional[Dict], Any]:
    logger = get_run_logger()
    try:
        result, model = perform_hyperparameter_tuning_core(best_model_name, X, y, X_train, y_train, X_val, y_val, X_test, y_test)
        if result:
            logger.info(f"Tuning complete — R²: {result['test_r2']:.4f}, RMSE: {result['test_rmse']:.4f}")
        return result, model
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        return None, None


@task(name="register_best_model", tags=["model-registry"])
def register_and_save_best_model(results_df: pd.DataFrame, best_model: Any, best_scaler: Any) -> Tuple:
    logger = get_run_logger()
    result = register_and_save_best_model_core(results_df, best_model, best_scaler)
    logger.info(f"Registered: {result[2]}")
    return result


@flow(name="ml-training-pipeline", version="1.0.0", persist_result=True, retries=1, retry_delay_seconds=60)
def ml_training_pipeline() -> Dict[str, Any]:
    logger = get_run_logger()

    experiment_id = _setup_mlflow()
    X, y, X_train, X_val, X_test, y_train, y_val, y_test = prepare_training_data()

    results_df, trained_models = train_all_models(X, y, X_train, X_test, y_train, y_test)
    best_model_name = results_df.loc[results_df['test_r2'].idxmax(), 'model_name']
    logger.info(f"Best model from sweep: {best_model_name}")

    tuning_result, tuned_model = perform_hyperparameter_optimization(
        best_model_name, X, y, X_train, y_train, X_val, y_val, X_test, y_test
    )
    if tuning_result:
        results_df = pd.concat([results_df, pd.DataFrame([tuning_result])], ignore_index=True)

    best_row = results_df.loc[results_df['test_r2'].idxmax()]
    best_model_name = best_row['model_name']

    if best_model_name.startswith('Hyperopt_Tuned_') and tuned_model is not None:
        best_model, best_scaler = tuned_model, None
    else:
        best_model, best_scaler = trained_models.get(best_model_name, (None, None))

    updated_results_df, comparison_df, registered_model_name = register_and_save_best_model(
        results_df, best_model, best_scaler
    )

    best_r2 = best_row['test_r2']
    best_rmse = best_row['test_rmse']
    logger.info(f"Best Model: {best_model_name} — R² = {best_r2:.4f}, RMSE = {best_rmse:.2f}")

    create_markdown_artifact(
        markdown=f"# Training Results\n- **Best Model**: {best_model_name}\n- **R²**: {best_r2:.4f}\n- **RMSE**: {best_rmse:.2f}\n- **Models Evaluated**: {len(updated_results_df)}",
        key="training-report",
        description="Training pipeline report"
    )
    create_table_artifact(
        table=updated_results_df.round(4).to_dict('records'),
        key="model-results",
        description="Model results"
    )

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
        "parameters": {}
    }


if __name__ == "__main__":
    result = ml_training_pipeline()
    print(f"Pipeline completed: {result}")
