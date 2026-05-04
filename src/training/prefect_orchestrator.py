import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import logging
import warnings
import os
import json

warnings.filterwarnings('ignore')

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.server.schemas.schedules import CronSchedule
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.runtime import flow_run
import prefect

from .train_core import (
    validate_environment_core, setup_mlflow_core, prepare_data_core,
    train_single_model_core, train_all_models_core, perform_hyperparameter_tuning_core,
    register_and_save_best_model_core, create_training_report_core, get_scale_sensitive_models
)
from ..models.models import get_models

@task(
    name="validate_environment",
    description="Validate environment and configurations",
    tags=["setup", "validation"],
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
def validate_environment() -> Dict[str, Any]:
    logger = get_run_logger()
    config = validate_environment_core()
    config['prefect_flow_run_id'] = str(flow_run.id) if flow_run else None
    config['timestamp'] = datetime.now().isoformat()
    logger.info("=== ML Training Pipeline Environment ===")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    return config

@task(
    name="setup_mlflow_experiment",
    description="Setup MLflow experiment with Prefect context",
    tags=["mlflow", "setup"],
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)

def setup_mlflow_experiment(config: Dict[str, Any]) -> str:
    logger = get_run_logger()
    
    logger.info("Setting up MLflow experiment...")
    experiment_id = setup_mlflow_core()
    
    import mlflow
    if mlflow.active_run():
        mlflow.set_tag("prefect_flow_run_id", config.get('prefect_flow_run_id'))
        mlflow.set_tag("prefect_flow_name", flow_run.flow_name if flow_run else "unknown")
        mlflow.set_tag("environment_validated", True)
        mlflow.log_param("prefect_version", prefect.__version__)
    
    logger.info(f"MLflow experiment setup complete. ID: {experiment_id}")
    return experiment_id

@task(
    name="prepare_training_data",
    description="Load, validate, and prepare training data",
    tags=["data", "preprocessing"],
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=6)
)

def prepare_training_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                   pd.Series, pd.Series, pd.Series]:
    logger = get_run_logger()
    
    logger.info("Loading and preprocessing data...")
    data = prepare_data_core()
    
    logger.info(f"Data split complete:")
    logger.info(f"  Train: {len(data[0])} samples")
    logger.info(f"  Validation: {len(data[1])} samples") 
    logger.info(f"  Test: {len(data[2])} samples")
    logger.info(f"  Features: {data[0].shape[1]}")
    
    return data

@task(
    name="train_single_model",
    description="Train and evaluate a single model",
    tags=["training", "model"],
    retries=2,
    retry_delay_seconds=30
)
def train_single_model(
    model_info: Tuple[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scale_sensitive_models: set
) -> Dict[str, Any]:
    logger = get_run_logger()
    
    model_name, model = model_info
    logger.info(f"Training {model_name}...")
    
    try:
        result = train_single_model_core(
            model, X_train, X_test, y_train, y_test, model_name, scale_sensitive_models
        )
        
        logger.info(f"{model_name} training completed successfully")
        logger.info(f"  Test R²: {result['test_r2']:.4f}")
        logger.info(f"  Test RMSE: {result['test_rmse']:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to train {model_name}: {str(e)}")
        raise

@task(
    name="train_all_models",
    description="Train all models in parallel",
    tags=["training", "all_models"]
)
def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> pd.DataFrame:
    logger = get_run_logger()
    
    logger.info("Training all models...")
    
    futures = []
    for name, model in get_models().items():
        future = train_single_model.submit(
            (name, model), X_train, X_test, y_train, y_test, 
            get_scale_sensitive_models()
        )
        futures.append(future)
    
    results = []
    for future in futures:
        result = future.result()
        results.append(result)
    
    results_df = pd.DataFrame(results)
    logger.info(f"Successfully trained {len(results)} models")
    
    return results_df

@task(
    name="perform_hyperparameter_optimization",
    description="Perform hyperparameter tuning for best model",
    tags=["optimization", "tuning"],
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=12)
)
def perform_hyperparameter_optimization(
    best_model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Optional[Dict[str, Any]]:
    logger = get_run_logger()
    logger.info(f"Starting hyperparameter optimization for {best_model_name}")
    try:
        result = perform_hyperparameter_tuning_core(
            best_model_name, X_train, y_train, X_val, y_val, X_test, y_test
        )
        if result:
            logger.info(f"Optimization complete:")
            logger.info(f"  Final test RMSE: {result['test_rmse']:.4f}")
            logger.info(f"  Final test R²: {result['test_r2']:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {str(e)}")
        return None

@task(
    name="register_and_save_best_model",
    description="Register best model and save results",
    tags=["model-registry", "persistence"]
)
def register_and_save_best_model(
    results_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    logger = get_run_logger()
    logger.info("Registering best model and saving results...")
    result = register_and_save_best_model_core(results_df)
    logger.info(f"Best model '{result[2]}' registered successfully")
    return result

@task(
    name="create_training_report",
    description="Create comprehensive training report",
    tags=["reporting", "artifacts"]
)
def create_training_report(
    config: Dict[str, Any],
    results_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    best_model_name: str,
    tuning_result: Optional[Dict[str, Any]]
) -> str:
    logger = get_run_logger()
    report = create_training_report_core(
        config, results_df, comparison_df, best_model_name, tuning_result
    )
    
    report += f"""
## Prefect Execution Details
- **Flow Run ID**: {config.get('prefect_flow_run_id', 'N/A')}
- **Prefect Version**: {prefect.__version__}
- **Execution Time**: {config['timestamp']}
"""
    
    create_markdown_artifact(
        markdown=report,
        key="training-report",
        description="Comprehensive ML training pipeline report"
    )
    
    create_table_artifact(
        table=results_df.round(4).to_dict('records'),
        key="model-results",
        description="Detailed model performance results"
    )
    
    logger.info("Training report created successfully")
    return report

@flow(
    name="ml-training-pipeline",
    description="Complete ML training pipeline with Prefect orchestration",
    version="1.0.0",
    flow_run_name="ml-training-pipeline",
    task_runner=None,
    persist_result=True,
    retries=1,
    retry_delay_seconds=60
)
def ml_training_pipeline() -> Dict[str, Any]:

    logger = get_run_logger()
    logger.info("🚀 Starting ML Training Pipeline")
    
    config = validate_environment()
    experiment_id = setup_mlflow_experiment(config)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_training_data()
    
    logger.info("Training multiple models...")
    results_df = train_all_models(X_train, X_test, y_train, y_test)
    best_model_name = results_df.loc[results_df['test_r2'].idxmax(), 'model_name']
    logger.info(f"Best performing model: {best_model_name}")
    logger.info("Performing hyperparameter optimization...")
    tuning_result = perform_hyperparameter_optimization(
        best_model_name, X_train, y_train, X_val, y_val, X_test, y_test
    )
    logger.info("Registering best model and saving results...")
    updated_results_df, comparison_df, registered_model_name = register_and_save_best_model(
        results_df
    )
    
    logger.info("Creating training report...")
    report = create_training_report(
        config, updated_results_df, comparison_df, best_model_name, tuning_result
    )
    best_r2 = updated_results_df['test_r2'].max()
    best_rmse = updated_results_df.loc[updated_results_df['test_r2'].idxmax(), 'test_rmse']
    
    logger.info(f"Best Model Performance: R² = {best_r2:.4f}, RMSE = {best_rmse:.2f}")
    logger.info(f"MLflow UI: {config['mlflow_tracking_uri']}")
    
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
    deployment_config = {
        "name": "ml-training-pipeline-deployment",
        "version": "1.0.0",
        "description": "Automated ML training pipeline for Seoul bike sharing prediction",
        "tags": ["ml", "training", "seoul-bike", "production"],
        "schedule": CronSchedule(cron="0 2 1 * *", timezone="UTC"),
        "work_pool_name": "default-agent-pool",
        "parameters": {
            "retrain_models": True,
            "optimize_hyperparameters": True,
            "deploy_best_model": False
        }
    }
    
    return deployment_config

def main():
    print("Running ML training pipeline with Prefect...")
    result = ml_training_pipeline()
    print(f"Pipeline completed successfully!")
    print(f"Results: {result}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        print("🚀 Creating deployment using modern Prefect API...")

        main()
    else:
        main() 