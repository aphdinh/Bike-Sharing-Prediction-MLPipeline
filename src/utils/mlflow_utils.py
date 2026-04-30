import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import logging
import warnings
import json
from datetime import datetime
import pickle
import joblib
from typing import Dict, List, Tuple, Optional, Any

from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .aws_utils import upload_to_s3, AWS_REGION, S3_BUCKET_NAME, aws_available
from .aws_utils import load_best_model_from_s3
import boto3

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', "seoul-bike-sharing")
MLFLOW_ARTIFACT_URI = os.getenv('MLFLOW_ARTIFACT_URI')

def status(msg):
    logging.info(msg)

def setup_mlflow():
    global EXPERIMENT_NAME
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # First try to get the experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        
        if experiment is None:
            # Experiment doesn't exist, create it with S3 artifact store
            if MLFLOW_ARTIFACT_URI:
                experiment_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_URI)
                print(f"✅ Created new experiment: {EXPERIMENT_NAME} with S3 artifact store: {MLFLOW_ARTIFACT_URI}")
            else:
                experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
                print(f"✅ Created new experiment: {EXPERIMENT_NAME}")
        elif experiment.lifecycle_stage == "deleted":
            # Experiment exists but is deleted, try to restore it
            try:
                client = mlflow.MlflowClient()
                client.restore_experiment(experiment.experiment_id)
                experiment_id = experiment.experiment_id
                print(f"✅ Restored deleted experiment: {EXPERIMENT_NAME}")
            except Exception as e:
                print(f"⚠️ Could not restore experiment: {e}")
                # Create new experiment with same name (this will work if the deleted one is permanently deleted)
                if MLFLOW_ARTIFACT_URI:
                    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_URI)
                    print(f"✅ Created new experiment with same name: {EXPERIMENT_NAME} with S3 artifact store")
                else:
                    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
                    print(f"✅ Created new experiment with same name: {EXPERIMENT_NAME}")
        else:
            # Experiment exists and is active
            experiment_id = experiment.experiment_id
            print(f"✅ Using existing experiment: {EXPERIMENT_NAME}")
            
    except mlflow.exceptions.MlflowException as e:
        # If we can't create with the same name, try to permanently delete and recreate
        try:
            client = mlflow.MlflowClient()
            experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
            if experiment and experiment.lifecycle_stage == "deleted":
                client.delete_experiment(experiment.experiment_id)
                print(f"🗑️ Permanently deleted experiment: {EXPERIMENT_NAME}")
        except:
            pass
        
        # Now try to create the experiment again
        try:
            if MLFLOW_ARTIFACT_URI:
                experiment_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_URI)
                print(f"✅ Created new experiment: {EXPERIMENT_NAME} with S3 artifact store")
            else:
                experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
                print(f"✅ Created new experiment: {EXPERIMENT_NAME}")
        except:
            # Last resort: use timestamp
            experiment_name_with_timestamp = f"{EXPERIMENT_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if MLFLOW_ARTIFACT_URI:
                experiment_id = mlflow.create_experiment(experiment_name_with_timestamp, artifact_location=MLFLOW_ARTIFACT_URI)
                print(f"⚠️ Created experiment with timestamp (fallback): {EXPERIMENT_NAME} with S3 artifact store")
            else:
                experiment_id = mlflow.create_experiment(experiment_name_with_timestamp)
                print(f"⚠️ Created experiment with timestamp (fallback): {EXPERIMENT_NAME}")
            EXPERIMENT_NAME = experiment_name_with_timestamp
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    if mlflow.active_run():
        mlflow.log_param("aws_region", AWS_REGION)
        mlflow.log_param("s3_bucket", S3_BUCKET_NAME)
        mlflow.log_param("aws_available", aws_available)
        if MLFLOW_ARTIFACT_URI:
            mlflow.log_param("artifact_uri", MLFLOW_ARTIFACT_URI)
    
    return experiment_id

def log_metrics(metrics):
    print(f"\n Logging metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            if np.isfinite(v) and not np.isnan(v):
                mlflow.log_metric(k, v)
            else:
                status(f"Warning: Skipping metric '{k}' with invalid value: {v}")
                mlflow.log_metric(k, 0.0)

def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    try:
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            print(f"  Raw MAPE: {mape} (calculated from {non_zero_mask.sum()} non-zero values)")
        else:
            mape = 0.0
            print(f"  Raw MAPE: {mape} (no non-zero values found)")
    except Exception as e:
        mape = 0.0
        print(f"  Raw MAPE: {mape} (error in calculation: {e})")
    
    metrics = {
        'rmse': float(rmse) if np.isfinite(rmse) else 0.0,
        'mae': float(mae) if np.isfinite(mae) else 0.0,
        'r2': float(r2) if np.isfinite(r2) else 0.0,
        'mape': float(mape) if np.isfinite(mape) else 0.0
    }
    
    return metrics

def create_prediction_plots(y_test, y_pred, model_name):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue', s=10)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Bike Count')
    axes[0, 0].set_ylabel('Predicted Bike Count')
    axes[0, 0].set_title(f'{model_name}: Predicted vs Actual')
    
    try:
        r2 = r2_score(y_test, y_pred)
        if not np.isfinite(r2) or np.isnan(r2):
            r2 = 0.0
    except:
        r2 = 0.0
    
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green', s=10)
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Bike Count')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title(f'{model_name}: Residuals Plot')
    
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].axvline(x=0, color='red', linestyle='--')
    
    bins = pd.cut(y_test, bins=10)
    error_by_range = pd.DataFrame({
        'Actual_Range': bins,
        'Absolute_Error': np.abs(residuals)
    }).groupby('Actual_Range')['Absolute_Error'].mean()
    
    axes[1, 1].bar(range(len(error_by_range)), error_by_range.values, alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Actual Value Ranges')
    axes[1, 1].set_ylabel('Mean Absolute Error')
    axes[1, 1].set_title('Prediction Error by Value Range')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    plot_file = f"prediction_analysis_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    
    mlflow.log_artifact(plot_file, artifact_path="plots")
    s3_key = f"models/{model_name.lower().replace(' ', '_')}/prediction_analysis.png"
    upload_to_s3(plot_file, s3_key)
    
    # Clean up local plot file after uploading to S3
    if os.path.exists(plot_file):
        os.remove(plot_file)
    
    plt.close()


def register_best_model(results_df):
    client = MlflowClient()
    
    # Find the best model based on R² score
    best_info = results_df.iloc[results_df['test_r2'].idxmax()]
    best_r2 = best_info['test_r2']
    best_run_id = best_info['run_id']
    best_name = best_info['model_name']
    
    status(f"Registering best model: {best_name} (R²: {best_r2:.4f})")
    
    # Check for hyperparameter tuned models that might be better
    try:
        for model in client.search_registered_models():
            if "tuned" in model.name.lower():
                for version in client.search_model_versions(f"name='{model.name}'"):
                    try:
                        metrics = client.get_run(version.run_id).data.metrics
                        r2 = metrics.get('test_r2') or metrics.get('test_r2_score', 0)
                        if r2 > best_r2:
                            best_r2, best_run_id, best_name = r2, version.run_id, model.name
                            status(f"Found better tuned model: {model.name} (R²: {r2:.4f})")
                    except:
                        continue
    except:
        pass
    
    # Create or get the production model registry
    prod_name = "seoul_bike_production_model"
    
    try:
        client.create_registered_model(prod_name)
        status(f"✅ Created registered model: {prod_name}")
    except mlflow.exceptions.MlflowException:
        status(f"✅ Using existing registered model: {prod_name}")
    
    # Create new model version
    model_uri = f"runs:/{best_run_id}/model"
    version = client.create_model_version(
        name=prod_name,
        source=model_uri,
        run_id=best_run_id,
        description=f"Best: {best_name} (R²: {best_r2:.4f})"
    )
    
    # Transition to Production stage
    client.transition_model_version_stage(
        name=prod_name,
        version=version.version,
        stage="Production",
        archive_existing_versions=True
    )
    
    # Get additional metrics
    try:
        metrics = client.get_run(best_run_id).data.metrics
        rmse, mae = metrics.get('test_rmse', 'N/A'), metrics.get('test_mae', 'N/A')
    except:
        rmse = mae = 'N/A'
    
    # Update model version description
    client.update_model_version(
        name=prod_name,
        version=version.version,
        description=f"Best: {best_name} | R²: {best_r2:.4f} | RMSE: {rmse} | MAE: {mae}"
    )
    
    # Set model version tags
    tags = {
        "best_model_name": best_name,
        "test_r2_score": str(best_r2),
        "test_rmse_score": str(rmse),
        "test_mae_score": str(mae),
        "is_hyperparameter_tuned": str("tuned" in best_name.lower()),
        "timestamp": datetime.now().isoformat()
    }
    
    for key, value in tags.items():
        client.set_model_version_tag(prod_name, version.version, key, value)
    
    # Set aliases for easy access
    aliases = {
        "production": "Current production model",
        "champion": "Best performing model",
        "latest": "Most recent model",
        "best": "Best model by R² score"
    }
    
    for alias, description in aliases.items():
        try:
            client.set_registered_model_alias(prod_name, alias, version.version)
            status(f"✅ Set alias '{alias}' for {prod_name}")
        except mlflow.exceptions.MlflowException:
            status(f"⚠️  Alias '{alias}' already exists for {prod_name}")
    
    return {
        "model_name": prod_name,
        "version": version.version,
        "run_id": best_run_id,
        "best_model_name": best_name,
        "test_r2_score": best_r2,
        "test_rmse_score": rmse,
        "test_mae_score": mae,
        "aliases": list(aliases.keys())
    }

def load_production_model():
    client = MlflowClient()
    model_name = "seoul_bike_production_model"
    
    try:
        # Try to load using 'production' alias first
        model_uri = f"models:/{model_name}/production"
        model = mlflow.sklearn.load_model(model_uri)
        status(f"✅ Loaded production model using 'production' alias")
        return model, None
    except Exception as e:
        status(f"⚠️ Could not load using 'production' alias: {e}")
        
        try:
            # Fallback to Production stage
            model_version = client.get_latest_versions(
                model_name, stages=["Production"]
            )[0]
            
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            
            status(f"✅ Loaded production model version {model_version.version}")
            return model, model_version
        except IndexError:
            status("❌ No production model found in MLflow registry")
            return None, None

def load_production_model_with_tracking(alias="production"):
    client = MlflowClient()
    model_name = "seoul_bike_production_model"
    
    try:
        model_version = client.get_model_version_by_alias(model_name, alias)
        
        model_uri = f"models:/{model_name}/{alias}"
        model = mlflow.sklearn.load_model(model_uri)
        
        model_info = {
            "model_name": model_name,
            "version": model_version.version,
            "stage": model_version.current_stage,
            "description": model_version.description,
            "run_id": model_version.run_id,
            "alias": alias,
            "tags": model_version.tags,
            "creation_timestamp": model_version.creation_timestamp,
            "last_updated_timestamp": model_version.last_updated_timestamp
        }
        
        # Get S3 tracking information
        s3_info = get_s3_tracking_info(model_version.run_id, model_name, model_version.version)
        
        status(f"✅ Loaded production model using '{alias}' alias")
        status(f"   Model Version: {model_version.version}")
        status(f"   Run ID: {model_version.run_id}")
        status(f"   S3 Artifacts: {s3_info.get('artifact_count', 0)} files tracked")
        
        return model, model_info, s3_info
        
    except Exception as e:
        status(f"❌ Error loading model with alias '{alias}': {e}")
        
        # Fallback to Production stage
        try:
            model_version = client.get_latest_versions(
                model_name, stages=["Production"]
            )[0]
            
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            
            model_info = {
                "model_name": model_name,
                "version": model_version.version,
                "stage": "Production",
                "description": model_version.description,
                "run_id": model_version.run_id,
                "alias": "Production (fallback)",
                "tags": model_version.tags,
                "creation_timestamp": model_version.creation_timestamp,
                "last_updated_timestamp": model_version.last_updated_timestamp
            }
            
            s3_info = get_s3_tracking_info(model_version.run_id, model_name, model_version.version)
            
        except IndexError:
            status("❌ No production model found in MLflow registry")
            return None, None, None

def get_s3_tracking_info(run_id, model_name, version):
    """
    Get S3 tracking information for a specific model version.
    
    Args:
        run_id (str): MLflow run ID
        model_name (str): Name of the model
        version (int): Model version number
    
    Returns:
        dict: S3 tracking information
    """
    try:
        # Get MLflow run information
        client = MlflowClient()
        run = client.get_run(run_id)
        
        # Extract S3 artifact information
        s3_info = {
            "run_id": run_id,
            "model_name": model_name,
            "version": version,
            "artifact_uri": run.info.artifact_uri,
            "artifact_count": 0,
            "s3_artifacts": [],
            "model_files": [],
            "metadata_files": []
        }
        
        # List artifacts in the run
        artifacts = client.list_artifacts(run_id)
        for artifact in artifacts:
            s3_info["artifact_count"] += 1
            s3_info["s3_artifacts"].append({
                "path": artifact.path,
                "is_dir": artifact.is_dir,
                "file_size": artifact.file_size if hasattr(artifact, 'file_size') else None
            })
            
            # Categorize artifacts
            if artifact.path.startswith("model"):
                s3_info["model_files"].append(artifact.path)
            elif artifact.path.endswith((".json", ".yaml", ".yml")):
                s3_info["metadata_files"].append(artifact.path)
        
        # Get S3 bucket information if available
        if aws_available:
            try:
                # Check if model artifacts exist in S3
                s3_prefix = f"mlflow/{run_id}/artifacts/"
                s3_objects = list_s3_artifacts(s3_prefix)
                s3_info["s3_bucket"] = S3_BUCKET_NAME
                s3_info["s3_prefix"] = s3_prefix
                s3_info["s3_objects_count"] = len(s3_objects)
                s3_info["s3_objects"] = s3_objects[:10]  # Limit to first 10 for display
            except Exception as e:
                s3_info["s3_error"] = str(e)
        
        return s3_info
        
    except Exception as e:
        return {
            "error": f"Failed to get S3 tracking info: {e}",
            "run_id": run_id,
            "model_name": model_name,
            "version": version
        }

def list_s3_artifacts(prefix):
    """
    List artifacts in S3 bucket with given prefix.
    """
    try:
        from .aws_utils import list_s3_artifacts
        return list_s3_artifacts(prefix)
    except ImportError:
        return []

def get_model_lineage_by_alias(alias="production"):
    """
    Get complete model lineage information including MLflow and S3 tracking.
    
    Args:
        alias (str): The alias to track
    
    Returns:
        dict: Complete lineage information
    """
    model, model_info, s3_info = load_production_model_with_tracking(alias)
    
    if model is None:
        return None
    
    lineage = {
        "alias": alias,
        "model_info": model_info,
        "s3_tracking": s3_info,
        "model_type": type(model).__name__,
        "model_parameters": getattr(model, 'get_params', lambda: {})() if hasattr(model, 'get_params') else {},
        "loading_timestamp": datetime.now().isoformat()
    }
    
    return lineage

def load_model_by_alias(model_name, alias):
    """
    Load a model using a specific alias from MLflow Model Registry.
    """
    try:
        model_uri = f"models:/{model_name}/{alias}"
        model = mlflow.sklearn.load_model(model_uri)
        status(f"✅ Loaded model {model_name} using alias '{alias}'")
        return model
    except Exception as e:
        status(f"❌ Error loading model {model_name} with alias '{alias}': {e}")
        return None

def load_model_by_alias_with_metadata(alias="production"):
    """
    Load model by alias with comprehensive metadata.
    
    Args:
        alias (str): The alias to use for loading
    
    Returns:
        tuple: (model, metadata) where metadata contains all tracking information
    """
    model, model_info, s3_info = load_production_model_with_tracking(alias)
    
    if model is None:
        return None, None
    
    metadata = {
        "model_info": model_info,
        "s3_tracking": s3_info,
        "model_type": type(model).__name__,
        "alias_used": alias,
        "load_timestamp": datetime.now().isoformat()
    }
    
    return model, metadata

def get_best_model():
    """
    Get the best model using the 'best' alias.
    """
    return load_model_by_alias("seoul_bike_production_model", "best")

def get_production_model():         
    """
    Get the production model using the 'production' alias.
    """
    return load_model_by_alias("seoul_bike_production_model", "production")

def get_champion_model():
    """
    Get the champion model using the 'champion' alias.
    """
    return load_model_by_alias("seoul_bike_production_model", "champion")

def get_latest_model(): 
    """
    Get the latest model using the 'latest' alias.
    """
    return load_model_by_alias("seoul_bike_production_model", "latest")

def get_model_info_by_alias(model_name, alias):
    """
    Get model information using a specific alias.
    """
    client = MlflowClient()
    try:
        model_version = client.get_model_version_by_alias(model_name, alias)
        return {
            "model_name": model_name,
            "version": model_version.version,
            "stage": model_version.current_stage,
            "description": model_version.description,
            "tags": model_version.tags,
            "run_id": model_version.run_id
        }
    except Exception as e:
        status(f"❌ Error getting model info for {model_name} with alias '{alias}': {e}")
        return None

def get_best_model_info():
    """
    Get information about the best model using MLflow Model Registry.
    This replaces the JSON file approach with MLflow's built-in model management.
    """
    # Try to get production model info using alias
    production_info = get_model_info_by_alias("seoul_bike_production_model", "production")
    
    if production_info:
        return {
            "mlflow_info": {
                "model_name": production_info["model_name"],
                "version": production_info["version"],
                "stage": production_info["stage"],
                "description": production_info["description"],
                "run_id": production_info["run_id"]
            },
            "model_metadata": {
                "best_model_name": production_info["tags"].get("best_model_name", "Unknown"),
                "test_r2_score": production_info["tags"].get("test_r2_score", "Unknown"),
                "test_rmse_score": production_info["tags"].get("test_rmse_score", "Unknown"),
                "test_mae_score": production_info["tags"].get("test_mae_score", "Unknown"),
                "is_hyperparameter_tuned": production_info["tags"].get("is_hyperparameter_tuned", "Unknown"),
                "timestamp": production_info["tags"].get("timestamp", "Unknown")
            },
            "aliases": {
                "production": "Current production model",
                "champion": "Best performing model", 
                "latest": "Most recent model",
                "best": "Best model by R² score"
            }
        }
    
    # Fallback: try to get from Production stage
    client = MlflowClient()
    model_name = "seoul_bike_production_model"
    
    try:
        model_version = client.get_latest_versions(
            model_name, stages=["Production"]
        )[0]
        
        return {
            "mlflow_info": {
                "model_name": model_name,
                "version": model_version.version,
                "stage": "Production",
                "description": model_version.description,
                "run_id": model_version.run_id
            },
            "model_metadata": {
                "best_model_name": model_version.tags.get("best_model_name", "Unknown"),
                "test_r2_score": model_version.tags.get("test_r2_score", "Unknown"),
                "test_rmse_score": model_version.tags.get("test_rmse_score", "Unknown"),
                "test_mae_score": model_version.tags.get("test_mae_score", "Unknown"),
                "is_hyperparameter_tuned": model_version.tags.get("is_hyperparameter_tuned", "Unknown"),
                "timestamp": model_version.tags.get("timestamp", "Unknown")
            }
        }
        
    except IndexError:
        status("❌ No production model found in MLflow registry")
        return None

def compare_models_mlflow(experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        status(f"Experiment {experiment_name} not found")
        return
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.test_r2 DESC"]
    )
    
    comparison_data = []
    for run in runs:
        if 'test_r2' in run.data.metrics:
            comparison_data.append({
                'run_id': run.info.run_id,
                'model_name': run.data.params.get('model_type', 'Unknown'),
                'test_r2': run.data.metrics.get('test_r2', 0),
                'test_rmse': run.data.metrics.get('test_rmse', float('inf')),
                'test_mae': run.data.metrics.get('test_mae', float('inf')),
                'overfitting_score': run.data.metrics.get('overfitting_score', 0),
                'start_time': run.info.start_time
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    status("\nModel Comparison Results:")
    status(comparison_df.to_string(index=False))
    
    return comparison_df 

def log_s3_artifacts_to_mlflow(run_id, model_name, s3_artifacts):
    try:
        client = MlflowClient()
        
        # Log S3 artifact metadata to MLflow
        s3_metadata = {
            "s3_bucket": S3_BUCKET_NAME,
            "s3_prefix": f"models/{model_name.lower().replace(' ', '_')}/",
            "artifact_count": len(s3_artifacts),
            "artifacts": s3_artifacts,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save S3 metadata as JSON artifact
        s3_metadata_file = f"s3_artifacts_{model_name.lower().replace(' ', '_')}.json"
        with open(s3_metadata_file, 'w') as f:
            json.dump(s3_metadata, f, indent=2)
        
        # Log to MLflow
        client.log_artifact(run_id, s3_metadata_file, artifact_path="s3_tracking")
        
        # Clean up
        if os.path.exists(s3_metadata_file):
            os.remove(s3_metadata_file)
            
        status(f"✅ S3 artifacts logged to MLflow run {run_id}")
        
    except Exception as e:
        status(f"⚠️ Failed to log S3 artifacts to MLflow: {e}")

def register_model_with_s3_tracking(model, model_name, run_id, scaler=None, additional_artifacts=None):

    try:
        # 1. Save model to S3
        s3_artifacts = save_model_to_s3_with_tracking(model, model_name, scaler)
        
        # 2. Log S3 artifacts to MLflow
        log_s3_artifacts_to_mlflow(run_id, model_name, s3_artifacts)
        
        # 3. Register model in MLflow Model Registry
        client = MlflowClient()
        model_uri = f"runs:/{run_id}/model"
        
        # Create registered model if it doesn't exist
        registered_model_name = f"seoul_bike_{model_name.lower().replace(' ', '_')}"
        try:
            client.create_registered_model(registered_model_name)
        except mlflow.exceptions.MlflowException:
            pass  # Model already exists
        
        # Create model version
        version = client.create_model_version(
            name=registered_model_name,
            source=model_uri,
            run_id=run_id,
            description=f"{model_name} - S3 tracked"
        )
        
        # Set S3 tracking tags
        s3_tags = {
            "s3_bucket": S3_BUCKET_NAME,
            "s3_model_path": s3_artifacts.get("model_path", ""),
            "s3_scaler_path": s3_artifacts.get("scaler_path", ""),
            "s3_artifacts_count": str(len(s3_artifacts)),
            "registration_timestamp": datetime.now().isoformat()
        }
        
        for key, value in s3_tags.items():
            client.set_model_version_tag(registered_model_name, version.version, key, value)
        
        # Add additional artifacts if provided
        if additional_artifacts:
            for artifact_name, artifact_path in additional_artifacts.items():
                s3_key = f"models/{model_name.lower().replace(' ', '_')}/{artifact_name}"
                if upload_to_s3(artifact_path, s3_key):
                    s3_artifacts[artifact_name] = s3_key
        
        registration_info = {
            "model_name": registered_model_name,
            "version": version.version,
            "run_id": run_id,
            "s3_artifacts": s3_artifacts,
            "model_uri": model_uri
        }
        
        status(f"✅ Model {model_name} registered with S3 tracking")
        return registration_info
        
    except Exception as e:
        status(f"❌ Failed to register model with S3 tracking: {e}")
        return None

def save_model_to_s3_with_tracking(model, model_name, scaler=None):
    """
    Save model to S3 with comprehensive tracking information.
    
    Args:
        model: The trained model
        model_name (str): Name of the model
        scaler: Optional scaler object
    
    Returns:
        dict: S3 artifact tracking information
    """
    s3_artifacts = {}
    
    try:
        # Save model
        model_file = f"model_{model_name.lower().replace(' ', '_').replace('-', '_')}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        
        s3_key = f"models/{model_name.lower().replace(' ', '_')}/model.pkl"
        if upload_to_s3(model_file, s3_key):
            s3_artifacts["model_path"] = s3_key
            s3_artifacts["model_file"] = model_file
            status(f"✅ Model uploaded to S3: {s3_key}")
        
        # Save scaler if provided
        if scaler is not None:
            scaler_file = f"scaler_{model_name.lower().replace(' ', '_').replace('-', '_')}.pkl"
            with open(scaler_file, "wb") as f:
                pickle.dump(scaler, f)
            
            scaler_s3_key = f"models/{model_name.lower().replace(' ', '_')}/scaler.pkl"
            if upload_to_s3(scaler_file, scaler_s3_key):
                s3_artifacts["scaler_path"] = scaler_s3_key
                s3_artifacts["scaler_file"] = scaler_file
                status(f"✅ Scaler uploaded to S3: {scaler_s3_key}")
            
            # Clean up local scaler file
            if os.path.exists(scaler_file):
                os.remove(scaler_file)
        
        # Clean up local model file
        if os.path.exists(model_file):
            os.remove(model_file)
        
        # Add metadata
        s3_artifacts["timestamp"] = datetime.now().isoformat()
        s3_artifacts["model_name"] = model_name
        s3_artifacts["model_type"] = type(model).__name__
        
        return s3_artifacts
        
    except Exception as e:
        status(f"❌ Failed to save model to S3: {e}")
        return s3_artifacts

def get_s3_artifacts_from_mlflow(run_id):
    """
    Get S3 artifacts information from MLflow run.
    
    Args:
        run_id (str): MLflow run ID
    
    Returns:
        dict: S3 artifacts information
    """
    try:
        client = MlflowClient()
        
        # Get S3 tracking artifacts
        artifacts = client.list_artifacts(run_id, path="s3_tracking")
        
        for artifact in artifacts:
            if artifact.path.endswith("s3_artifacts.json"):
                # Download and read S3 metadata
                local_path = f"temp_s3_artifacts_{run_id}.json"
                client.download_artifacts(run_id, artifact.path, local_path)
                
                with open(local_path, 'r') as f:
                    s3_info = json.load(f)
                
                # Clean up
                if os.path.exists(local_path):
                    os.remove(local_path)
                
                return s3_info
        
        return None
        
    except Exception as e:
        status(f"❌ Failed to get S3 artifacts from MLflow: {e}")
        return None

def load_model_with_s3_verification(model_name, alias="production"):

    try:
        # Load from MLflow
        model, model_info, s3_info = load_production_model_with_tracking(alias)
        
        if model is None:
            return None, None, None, "failed_to_load"
        
        # Verify S3 artifacts exist
        verification_status = "verified"
        missing_artifacts = []
        
        if s3_info and "s3_artifacts" in s3_info:
            for artifact_path in s3_info["s3_artifacts"]:
                if not artifact_exists_in_s3(artifact_path):
                    missing_artifacts.append(artifact_path)
                    verification_status = "missing_artifacts"
        
        if missing_artifacts:
            status(f"⚠️ Missing S3 artifacts: {missing_artifacts}")
        
        return model, model_info, s3_info, verification_status
        
    except Exception as e:
        status(f"❌ Error in model loading with S3 verification: {e}")
        return None, None, None, "error"

def artifact_exists_in_s3(s3_key):
    """
    Check if artifact exists in S3.
    
    Args:
        s3_key (str): S3 key to check
    
    Returns:
        bool: True if artifact exists
    """
    try:
        if not aws_available:
            return False
        
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return True
    except:
        return False

def sync_mlflow_s3_artifacts():
    """
    Synchronize MLflow and S3 artifacts to ensure consistency.
    
    Returns:
        dict: Synchronization report
    """
    try:
        client = MlflowClient()
        sync_report = {
            "total_models": 0,
            "synced_models": 0,
            "missing_s3": 0,
            "missing_mlflow": 0,
            "errors": []
        }
        
        # Get all registered models
        registered_models = client.search_registered_models()
        
        for model in registered_models:
            sync_report["total_models"] += 1
            
            try:
                # Get latest version
                versions = client.search_model_versions(f"name='{model.name}'")
                if not versions:
                    continue
                
                latest_version = versions[0]
                run_id = latest_version.run_id
                
                # Check S3 artifacts
                s3_info = get_s3_artifacts_from_mlflow(run_id)
                
                if s3_info and s3_info.get("artifacts"):
                    # Verify S3 artifacts exist
                    all_exist = True
                    for artifact_path in s3_info["artifacts"]:
                        if not artifact_exists_in_s3(artifact_path):
                            all_exist = False
                            break
                    
                    if all_exist:
                        sync_report["synced_models"] += 1
                    else:
                        sync_report["missing_s3"] += 1
                else:
                    sync_report["missing_mlflow"] += 1
                    
            except Exception as e:
                sync_report["errors"].append(f"Error syncing {model.name}: {e}")
        
        status(f"✅ S3-MLflow sync completed: {sync_report}")
        return sync_report
        
    except Exception as e:
        status(f"❌ Failed to sync S3-MLflow artifacts: {e}")
        return {"error": str(e)} 