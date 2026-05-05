import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import logging
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.tracking import MlflowClient

from .aws_utils import upload_to_s3, AWS_REGION, S3_BUCKET_NAME, aws_available

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', "seoul-bike-sharing")
MLFLOW_ARTIFACT_URI = os.getenv('MLFLOW_ARTIFACT_URI')


def _create_experiment(name):
    if MLFLOW_ARTIFACT_URI:
        return mlflow.create_experiment(name, artifact_location=MLFLOW_ARTIFACT_URI)
    return mlflow.create_experiment(name)


def setup_mlflow():
    global EXPERIMENT_NAME
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = _create_experiment(EXPERIMENT_NAME)
        elif experiment.lifecycle_stage == "deleted":
            try:
                MlflowClient().restore_experiment(experiment.experiment_id)
                experiment_id = experiment.experiment_id
            except Exception:
                experiment_id = _create_experiment(EXPERIMENT_NAME)
        else:
            experiment_id = experiment.experiment_id
    except mlflow.exceptions.MlflowException:
        try:
            experiment_id = _create_experiment(EXPERIMENT_NAME)
        except Exception:
            fallback = f"{EXPERIMENT_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            experiment_id = _create_experiment(fallback)
            EXPERIMENT_NAME = fallback

    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id


def log_metrics(metrics):
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v if (np.isfinite(v) and not np.isnan(v)) else 0.0)


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100 if non_zero.sum() > 0 else 0.0

    return {
        'rmse': float(rmse) if np.isfinite(rmse) else 0.0,
        'mae': float(mae) if np.isfinite(mae) else 0.0,
        'r2': float(r2) if np.isfinite(r2) else 0.0,
        'mape': float(mape) if np.isfinite(mape) else 0.0,
    }


def create_prediction_plots(y_test, y_pred, model_name):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    slug = model_name.lower().replace(' ', '_').replace('-', '_')

    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue', s=10)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set(xlabel='Actual', ylabel='Predicted', title=f'{model_name}: Predicted vs Actual')
    r2 = r2_score(y_test, y_pred) if np.isfinite(r2_score(y_test, y_pred)) else 0.0
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green', s=10)
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    axes[0, 1].set(xlabel='Predicted', ylabel='Residuals', title=f'{model_name}: Residuals')

    axes[1, 0].hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--')
    axes[1, 0].set(xlabel='Residuals', ylabel='Frequency', title='Residual Distribution')

    bins = pd.cut(y_test, bins=10)
    error_by_range = pd.DataFrame({'range': bins, 'error': np.abs(residuals)}).groupby('range')['error'].mean()
    axes[1, 1].bar(range(len(error_by_range)), error_by_range.values, alpha=0.7, color='orange')
    axes[1, 1].set(xlabel='Actual Value Ranges', ylabel='MAE', title='Error by Value Range')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plot_file = f"prediction_analysis_{slug}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(plot_file, artifact_path="plots")
    upload_to_s3(plot_file, f"models/{model_name.lower().replace(' ', '_')}/prediction_analysis.png")
    os.remove(plot_file)
    plt.close()


def _get_best_tuned_model(client, best_r2, best_run_id, best_name):
    tuned_models = [m for m in client.search_registered_models() if "tuned" in m.name.lower()]
    for model in tuned_models:
        for v in client.search_model_versions(f"name='{model.name}'"):
            try:
                r2 = client.get_run(v.run_id).data.metrics.get('test_r2') or 0
                if r2 > best_r2:
                    best_r2, best_run_id, best_name = r2, v.run_id, model.name
            except Exception:
                continue
    return best_r2, best_run_id, best_name


def register_best_model(results_df):
    client = MlflowClient()
    best = results_df.iloc[results_df['test_r2'].idxmax()]
    best_r2, best_run_id, best_name = best['test_r2'], best['run_id'], best['model_name']

    try:
        best_r2, best_run_id, best_name = _get_best_tuned_model(client, best_r2, best_run_id, best_name)
    except Exception:
        pass

    prod_name = "seoul_bike_production_model"
    try:
        client.create_registered_model(prod_name)
    except mlflow.exceptions.MlflowException:
        pass

    version = client.create_model_version(
        name=prod_name,
        source=f"runs:/{best_run_id}/model",
        run_id=best_run_id,
        description=f"Best: {best_name} (R²: {best_r2:.4f})"
    )
    client.transition_model_version_stage(name=prod_name, version=version.version,
                                          stage="Production", archive_existing_versions=True)

    try:
        metrics = client.get_run(best_run_id).data.metrics
        rmse, mae = metrics.get('test_rmse', 'N/A'), metrics.get('test_mae', 'N/A')
    except Exception:
        rmse = mae = 'N/A'

    client.update_model_version(name=prod_name, version=version.version,
                                description=f"Best: {best_name} | R²: {best_r2:.4f} | RMSE: {rmse} | MAE: {mae}")

    tags = {
        "best_model_name": best_name,
        "test_r2_score": str(best_r2),
        "test_rmse_score": str(rmse),
        "test_mae_score": str(mae),
        "is_hyperparameter_tuned": str("tuned" in best_name.lower()),
        "timestamp": datetime.now().isoformat()
    }
    for k, v in tags.items():
        client.set_model_version_tag(prod_name, version.version, k, v)

    for alias in ["production", "champion", "latest", "best"]:
        try:
            client.set_registered_model_alias(prod_name, alias, version.version)
        except mlflow.exceptions.MlflowException:
            pass

    return {"model_name": prod_name, "version": version.version, "run_id": best_run_id,
            "best_model_name": best_name, "test_r2_score": best_r2,
            "test_rmse_score": rmse, "test_mae_score": mae}


def get_model_info_by_alias(model_name, alias):
    try:
        v = MlflowClient().get_model_version_by_alias(model_name, alias)
        return {"model_name": model_name, "version": v.version, "stage": v.current_stage,
                "description": v.description, "tags": v.tags, "run_id": v.run_id}
    except Exception:
        return None


def get_best_model_info():
    client = MlflowClient()
    model_name = "seoul_bike_production_model"

    info = get_model_info_by_alias(model_name, "production")
    if info:
        return {
            "mlflow_info": {k: info[k] for k in ["model_name", "version", "stage", "description", "run_id"]},
            "model_metadata": {k: info["tags"].get(k) for k in
                               ["best_model_name", "test_r2_score", "test_rmse_score",
                                "test_mae_score", "is_hyperparameter_tuned", "timestamp"]}
        }

    try:
        v = client.get_latest_versions(model_name, stages=["Production"])[0]
        return {
            "mlflow_info": {"model_name": model_name, "version": v.version, "stage": "Production",
                            "description": v.description, "run_id": v.run_id},
            "model_metadata": {k: v.tags.get(k) for k in
                               ["best_model_name", "test_r2_score", "test_rmse_score",
                                "test_mae_score", "is_hyperparameter_tuned", "timestamp"]}
        }
    except IndexError:
        return None


def compare_models_mlflow(experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return

    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                               order_by=["metrics.test_r2 DESC"])
    data = [
        {'run_id': r.info.run_id, 'model_name': r.data.params.get('model_type', 'Unknown'),
         'test_r2': r.data.metrics.get('test_r2', 0), 'test_rmse': r.data.metrics.get('test_rmse', float('inf')),
         'test_mae': r.data.metrics.get('test_mae', float('inf')),
         'overfitting_score': r.data.metrics.get('overfitting_score', 0)}
        for r in runs if 'test_r2' in r.data.metrics
    ]
    return pd.DataFrame(data)


def load_production_model_with_tracking(alias="production"):
    client = MlflowClient()
    model_name = "seoul_bike_production_model"

    try:
        v = client.get_model_version_by_alias(model_name, alias)
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{alias}")
        model_info = {"model_name": model_name, "version": v.version, "stage": v.current_stage,
                      "description": v.description, "run_id": v.run_id, "alias": alias}
        return model, model_info
    except Exception:
        pass

    try:
        v = client.get_latest_versions(model_name, stages=["Production"])[0]
        model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
        model_info = {"model_name": model_name, "version": v.version, "stage": "Production",
                      "description": v.description, "run_id": v.run_id}
        return model, model_info
    except (IndexError, Exception):
        return None, None


def save_model_to_s3_with_tracking(model, model_name, scaler=None):
    """Serialize model (and optionally scaler) as pickle files and upload to S3.
    Returns a dict of S3 keys for everything that was saved."""
    s3_artifacts = {}
    slug = model_name.lower().replace(' ', '_').replace('-', '_')
    prefix = model_name.lower().replace(' ', '_')

    model_file = f"model_{slug}.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    s3_key = f"models/{prefix}/model.pkl"
    if upload_to_s3(model_file, s3_key):
        s3_artifacts["model_path"] = s3_key
    os.remove(model_file)

    if scaler is not None:
        scaler_file = f"scaler_{slug}.pkl"
        with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)
        scaler_key = f"models/{prefix}/scaler.pkl"
        if upload_to_s3(scaler_file, scaler_key):
            s3_artifacts["scaler_path"] = scaler_key
        os.remove(scaler_file)

    s3_artifacts.update({"timestamp": datetime.now().isoformat(),
                          "model_name": model_name, "model_type": type(model).__name__})
    return s3_artifacts


def register_model_with_s3_tracking(model, model_name, run_id, scaler=None, additional_artifacts=None):
    """Save model to S3 and register a versioned entry in the MLflow model registry.
    MLflow version is tagged with its S3 path so the two stores stay linked."""
    try:
        s3_artifacts = save_model_to_s3_with_tracking(model, model_name, scaler)

        client = MlflowClient()
        registered_name = f"seoul_bike_{model_name.lower().replace(' ', '_')}"
        try:
            client.create_registered_model(registered_name)
        except mlflow.exceptions.MlflowException:
            pass

        version = client.create_model_version(name=registered_name, source=f"runs:/{run_id}/model",
                                               run_id=run_id, description=f"{model_name} - S3 tracked")
        tags = {"s3_bucket": S3_BUCKET_NAME, "s3_model_path": s3_artifacts.get("model_path", ""),
                "registration_timestamp": datetime.now().isoformat()}
        for k, v in tags.items():
            client.set_model_version_tag(registered_name, version.version, k, v)

        if additional_artifacts:
            for name, path in additional_artifacts.items():
                key = f"models/{model_name.lower().replace(' ', '_')}/{name}"
                if upload_to_s3(path, key):
                    s3_artifacts[name] = key

        return {"model_name": registered_name, "version": version.version,
                "run_id": run_id, "s3_artifacts": s3_artifacts}
    except Exception as e:
        logging.warning(f"Failed to register model with S3 tracking: {e}")
        return None
