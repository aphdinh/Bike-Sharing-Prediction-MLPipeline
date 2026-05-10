import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import r2_score
import warnings
import logging
import os
import requests
from datetime import datetime
import mlflow
from typing import Dict, List, Tuple, Optional, Any

from ..utils.aws_utils import aws_available, save_results_to_s3
from ..data.data_processing import load_data, feature_engineering, prepare_features
from ..models.models import get_models, hyperparameter_comparison
from ..utils.mlflow_utils import (
    setup_mlflow, log_metrics, calc_metrics, create_prediction_plots,
    register_best_model, compare_models_mlflow, save_model_to_s3_with_tracking
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def prepare_data_core():
    df = load_data()
    if df.empty:
        raise ValueError("Loaded dataset is empty")

    df_features = feature_engineering(df)

    os.makedirs('data', exist_ok=True)
    reference_sample = df_features.drop(columns=['date', 'day', 'day_name'], errors='ignore')
    reference_sample.sample(min(200, len(reference_sample)), random_state=42).to_csv(
        'data/reference_data.csv', index=False
    )
    logging.info("Saved reference data for monitoring")

    X, y, feature_names = prepare_features(df_features)

    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))

    # Last fold: held-out test set at the end of the year for final evaluation
    train_val_idx, test_idx = splits[-1]
    X_tv, y_tv = X.iloc[train_val_idx], y.iloc[train_val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Split train_val into train / val (last 20% of that window)
    val_size = int(len(X_tv) * 0.2)
    X_train, y_train = X_tv.iloc[:-val_size], y_tv.iloc[:-val_size]
    X_val, y_val = X_tv.iloc[-val_size:], y_tv.iloc[-val_size:]

    # Return full X, y alongside splits so CV can use all folds
    return X, y, X_train, X_val, X_test, y_train, y_val, y_test


def perform_hyperparameter_tuning_core(
    best_model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Optional[Dict[str, Any]], Any]:
    if best_model_name not in ['LightGBM', 'XGBoost', 'Random Forest']:
        logging.info(f"Hyperparameter tuning not available for {best_model_name}")
        return None, None

    logging.info(f"Starting hyperparameter optimization for {best_model_name}")
    try:
        tuned_model = hyperparameter_comparison(X_train, y_train, X_val, y_val, best_model_name)
        if tuned_model is not None:
            result = evaluate_single_model(
                tuned_model, X, y, X_train, X_test, y_train, y_test,
                f"Hyperopt_Tuned_{best_model_name}", scaler=None
            )
            logging.info(f"Tuning complete — RMSE: {result['test_rmse']:.2f}, R²: {result['test_r2']:.4f}")
            return result, tuned_model
    except Exception as e:
        logging.error(f"Hyperparameter optimization failed: {e}")
    return None, None


def register_and_save_best_model_core(
    results_df: pd.DataFrame,
    best_model: Any,
    best_scaler: Any
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    registration_result = register_best_model(results_df)
    registered_model_name = registration_result["model_name"]
    comparison_df = compare_models_mlflow("seoul-bike-sharing")

    best_name = results_df.loc[results_df['test_r2'].idxmax(), 'model_name']
    if best_model is not None:
        save_model_to_s3_with_tracking(best_model, best_name, best_scaler)
        logging.info(f"Best model '{best_name}' saved to S3")

    save_results_to_s3(results_df, comparison_df)
    return results_df, comparison_df, registered_model_name


def log_model_parameters(model, model_name, X_train, X_test, scaler):
    if hasattr(model, 'get_params'):
        for param, value in model.get_params().items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(param, value)
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("model_type", model_name)
    mlflow.log_param("scaled", scaler is not None)


def log_aws_tags():
    mlflow.set_tag("developer", "Phuong")
    mlflow.set_tag("aws_region", os.getenv('AWS_REGION'))
    mlflow.set_tag("s3_bucket", os.getenv('S3_BUCKET_NAME'))


def train_and_predict(model, X_train, X_test, y_train, scaler):
    X_train_p = scaler.fit_transform(X_train) if scaler else X_train
    X_test_p = scaler.transform(X_test) if scaler else X_test
    model.fit(X_train_p, y_train)
    return X_train_p, X_test_p, model.predict(X_train_p), model.predict(X_test_p)


def handle_feature_importance(model, X_train, model_name):
    if not hasattr(model, 'feature_importances_'):
        return

    importance_df = pd.DataFrame({
        'feature': X_train.columns.tolist(),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_csv = "feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)
    mlflow.log_artifact(importance_csv, artifact_path="analysis")
    os.remove(importance_csv)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(20), y='feature', x='importance', palette='viridis')
    plt.title(f'Top 20 Feature Importances - {model_name}')
    plt.tight_layout()
    importance_plot = f"feature_importance_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(importance_plot, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(importance_plot, artifact_path="plots")
    os.remove(importance_plot)
    plt.close()


def create_model_results(model_name, train_metrics, test_metrics, overfit, run_id):
    def safe(v):
        return 0.0 if pd.isna(v) or np.isinf(v) else float(v)
    return {
        'model_name': model_name,
        'train_r2': safe(train_metrics['r2']),
        'train_rmse': safe(train_metrics['rmse']),
        'train_mae': safe(train_metrics['mae']),
        'test_r2': safe(test_metrics['r2']),
        'test_rmse': safe(test_metrics['rmse']),
        'test_mae': safe(test_metrics['mae']),
        'overfit_score': safe(overfit),
        'run_id': run_id
    }


def cross_validate_model(model, X, y, scaler, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
        y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
        m = clone(model)
        if scaler:
            sc = StandardScaler()
            X_cv_train = sc.fit_transform(X_cv_train)
            X_cv_val = sc.transform(X_cv_val)
        m.fit(X_cv_train, y_cv_train)
        scores.append(r2_score(y_cv_val, m.predict(X_cv_val)))
    return float(np.mean(scores)), float(np.std(scores))


def evaluate_single_model(model, X, y, X_train, X_test, y_train, y_test, model_name, scaler=None):
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        log_aws_tags()
        log_model_parameters(model, model_name, X_train, X_test, scaler)

        cv_mean, cv_std = cross_validate_model(model, X, y, scaler)

        X_train_p, X_test_p, y_pred_train, y_pred_test = train_and_predict(
            model, X_train, X_test, y_train, scaler
        )

        train_metrics = calc_metrics(y_train, y_pred_train)
        test_metrics = calc_metrics(y_test, y_pred_test)

        try:
            overfit = train_metrics['rmse'] - test_metrics['rmse']
            if not np.isfinite(overfit):
                overfit = 0.0
        except (TypeError, ValueError):
            overfit = 0.0

        log_metrics({
            'cv_mean_r2': cv_mean, 'cv_std_r2': cv_std,
            'train_rmse': train_metrics['rmse'], 'test_rmse': test_metrics['rmse'],
            'train_mae': train_metrics['mae'], 'test_mae': test_metrics['mae'],
            'train_r2': train_metrics['r2'], 'test_r2': test_metrics['r2'],
            'train_mape': train_metrics['mape'], 'test_mape': test_metrics['mape'],
            'overfitting_score': overfit
        })

        create_prediction_plots(y_test, y_pred_test, model_name)
        handle_feature_importance(model, X_train, model_name)

        return create_model_results(
            model_name, train_metrics, test_metrics, overfit,
            mlflow.active_run().info.run_id
        )


def get_scale_sensitive_models():
    return {
        'Linear Regression', 'Ridge Regression', 'Lasso Regression',
        'Elastic Net', 'K-Nearest Neighbors', 'Support Vector Regression'
    }


def train_all_models_core(X, y, X_train, X_test, y_train, y_test):
    models = get_models()
    scale_sensitive = get_scale_sensitive_models()
    results = []
    trained_models = {}
    for name, model in models.items():
        scaler = StandardScaler() if name in scale_sensitive else None
        result = evaluate_single_model(model, X, y, X_train, X_test, y_train, y_test, name, scaler=scaler)
        results.append(result)
        trained_models[name] = (model, scaler)
    return pd.DataFrame(results), trained_models


def main_training_pipeline() -> Dict[str, Any]:
    try:
        mlflow.end_run()
    except mlflow.exceptions.MlflowException:
        pass
    experiment_id = setup_mlflow()

    X, y, X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_core()

    results_df, trained_models = train_all_models_core(X, y, X_train, X_test, y_train, y_test)
    best_model_name = results_df.loc[results_df['test_r2'].idxmax(), 'model_name']

    tuning_result, tuned_model = perform_hyperparameter_tuning_core(
        best_model_name, X, y, X_train, y_train, X_val, y_val, X_test, y_test
    )
    if tuning_result:
        results_df = pd.concat([results_df, pd.DataFrame([tuning_result])], ignore_index=True)
        logging.info(f"Tuned model added — best R² now: {results_df['test_r2'].max():.4f}")

    best_row = results_df.loc[results_df['test_r2'].idxmax()]
    best_model_name = best_row['model_name']

    if best_model_name.startswith('Hyperopt_Tuned_') and tuned_model is not None:
        best_model, best_scaler = tuned_model, None
    else:
        best_model, best_scaler = trained_models.get(best_model_name, (None, None))

    updated_results_df, comparison_df, registered_model_name = register_and_save_best_model_core(
        results_df, best_model, best_scaler
    )

    best_r2 = best_row['test_r2']
    best_rmse = best_row['test_rmse']

    print(f"\n{'='*50}")
    print(f"Best Model: {best_model_name}")
    print(f"  R²   = {best_r2:.4f}")
    print(f"  RMSE = {best_rmse:.2f}")
    print(f"{'='*50}\n")

    try:
        r = requests.post("http://localhost:8000/reload-model", timeout=10)
        logging.info(f"API reloaded with new model: {r.json()}")
    except Exception:
        logging.info("API not running — skipping hot reload")

    return {
        'status': 'success',
        'best_model': best_model_name,
        'best_r2_score': best_r2,
        'best_rmse': best_rmse,
        'total_models_trained': len(updated_results_df),
        'registered_model_name': registered_model_name,
        'mlflow_experiment_id': experiment_id,
        'execution_time': datetime.now().isoformat()
    }


if __name__ == "__main__":
    result = main_training_pipeline()
    print(f"Training completed: {result}")
