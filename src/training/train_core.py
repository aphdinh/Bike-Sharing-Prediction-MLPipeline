import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import logging
import os
import pickle
from datetime import datetime
import mlflow
from typing import Dict, List, Tuple, Optional, Any

from ..utils.aws_utils import aws_available, save_results_to_s3, upload_to_s3
from ..data.data_processing import load_data, feature_engineering, prepare_features
from ..models.models import get_models, hyperparameter_comparison
from ..utils.mlflow_utils import (
    setup_mlflow, log_metrics, calc_metrics, create_prediction_plots,
    register_best_model, get_best_model_info, compare_models_mlflow,
    register_model_with_s3_tracking
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def validate_environment_core() -> Dict[str, Any]:
    config = {
        'aws_region': os.getenv('AWS_REGION'),
        's3_bucket': os.getenv('S3_BUCKET_NAME'),
        'aws_available': aws_available,
        'mlflow_tracking_uri': os.getenv('MLFLOW_TRACKING_URI'),
        'mlflow_artifact_uri': os.getenv('MLFLOW_ARTIFACT_URI'),
    }
    return config

def setup_mlflow_core() -> str:
    try:
        mlflow.end_run()
    except:
        pass
    
    experiment_id = setup_mlflow()
    return experiment_id

def prepare_data_core() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                pd.Series, pd.Series, pd.Series]:
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
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5)
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=pd.cut(y_temp, bins=5)
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_single_model_core(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    scale_sensitive_models: set
) -> Dict[str, Any]:
    scaler = StandardScaler() if model_name in scale_sensitive_models else None
    result = evaluate_single_model(
        model, X_train, X_test, y_train, y_test, model_name, scaler=scaler
    )
    return result

def train_all_models_core(X_train, X_test, y_train, y_test):
    models = get_models()
    scale_sensitive = get_scale_sensitive_models()

    results = []
    for name, model in models.items():
        scaler = StandardScaler() if name in scale_sensitive else None
        result = evaluate_single_model(
            model, X_train, X_test, y_train, y_test, name, 
            scaler=scaler
        )
        results.append(result)
    
    return pd.DataFrame(results)

def perform_hyperparameter_tuning_core(
    best_model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Optional[Dict[str, Any]]:
    if best_model_name not in ['LightGBM', 'XGBoost', 'Random Forest']:
        logging.info(f"Hyperparameter tuning not available for {best_model_name}")
        return None
    
    logging.info(f"Starting hyperparameter optimization for {best_model_name}")
    
    try:
        best_tuned_model = hyperparameter_comparison(
            X_train, y_train, X_val, y_val, best_model_name
        )
        
        if best_tuned_model is not None:
            logging.info(f"Evaluating optimized {best_model_name}...")
            final_result = evaluate_single_model(
                best_tuned_model, X_train, X_test, y_train, y_test,
                f"Hyperopt_Tuned_{best_model_name}", scaler=None
            )
            
            logging.info(f"Optimization complete:")
            logging.info(f"  Final test RMSE: {final_result['test_rmse']:.4f}")
            logging.info(f"  Final test R²: {final_result['test_r2']:.4f}")
            
            return final_result
        
    except Exception as e:
        logging.error(f"Hyperparameter optimization failed: {str(e)}")
        return None
    
    return None

def register_and_save_best_model_core(
    results_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    registration_result = register_best_model(results_df)
    registered_model_name = registration_result["model_name"]
    comparison_df = compare_models_mlflow("seoul-bike-sharing")
    save_results_to_s3(results_df, comparison_df)

    return results_df, comparison_df, registered_model_name

def create_training_report_core(
    config: Dict[str, Any],
    results_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    best_model_name: str,
    tuning_result: Optional[Dict[str, Any]]
) -> str:
    report = f"""# ML Training Pipeline Report

## Environment Configuration
- AWS Region: {config.get('aws_region', 'Not set')}
- S3 Bucket: {config.get('s3_bucket', 'Not set')}
- AWS Available: {config.get('aws_available', False)}
- MLflow Tracking URI: {config.get('mlflow_tracking_uri', 'Not set')}
- MLflow Artifact URI: {config.get('mlflow_artifact_uri', 'Not set')}

## Model Performance Summary
- Best Model: {best_model_name}
- Total Models Evaluated: {len(results_df)}
- Average R² Score: {results_df['test_r2'].mean():.4f}
- Best R² Score: {results_df['test_r2'].max():.4f}

## Top 5 Models by R² Score
{results_df.nlargest(5, 'test_r2')[['model_name', 'test_r2', 'test_rmse', 'test_mae']].to_string(index=False)}

## Hyperparameter Tuning Results
"""
    
    if tuning_result:
        report += f"""
- Tuning Method: {tuning_result.get('method', 'Unknown')}
- Best Parameters: {tuning_result.get('best_params', 'Not available')}
- Tuning Score: {tuning_result.get('best_score', 'Not available')}
"""
    else:
        report += "No hyperparameter tuning performed.\n"
    
    report += f"""
## Model Comparison Details
{comparison_df.to_string(index=False)}

---
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report

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
    
    y_pred_train = model.predict(X_train_p)
    y_pred_test = model.predict(X_test_p)
    
    return X_train_p, X_test_p, y_pred_train, y_pred_test

def save_model_to_s3(model, model_name, scaler=None):
    slug = model_name.lower().replace(' ', '_').replace('-', '_')
    model_file = f"model_{slug}.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    upload_to_s3(model_file, f"models/{model_name.lower().replace(' ', '_')}/model.pkl")
    os.remove(model_file)

    if scaler is not None:
        scaler_file = f"scaler_{slug}.pkl"
        with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)
        upload_to_s3(scaler_file, f"models/{model_name.lower().replace(' ', '_')}/scaler.pkl")
        os.remove(scaler_file)

def handle_feature_importance(model, X_train, model_name):
    if not hasattr(model, 'feature_importances_'):
        return
    
    feature_names = X_train.columns.tolist()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_csv = "feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)
    mlflow.log_artifact(importance_csv, artifact_path="analysis")
    
    upload_to_s3(importance_csv, f"models/{model_name.lower().replace(' ', '_')}/feature_importance.csv")
    os.remove(importance_csv)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(20), y='feature', x='importance', palette='viridis')
    plt.title(f'Top 20 Feature Importances - {model_name}')
    plt.tight_layout()

    importance_plot = f"feature_importance_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(importance_plot, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(importance_plot, artifact_path="plots")
    upload_to_s3(importance_plot, f"models/{model_name.lower().replace(' ', '_')}/feature_importance.png")
    os.remove(importance_plot)
    plt.close()
    

def create_model_results(model_name, train_metrics, test_metrics, overfit, run_id): 
    def safe_value(value):
        if pd.isna(value) or np.isinf(value):
            return 0.0
        return float(value)
    
    return {
        'model_name': model_name,
        'train_r2': safe_value(train_metrics['r2']),
        'train_rmse': safe_value(train_metrics['rmse']),
        'train_mae': safe_value(train_metrics['mae']),
        'test_r2': safe_value(test_metrics['r2']),
        'test_rmse': safe_value(test_metrics['rmse']),
        'test_mae': safe_value(test_metrics['mae']),
        'overfit_score': safe_value(overfit),
        'run_id': run_id
    }

def evaluate_single_model(model, X_train, X_test, y_train, y_test, model_name, scaler=None, log_model=True):
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        log_aws_tags()
        log_model_parameters(model, model_name, X_train, X_test, scaler)
        
        X_train_p, X_test_p, y_pred_train, y_pred_test = train_and_predict(
            model, X_train, X_test, y_train, scaler
        )
        
        train_metrics = calc_metrics(y_train, y_pred_train)
        test_metrics = calc_metrics(y_test, y_pred_test)
        
        # Safe overfitting calculation
        try:
            overfit = train_metrics['rmse'] - test_metrics['rmse']
            if not np.isfinite(overfit):
                overfit = 0.0
        except:
            overfit = 0.0
        
        log_metrics({
            'train_rmse': train_metrics['rmse'],
            'test_rmse': test_metrics['rmse'],
            'train_mae': train_metrics['mae'],
            'test_mae': test_metrics['mae'],
            'train_r2': train_metrics['r2'],
            'test_r2': test_metrics['r2'],
            'train_mape': train_metrics['mape'],
            'test_mape': test_metrics['mape'],
            'overfitting_score': overfit
        })
        
        create_prediction_plots(y_test, y_pred_test, model_name)
        
        if log_model:
            # Use new consistent registration approach
            from ..utils.mlflow_utils import register_model_with_s3_tracking
            
            # Prepare additional artifacts
            additional_artifacts = {}
            
            # Add feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance_file = f"feature_importance_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"
                if os.path.exists(feature_importance_file):
                    additional_artifacts["feature_importance.png"] = feature_importance_file
            
            # Register model with S3 tracking
            registration_info = register_model_with_s3_tracking(
                model, model_name, mlflow.active_run().info.run_id, 
                scaler, additional_artifacts
            )
            
            if registration_info:
                logging.info(f"✅ Model {model_name} registered with S3 tracking")
                logging.info(f"   S3 Artifacts: {len(registration_info['s3_artifacts'])} files")
            else:
                logging.warning(f"⚠️ Failed to register model {model_name} with S3 tracking")
        
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

def main_training_pipeline() -> Dict[str, Any]:
    """Main training pipeline without orchestration dependencies."""
    config = validate_environment_core()
    experiment_id = setup_mlflow_core()
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_core()
    
    results_df = train_all_models_core(X_train, X_test, y_train, y_test)
    best_model_name = results_df.loc[results_df['test_r2'].idxmax(), 'model_name']

    tuning_result = perform_hyperparameter_tuning_core(best_model_name, X_train, y_train, X_val, y_val, X_test, y_test)

    updated_results_df, comparison_df, registered_model_name = register_and_save_best_model_core(
        results_df
    )

    best_r2 = updated_results_df['test_r2'].max()
    best_rmse = updated_results_df.loc[updated_results_df['test_r2'].idxmax(), 'test_rmse']
    
    print(f"Best Model: {best_model_name} (R² = {best_r2:.4f}, RMSE = {best_rmse:.2f})")
    
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