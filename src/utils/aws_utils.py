import boto3
import logging
import os
import json
import pickle
from botocore.exceptions import ClientError, NoCredentialsError
from datetime import datetime

AWS_REGION = os.getenv('AWS_REGION')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

try:
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    aws_available = True
    logging.info(f"AWS S3 initialized for bucket: {S3_BUCKET_NAME}")
except (NoCredentialsError, ClientError) as e:
    aws_available = False
    logging.warning(f"AWS S3 not available: {e}")


def upload_to_s3(local_file_path, s3_key):
    try:
        s3_client.upload_file(local_file_path, S3_BUCKET_NAME, s3_key)
        return True
    except Exception as e:
        logging.error(f"Failed to upload to S3: {e}")
        return False


def download_from_s3(s3_key, local_file_path):
    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_file_path)
        return True
    except Exception as e:
        if isinstance(e, ClientError) and e.response['Error']['Code'] == '404':
            return False
        logging.error(f"Failed to download from S3: {e}")
        return False


def _load_pickle_from_s3(s3_key, local_path):
    if not download_from_s3(s3_key, local_path):
        return None
    with open(local_path, "rb") as f:
        obj = pickle.load(f)
    os.remove(local_path)
    return obj


def _s3_key_exists(s3_key):
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return True
    except Exception:
        return False


def load_best_model_from_s3():
    if not download_from_s3("best_model_info.json", "temp_best_model_info.json"):
        return None, None, None

    with open("temp_best_model_info.json", "r") as f:
        best_model_info = json.load(f)
    os.remove("temp_best_model_info.json")

    slug = best_model_info["best_model_name"].lower().replace(' ', '_')
    model = _load_pickle_from_s3(f"models/{slug}/model.pkl", f"{slug}_model.pkl")
    scaler = _load_pickle_from_s3(f"models/{slug}/scaler.pkl", f"{slug}_scaler.pkl")

    if model is not None:
        return model, scaler, best_model_info
    return None, None, None


def save_results_to_s3(results_df, comparison_df=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best = results_df.iloc[results_df['test_r2'].idxmax()]
    best_name = best['model_name']

    results_file = f"training_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    upload_to_s3(results_file, f"results/{results_file}")
    os.remove(results_file)

    if comparison_df is not None:
        comparison_file = f"model_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        upload_to_s3(comparison_file, f"results/{comparison_file}")
        os.remove(comparison_file)

    summary = {
        'timestamp': timestamp,
        'best_model': best_name,
        'best_r2_score': results_df['test_r2'].max(),
        'best_rmse_score': best['test_rmse'],
        'best_mae_score': best['test_mae'],
        'average_r2_score': results_df['test_r2'].mean(),
        'model_rankings': results_df.sort_values('test_r2', ascending=False)[
            ['model_name', 'test_r2', 'test_rmse']].to_dict('records')
    }
    summary_file = f"training_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    upload_to_s3(summary_file, f"results/{summary_file}")
    os.remove(summary_file)

    slug = best_name.lower().replace(' ', '_')
    best_model_marker = {
        'best_model_name': best_name,
        'best_model_path': f"models/{slug}/model.pkl",
        'best_scaler_path': f"models/{slug}/scaler.pkl",
        'performance_metrics': {
            'test_r2': best['test_r2'],
            'test_rmse': best['test_rmse'],
            'test_mae': best['test_mae'],
        },
        'timestamp': timestamp,
        'training_run_id': best.get('run_id', 'unknown')
    }
    with open("best_model_info.json", 'w') as f:
        json.dump(best_model_marker, f, indent=2)
    upload_to_s3("best_model_info.json", "best_model_info.json")
    os.remove("best_model_info.json")

    logging.info(f"Results saved to S3. Best model: {best_name} (R²: {best['test_r2']:.4f})")


def check_s3_model_completeness():
    if not download_from_s3("best_model_info.json", "temp_best_model_info.json"):
        return {"error": "No best_model_info.json found in S3"}

    with open("temp_best_model_info.json", "r") as f:
        best_model_info = json.load(f)
    os.remove("temp_best_model_info.json")

    slug = best_model_info["best_model_name"].lower().replace(' ', '_')
    model_path = f"models/{slug}/model.pkl"
    scaler_path = f"models/{slug}/scaler.pkl"
    model_exists = _s3_key_exists(model_path)
    scaler_exists = _s3_key_exists(scaler_path)

    return {
        "model_name": best_model_info["best_model_name"],
        "model_path": model_path,
        "scaler_path": scaler_path,
        "model_exists": model_exists,
        "scaler_exists": scaler_exists,
        "can_load": model_exists,
        "performance_metrics": best_model_info.get("performance_metrics", {})
    }
