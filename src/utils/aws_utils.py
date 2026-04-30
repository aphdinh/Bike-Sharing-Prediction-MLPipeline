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
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        logging.error(f"Failed to download from S3: {e}")
        return False
    except Exception as e:
        logging.error(f"Failed to download from S3: {e}")
        return False

def list_s3_artifacts(prefix=""):
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]
    except Exception as e:
        logging.error(f"Failed to list S3 artifacts: {e}")
        return []

def load_model_from_s3(model_name, local_path=None):
    if local_path is None:
        local_path = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    
    s3_key = f"models/{model_name.lower().replace(' ', '_')}/model.pkl"
    
    if download_from_s3(s3_key, local_path):
        with open(local_path, "rb") as f:
            model = pickle.load(f)
        os.remove(local_path)
        return model
    return None

def load_scaler_from_s3(model_name, local_path=None):
    if local_path is None:
        local_path = f"{model_name.lower().replace(' ', '_')}_scaler.pkl"
    
    s3_key = f"models/{model_name.lower().replace(' ', '_')}/scaler.pkl"
    
    if download_from_s3(s3_key, local_path):
        with open(local_path, "rb") as f:
            scaler = pickle.load(f)
        os.remove(local_path)
        return scaler
    return None

def load_best_model_from_s3():
    if not download_from_s3("best_model_info.json", "temp_best_model_info.json"):
        return None, None, None
    
    with open("temp_best_model_info.json", "r") as f:
        best_model_info = json.load(f)
    os.remove("temp_best_model_info.json")
    
    model_name = best_model_info["best_model_name"]
    model = load_model_from_s3(model_name)
    scaler = load_scaler_from_s3(model_name)
    
    if model is not None:
        return model, scaler, best_model_info
    return None, None, None

def list_available_models_in_s3():
    available_models = {
        "best_model_info": None,
        "model_directories": [],
        "total_models": 0
    }
    
    if download_from_s3("best_model_info.json", "temp_best_model_info.json"):
        with open("temp_best_model_info.json", "r") as f:
            best_model_info = json.load(f)
        os.remove("temp_best_model_info.json")
        available_models["best_model_info"] = best_model_info
    
    model_dirs = list_s3_artifacts("models/")
    for model_dir in model_dirs:
        if model_dir.endswith("/"):
            model_name = model_dir.rstrip("/").split("/")[-1]
            model_files = list_s3_artifacts(model_dir)
            
            model_info = {
                "name": model_name,
                "path": model_dir,
                "files": model_files,
                "has_model": any(f.endswith("model.pkl") for f in model_files),
                "has_scaler": any(f.endswith("scaler.pkl") for f in model_files),
                "file_count": len(model_files)
            }
            
            available_models["model_directories"].append(model_info)
            available_models["total_models"] += 1
    
    return available_models

def save_results_to_s3(results_df, comparison_df=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    best_model_idx = results_df['test_r2'].idxmax()
    best_model_info = results_df.iloc[best_model_idx]
    best_model_name = best_model_info['model_name']
    
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
        'aws_region': AWS_REGION,
        's3_bucket': S3_BUCKET_NAME,
        'total_models_trained': len(results_df),
        'best_model': best_model_name,
        'best_r2_score': results_df['test_r2'].max(),
        'best_rmse_score': best_model_info['test_rmse'],
        'best_mae_score': best_model_info['test_mae'],
        'average_r2_score': results_df['test_r2'].mean(),
        'average_rmse_score': results_df['test_rmse'].mean(),
        'model_rankings': results_df.sort_values('test_r2', ascending=False)[['model_name', 'test_r2', 'test_rmse']].to_dict('records')
    }
    
    summary_file = f"training_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    upload_to_s3(summary_file, f"results/{summary_file}")
    os.remove(summary_file)
    
    best_model_marker = {
        'best_model_name': best_model_name,
        'best_model_path': f"models/{best_model_name.lower().replace(' ', '_')}/model.pkl",
        'best_scaler_path': f"models/{best_model_name.lower().replace(' ', '_')}/scaler.pkl",
        'performance_metrics': {
            'test_r2': best_model_info['test_r2'],
            'test_rmse': best_model_info['test_rmse'],
            'test_mae': best_model_info['test_mae'],
            'test_mape': best_model_info.get('test_mape', None)
        },
        'timestamp': timestamp,
        'training_run_id': best_model_info.get('run_id', 'unknown')
    }
    
    best_model_file = "best_model_info.json"
    with open(best_model_file, 'w') as f:
        json.dump(best_model_marker, f, indent=2)
    upload_to_s3(best_model_file, "best_model_info.json")
    os.remove(best_model_file)
    
    logging.info(f"Results saved to S3: s3://{S3_BUCKET_NAME}/results/")
    logging.info(f"Best model: {best_model_name} (R²: {best_model_info['test_r2']:.4f}, RMSE: {best_model_info['test_rmse']:.2f})")

def check_s3_model_completeness():
    if not download_from_s3("best_model_info.json", "temp_best_model_info.json"):
        return {"error": "No best_model_info.json found in S3"}
    
    with open("temp_best_model_info.json", "r") as f:
        best_model_info = json.load(f)
    os.remove("temp_best_model_info.json")
    
    model_name = best_model_info["best_model_name"]
    model_path = f"models/{model_name.lower().replace(' ', '_')}/model.pkl"
    scaler_path = f"models/{model_name.lower().replace(' ', '_')}/scaler.pkl"
    
    model_exists = False
    scaler_exists = False
    
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=model_path)
        model_exists = True
    except:
        pass
    
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=scaler_path)
        scaler_exists = True
    except:
        pass
    
    return {
        "model_name": model_name,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "model_exists": model_exists,
        "scaler_exists": scaler_exists,
        "can_load": model_exists,
        "performance_metrics": best_model_info.get("performance_metrics", {})
    } 