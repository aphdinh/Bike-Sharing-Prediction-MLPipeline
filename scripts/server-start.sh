echo 'export AWS_REGION=eu-north-1' >> ~/.zshrc
echo 'export S3_BUCKET_NAME=seoul-bike-sharing-aphdinh-2' >> ~/.zshrc
echo 'export MLFLOW_TRACKING_URI=sqlite:///mlflow.db' >> ~/.zshrc
echo 'export MLFLOW_ARTIFACT_URI=s3://seoul-bike-sharing-aphdinh-2/mlflow-artifacts/' >> ~/.zshrc
source ~/.zshrc
