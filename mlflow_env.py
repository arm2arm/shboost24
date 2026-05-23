import os
import mlflow

mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
else:
    mfl_port = os.environ.get('MFL_PORT', '5123')
    mlflow.set_tracking_uri(f'http://127.0.0.1:{mfl_port}')
