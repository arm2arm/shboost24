#!/usr/bin/env bash
set -euo pipefail

# Start MLflow tracking server in background using local file store
# Adjust ARTIFACT_ROOT to a mounted volume for persistence
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/app/mlruns}
MFL_PORT=${MFL_PORT:-5123}

mkdir -p "$ARTIFACT_ROOT"

# Start mlflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root "$ARTIFACT_ROOT" --host 0.0.0.0 --port "$MFL_PORT" &
MLFLOW_PID=$!

echo "MLflow server started with PID $MLFLOW_PID"

# Run training script
python mlflowxgb.py "$@"

# Wait for mlflow server if training is interactive or long-lived
wait $MLFLOW_PID
