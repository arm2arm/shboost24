Automl MLflow project

Dev Container (Python 3.10 + Docker-in-Docker)

This repository now includes a VS Code Dev Container setup in `.devcontainer/`.
It provides:

- Python 3.10.x development runtime
- Project dependencies preinstalled from `requirements.txt`
- Docker-in-Docker support so you can run `docker build` / `docker compose build` from inside the container

How to use

1) Install prerequisites on your host:
   - Docker Desktop (or equivalent Docker Engine)
   - VS Code extension: Dev Containers

2) Open this repository in VS Code.

3) Run: "Dev Containers: Reopen in Container".

4) Verify inside the dev container terminal:
   - `python --version` (should report 3.10.x)
   - `docker version`
   - `docker build -t automl-mlflow:dev .`

Notes

- The project runtime image in `Dockerfile` is still independent from the dev container setup.
- The dev container uses a persistent Docker data volume (`/var/lib/docker`) so image layers survive container rebuilds.

Quick start (single-container, development)

1) Build the Docker image (run on a machine with Docker):

   cd /lustre/arm2arm/Projects/AutoML/mlflow
   docker build -t automl-mlflow:latest .

2) Start the container (recommended: mount a host directory for persisted MLflow runs):

   # map host /data/mlruns -> container /data/mlruns and expose MLflow UI on host:5123
   docker run --rm -it \
     -p 5123:5123 \
     -v /data/mlruns:/data/mlruns \
     -e ARTIFACT_ROOT=/data/mlruns \
     automl-mlflow:latest

   The entrypoint will start a local MLflow server inside the container (sqlite backend) and
   then run the training script. The MLflow UI will be available at http://localhost:5123 on the host.

Environment variables

- ARTIFACT_ROOT: where MLflow stores artifacts inside the container. Default: /data/mlruns (recommended to mount host /data/mlruns).
- MFL_PORT: MLflow server port inside container (default 5123). The container exposes this port to the host.
- MLFLOW_TRACKING_URI: if set, the training script will use this tracking URI instead of the local MLflow server.
  Example: -e MLFLOW_TRACKING_URI=http://mlflow-server.example:5000

Run training only (use an external MLflow server)

If you already run MLflow elsewhere and do not want the container to start a local server, set MLFLOW_TRACKING_URI to
point at your server and run the trainer as the container command:

   docker run --rm -it \
     -v /data/mlruns:/data/mlruns \
     -e MLFLOW_TRACKING_URI=http://mlflow.example:5000 \
     automl-mlflow:latest \
     python src/mlflowxgb.py --data /data/your_training.parq

GPU note

The default image is CPU-only. To enable GPU training with XGBoost (tree_method=gpu_hist) you need:
- Host with NVIDIA drivers and nvidia-container-toolkit installed
- A CUDA-enabled Docker base image and a GPU-capable xgboost build (not the default CPU wheel)

We can provide a GPU Dockerfile on request.

Troubleshooting

- If the MLflow UI does not appear, check the container logs and ensure the ARTIFACT_ROOT directory is writable by the container user.
- To persist MLflow runs between container restarts, mount a host directory (e.g. /data/mlruns) and pass ARTIFACT_ROOT=/data/mlruns when running the container.

Files of interest

- Dockerfile            # Docker build recipe
- entrypoint.sh         # Starts MLflow server then runs the trainer
- requirements.txt      # Python dependencies
- src/mlflowxgb.py      # Main training script


Developer helpers

You can use the provided `Makefile` to quickly execute commands:
- `make install` - Install python dependencies
- `make start` or `make start-mlflow` - Start MLflow server
- `make image` - Build the Docker image
- `make data` - Download the combined training set data file and sample files (`sample_1.parq`, `sample_5.parq`, `sample_10.parq`) if they don't exist (requires ~4GB disk space)
- `make run-help` - Show run options for mlflowxgb.py
- `make run-<n> [percent]` - Run mlflowxgb.py with argument `<n>` and optionally `<percent>%` data (e.g., `make run-0 10` runs with 10% data, default is 1)

- scripts/make_sample_parquet.py : create a small synthetic sample at data/sample.parq for smoke tests.
- scripts/download_from_s3.py : download a public s3/http file into data/ (uses anon S3 access).
- scripts/run_smoke.sh : helper to run a quick local smoke test (see below).

Smoke test

1) Create sample data: python3 scripts/make_sample_parquet.py
2) Start MLflow server (inside apptainer or locally):
   apptainer exec -B /data/mlruns:/data/mlruns -B /home/hermes:/app mlflow.sif /app/entrypoint.sh
   or (server-only): apptainer exec -B /data/mlruns:/data/mlruns -B /home/hermes:/app mlflow.sif mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /data/mlruns --host 0.0.0.0 --port 5123
3) Run training on sample: python3 src/mlflowxgb.py 0
   Note: For quick testing and iteration, you can run training on a 5% sample of the data and reduce the `num_boost_round` parameter in `xgb.train` to 250 (from the default 1000) inside `src/mlflowxgb.py`, e.g. using `make run-0 5`. Running with 5% data requires about 4-6GB RAM maximum.

