# mlflowxgb.py — Documentation

This document describes the `src/mlflowxgb.py` training script: purpose, usage, configuration and examples.

## Purpose

`mlflowxgb.py` trains XGBoost models on a prepared dataset and logs runs and artifacts to MLflow. It's a lightweight driver that calls the following major functions in the source:

- `fetch_data(fpath)` — read input parquet/csv data
- `prepare_data(trainingset, label, ...)` — filter, sample and prepare features/targets
- `train_sh23_model(data, params, label, mlflow_experiment_id)` — train using xgboost.train and log metrics/artifacts

## Quick start

Clone the repository and run from the repository root:

```bash
# run training for the label mapped to index 4 (met50)
python src/mlflowxgb.py 4
```

### Integer-to-label mapping

Pass a single integer positional argument (0..5) to select the prediction label:

- 0 → `logdist50`
- 1 → `av50`
- 2 → `logteff50`
- 3 → `logg50`
- 4 → `met50`
- 5 → `mass50`


## Command-line interface

The script uses `argparse` and expects one positional integer argument. Example:

```bash
python src/mlflowxgb.py 0
```

If you need more flexible invocation or parameter sweeps, import the functions in `src/mlflowxgb.py` from another driver script and call `prepare_data` / `train_sh23_model` directly.

## Data input

- Default reader: `fetch_data(fpath)` uses a path set in the source code. Update the path there or modify `fetch_data` to accept CLI args if you need runtime control.
- `prepare_data(trainingset, label, random_state=0, frac=0.1, fillvalue=-9.99, cachepath="cache")`:
  - `frac`: fraction of rows to sample (set `1.0` to use all rows)
  - `random_state`: reproducible sampling
  - `fillvalue`: value used to fill missing features

## Training function

`train_sh23_model(data, params, label="target", mlflow_experiment_id=1)`

- Input: prepared `pandas.DataFrame` (output of `prepare_data`).
- `params`: dictionary of XGBoost parameters compatible with `xgboost.train()`.
- Behavior:
  - Splits data into train/test via `train_test_split`.
  - Creates `xgboost.DMatrix` for train and test sets.
  - Calls `mlflow.xgboost.autolog()` and wraps training in an MLflow run.
  - Trains with `xgb.train(...)`, logs metrics and prediction plots as artifacts.

Example usage from Python:

```py
from src.mlflowxgb import fetch_data, prepare_data, train_sh23_model

raw = fetch_data('/path/to/data.parquet')
data = prepare_data(raw, label='met50', frac=0.5)
params = {'objective':'reg:squarederror', 'eta':0.1, 'max_depth':6}
train_sh23_model(data, params, label='met50', mlflow_experiment_id=1)
```

## MLflow configuration

The script may set an MLflow tracking URI in the source. Recommended approach:

- Do not hardcode raw IP addresses in source. Instead set the environment variable:

```bash
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5123"
python src/mlflowxgb.py 4
```

- Alternatively change the script to read configuration from a small JSON/YAML config file or environment variables.

## Environment & dependencies

Minimum dependencies (examples):

- Python 3.8+
- numpy, pandas
- scikit-learn
- scipy
- matplotlib
- xgboost
- mlflow

Install with pip:

```bash
pip install numpy pandas scikit-learn scipy matplotlib xgboost mlflow
```

## Best practices

- Do NOT commit large binary artifacts (models, datasets) to Git. Use the repository `.gitignore` and consider Git LFS or external object storage (S3, MinIO) for model artifacts.
- After any history rewrite (we removed large files earlier), collaborators should re-clone the repository or follow a forced fetch/reset procedure:

```bash
git fetch origin --prune
git reset --hard origin/main
```

- Prefer configuring MLflow via environment variables rather than hardcoding addresses in source.

## Examples

1) Quick run (small sample):

```bash
python src/mlflowxgb.py 4
```

2) Full-data training (edit `frac=1.0` in `prepare_data` or call `prepare_data(..., frac=1.0)` from a driver script).

3) Custom params + programmatic run:

```py
from src.mlflowxgb import fetch_data, prepare_data, train_sh23_model
raw = fetch_data('/data/combined.parq')
data = prepare_data(raw, 'met50', frac=1.0)
params = {'objective':'reg:squarederror','eta':0.05,'max_depth':8}
train_sh23_model(data, params, 'met50', mlflow_experiment_id=2)
```

## Contact / Maintainers

If you have questions about label meanings, MLflow setup, or dataset provenance, contact the repository maintainer.

