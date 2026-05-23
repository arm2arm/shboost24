README: src/mlflowxgb.py

This README explains how to run src/mlflowxgb.py and configure its main parameters.

Usage:
  python src/mlflowxgb.py <n>
Where <n> is one of:
  0: logdist50
  1: av50
  2: logteff50
  3: logg50
  4: met50
  5: mass50

Quick notes:
- The script reads data via fetch_data(); adjust the path in the source if needed.
- Use MLFLOW_TRACKING_URI to point to your MLflow server instead of hardcoded IP.
- Keep large data/model files out of git (use .gitignore / Git LFS / external storage).

Example:
  python src/mlflowxgb.py 4

