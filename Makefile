.PHONY: install start start-mlflow

install:
	pip install -r requirements.txt

start start-mlflow:
	mlflow server --host 0.0.0.0 --port 5000
