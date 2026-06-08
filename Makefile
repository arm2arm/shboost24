.PHONY: help install start start-mlflow image data

help:
	@echo "Available commands:"
	@echo "  make help         - Show this help message"
	@echo "  make install      - Install python dependencies"
	@echo "  make start        - Start MLflow server"
	@echo "  make start-mlflow - Start MLflow server (alias for start)"
	@echo "  make image        - Build the Docker image"
	@echo "  make data         - Download the data file if it doesn't exist"
	@echo "  make run-help     - Show run options for mlflowxgb.py"
	@echo "  make run-<n>      - Run mlflowxgb.py with argument <n> (e.g., make run-0)"

install:
	pip install -r requirements.txt

start start-mlflow:
	mlflow server --host 0.0.0.0 --port 5000

image:
	docker build -t shboost24 .

data/combined_trainingset2_with_xp_norm.ONE.parq:
	mkdir -p data
	curl -o data/combined_trainingset2_with_xp_norm.ONE.parq https://s3.data.aip.de:9000/shboost2024/combined_trainingset2_with_xp_norm.ONE.parq

data: data/combined_trainingset2_with_xp_norm.ONE.parq

run-help:
	python src/mlflowxgb.py --help

run-%:
	python src/mlflowxgb.py $* $(filter-out $@,$(MAKECMDGOALS))

# Catch-all rule for extraneous arguments so make doesn't complain
%:
	@:
