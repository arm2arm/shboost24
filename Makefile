.PHONY: help install start start-mlflow image

help:
	@echo "Available commands:"
	@echo "  make help         - Show this help message"
	@echo "  make install      - Install python dependencies"
	@echo "  make start        - Start MLflow server"
	@echo "  make start-mlflow - Start MLflow server (alias for start)"
	@echo "  make image        - Build the Docker image"

install:
	pip install -r requirements.txt

start start-mlflow:
	mlflow server --host 0.0.0.0 --port 5000

image:
	docker build -t shboost24 .
