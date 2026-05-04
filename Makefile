.PHONY: help install setup train test deploy clean

# Default target
help:
	@echo "Available commands:"
	@echo "  install    - Install dependencies"
	@echo "  setup      - Setup environment and data"
	@echo "  train      - Train models"
	@echo "  test       - Run tests"
	@echo "  test-all   - Run all tests"
	@echo "  test-report - Run tests with detailed report"
	@echo "  deploy     - Deploy to production"
	@echo "  clean      - Clean generated files"
	@echo "  api        - Start API server"
	@echo "  monitor    - Run monitoring tests"

# Install dependencies
install:
	pip install -r requirements.txt

# Setup environment
setup:
	@echo "Setting up environment..."
	chmod +x scripts/server-start.sh
	./scripts/server-start.sh
	@echo "Environment setup complete!"

# Train models
train:
	@echo "Training models..."
	python -m src.training.train core

# Train with Prefect
train-prefect:
	@echo "Training with Prefect orchestration..."
	python -m src.training.train prefect

# Run basic tests
test:
	@echo "Running basic tests..."
	python -m pytest tests/test_data_processing.py tests/test_api.py -v

# Run all tests
test-all:
	@echo "Running all tests..."
	python -m pytest tests/ -v

# Run tests with detailed report
test-report:
	@echo "Running tests with detailed report..."
	python -m pytest tests/ -v --tb=short --strict-markers
	@echo "Test summary:"
	@python -m pytest tests/ --tb=no -q

# Test monitoring
test-monitoring:
	@echo "Testing monitoring system..."
	python -m pytest tests/test_monitoring.py -v

# Start API server
api:
	@echo "Starting API server..."
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Run monitoring
monitor:
	@echo "Running monitoring..."
	python -m src.monitoring.integration

# Deploy infrastructure
deploy-infra:
	@echo "Deploying infrastructure..."
	cd terraform && terraform init && terraform apply && cd ..

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf artifacts/logs/*
	rm -rf artifacts/reports/*
	@echo "Cleanup complete!"

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

# Lint code
lint:
	@echo "Linting code..."
	flake8 src/ tests/
	pylint src/ tests/

# Full pipeline
pipeline: setup train test api

# Development setup
dev: install setup train-prefect api 