#!/bin/bash

# Create necessary directories
mkdir -p reports
mkdir -p tests/data

# Clean up previous test results
rm -rf reports/*
rm -rf .coverage
rm -rf htmlcov

# Run tests with different configurations
echo "Running unit tests..."
pytest tests/unit -v --cov=src --cov-report=term-missing --cov-report=html -m unit

echo "Running functional tests..."
pytest tests/functional -v --cov=src --cov-report=term-missing --cov-report=html -m functional

echo "Running integration tests..."
pytest tests/integration -v --cov=src --cov-report=term-missing --cov-report=html -m integration

# Generate combined coverage report
coverage combine
coverage html

# Generate test report
pytest --html=reports/test_report.html --self-contained-html

echo "Test results available in:"
echo "- Coverage report: htmlcov/index.html"
echo "- Test report: reports/test_report.html" 