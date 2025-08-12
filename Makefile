# Makefile for Solar PV Prediction Pipeline
# ======================================

.PHONY: help full-run clean test install

# Default target
help:
	@echo "Solar PV Prediction Pipeline"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@echo "  full-run    - Run baseline pipeline and generate predictions.parquet"
	@echo "  install     - Install dependencies using Poetry"
	@echo "  test        - Run pipeline tests"
	@echo "  clean       - Clean output files and logs"
	@echo "  help        - Show this help message"
	@echo ""

# Main target: run baseline pipeline
full-run:
	@echo "🚀 Starting Baseline Solar PV Prediction Pipeline..."
	@echo "=================================================="
	@echo ""
	@echo "This will:"
	@echo "  ✓ Load and preprocess radiation data (5% sampling)"
	@echo "  ✓ Load comprehensive PV locations (14,861 assets)"
	@echo "  ✓ Create synthetic generation data"
	@echo "  ✓ Apply feature engineering"
	@echo "  ✓ Train baseline linear model"
	@echo "  ✓ Generate predictions"
	@echo "  ✓ Save predictions.parquet"
	@echo "  ✓ Generate runtime log with profiling"
	@echo ""
	@echo "Starting baseline pipeline execution..."
	@echo ""
	python src/baseline_pipeline_profiled.py
	@echo ""
	@echo "✅ Baseline pipeline completed!"
	@echo "📁 Output files:"
	@echo "  - predictions.parquet (main output)"
	@echo "  - runtime.log (detailed execution log)"
	@echo "  - pipeline_output/pipeline_summary.json (profiling summary)"
	@echo "  - pipeline_output/profiling_report.txt (detailed profiling)"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing dependencies with Poetry..."
	poetry install
	@echo "✅ Dependencies installed successfully!"

# Run tests
test:
	@echo "🧪 Running pipeline tests..."
	python -m pytest tests/ -v
	@echo "✅ Tests completed!"

# Clean output files
clean:
	@echo "🧹 Cleaning output files..."
	rm -rf pipeline_output/
	rm -f runtime.log
	rm -f predictions.parquet
	@echo "✅ Cleanup completed!"

# Docker targets
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t solar-pv-pipeline .
	@echo "✅ Docker image built successfully!"

docker-run:
	@echo "🐳 Running pipeline in Docker..."
	docker run --rm -v $(PWD):/workspace solar-pv-pipeline make full-run
	@echo "✅ Docker pipeline completed!"

# Development targets
dev-install:
	@echo "🔧 Installing development dependencies..."
	poetry install --with dev
	@echo "✅ Development dependencies installed!"

lint:
	@echo "🔍 Running code linting..."
	poetry run flake8 src/ tests/
	@echo "✅ Linting completed!"

format:
	@echo "🎨 Formatting code..."
	poetry run black src/ tests/
	@echo "✅ Code formatting completed!"

# Quick test run with smaller dataset
quick-run:
	@echo "⚡ Running quick test pipeline..."
	@echo "This uses synthetic data for faster testing..."
	python src/main_pipeline.py --quick
	@echo "✅ Quick test completed!"

# Show pipeline status
status:
	@echo "📊 Pipeline Status:"
	@echo "=================="
	@if [ -f "predictions.parquet" ]; then \
		echo "✅ predictions.parquet exists"; \
		ls -lh predictions.parquet; \
	else \
		echo "❌ predictions.parquet not found"; \
	fi
	@if [ -f "runtime.log" ]; then \
		echo "✅ runtime.log exists"; \
		ls -lh runtime.log; \
	else \
		echo "❌ runtime.log not found"; \
	fi
	@if [ -d "pipeline_output" ]; then \
		echo "✅ pipeline_output directory exists"; \
		ls -la pipeline_output/; \
	else \
		echo "❌ pipeline_output directory not found"; \
	fi
