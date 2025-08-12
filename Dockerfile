# Dockerfile for Solar PV Prediction Pipeline
# ==========================================

# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not create virtual environment (use system Python)
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --only main --no-interaction --no-ansi --no-root

# Copy source code
COPY src/ ./src/
COPY Makefile ./

# Create output directory
RUN mkdir -p pipeline_output

# Set environment variables
ENV PYTHONPATH=/workspace/src
ENV PYTHONUNBUFFERED=1
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV GDAL_VERSION=3.4.3

# Default command
CMD ["make", "help"]
