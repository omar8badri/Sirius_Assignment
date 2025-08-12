#!/usr/bin/env python3
"""
Configuration file for the baseline irradiance model.
====================================================

This file contains all configurable parameters for the baseline model,
making it easy to experiment with different settings.
"""

from pathlib import Path
from typing import Dict, Any, List

# Data paths (will be auto-detected if None)
DATA_PATHS = {
    'irradiance_data': None,  # Auto-detect from common locations
    'pv_actuals_data': None,  # Auto-detect from common locations
}

# Model configuration
MODEL_CONFIG = {
    'model_type': 'linear',  # 'linear' or 'ridge'
    'rolling_window_days': 30,  # Number of days for rolling window
    'min_irradiance_threshold': 10.0,  # W/m² threshold for daytime detection
}

# Ridge regression specific parameters
RIDGE_CONFIG = {
    'alpha': 1.0,  # Regularization strength
    'solver': 'auto',  # Solver for ridge regression
}

# Feature engineering
FEATURE_CONFIG = {
    'use_time_features': True,  # Include time-based features
    'use_lag_features': True,   # Include lag features
    'use_rolling_features': True,  # Include rolling statistics
    'lag_periods': [1, 4, 96],  # Lag periods in 15-min intervals
    'rolling_windows': [4, 24, 96],  # Rolling window sizes
}

# Training configuration
TRAINING_CONFIG = {
    'step_size_days': 1,  # How many days to step forward in rolling window
    'min_training_samples': 0.8,  # Minimum fraction of window size for training
    'validation_split': 0.2,  # Fraction of training data for validation
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'metrics': ['rmse', 'mae', 'r2', 'mape', 'bias'],
    'time_based_analysis': True,  # Perform hourly/monthly analysis
    'create_visualizations': True,  # Generate plots
}

# Output configuration
OUTPUT_CONFIG = {
    'output_dir': Path('baseline_results'),
    'save_predictions': True,
    'save_coefficients': True,
    'save_metrics': True,
    'save_report': True,
    'save_visualizations': True,
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_file': 'baseline_model.log',
}

# Performance optimization
PERFORMANCE_CONFIG = {
    'use_multiprocessing': False,  # Enable multiprocessing for large datasets
    'chunk_size': 10000,  # Process data in chunks
    'memory_limit_gb': 8,  # Memory limit for processing
}

# Model variants for experimentation
MODEL_VARIANTS = {
    'baseline_linear': {
        'model_type': 'linear',
        'rolling_window_days': 30,
        'min_irradiance_threshold': 10.0,
    },
    'baseline_ridge': {
        'model_type': 'ridge',
        'rolling_window_days': 30,
        'min_irradiance_threshold': 10.0,
        'ridge_alpha': 1.0,
    },
    'short_window': {
        'model_type': 'linear',
        'rolling_window_days': 7,
        'min_irradiance_threshold': 10.0,
    },
    'long_window': {
        'model_type': 'linear',
        'rolling_window_days': 90,
        'min_irradiance_threshold': 10.0,
    },
    'high_threshold': {
        'model_type': 'linear',
        'rolling_window_days': 30,
        'min_irradiance_threshold': 50.0,
    },
    'low_threshold': {
        'model_type': 'linear',
        'rolling_window_days': 30,
        'min_irradiance_threshold': 5.0,
    },
}

# Data quality thresholds
DATA_QUALITY_CONFIG = {
    'max_missing_ratio': 0.1,  # Maximum allowed missing data ratio
    'min_data_points': 1000,   # Minimum required data points
    'outlier_threshold': 3.0,  # Standard deviations for outlier detection
    'min_irradiance_value': 0.0,  # Minimum valid irradiance value
    'max_irradiance_value': 1500.0,  # Maximum valid irradiance value (W/m²)
    'min_generation_value': 0.0,  # Minimum valid generation value
    'max_generation_value': 100000.0,  # Maximum valid generation value (MW)
}

# Validation configuration
VALIDATION_CONFIG = {
    'cross_validation_folds': 5,
    'test_size_ratio': 0.2,
    'random_state': 42,
    'stratify_by_time': True,  # Stratify by time periods
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'figure_size': (18, 12),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis',
    'save_format': 'png',
    'show_plots': False,  # Don't display plots interactively
}

# Units and conversions
UNITS_CONFIG = {
    'irradiance_unit': 'W/m²',
    'generation_unit': 'MW',
    'time_unit': '15min',
    'area_unit': 'km²',
    'capacity_unit': 'MW',
}

# Model interpretation
INTERPRETATION_CONFIG = {
    'feature_importance': True,
    'coefficient_analysis': True,
    'residual_analysis': True,
    'temporal_analysis': True,
    'seasonal_analysis': True,
}

def get_config(variant: str = 'baseline_linear') -> Dict[str, Any]:
    """
    Get configuration for a specific model variant.
    
    Parameters:
    -----------
    variant : str
        Model variant name from MODEL_VARIANTS
        
    Returns:
    --------
    Dict[str, Any]
        Complete configuration dictionary
    """
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    
    config = {
        'data_paths': DATA_PATHS.copy(),
        'model_config': MODEL_CONFIG.copy(),
        'ridge_config': RIDGE_CONFIG.copy(),
        'feature_config': FEATURE_CONFIG.copy(),
        'training_config': TRAINING_CONFIG.copy(),
        'evaluation_config': EVALUATION_CONFIG.copy(),
        'output_config': OUTPUT_CONFIG.copy(),
        'logging_config': LOGGING_CONFIG.copy(),
        'performance_config': PERFORMANCE_CONFIG.copy(),
        'data_quality_config': DATA_QUALITY_CONFIG.copy(),
        'validation_config': VALIDATION_CONFIG.copy(),
        'visualization_config': VISUALIZATION_CONFIG.copy(),
        'units_config': UNITS_CONFIG.copy(),
        'interpretation_config': INTERPRETATION_CONFIG.copy(),
    }
    
    # Override with variant-specific settings
    variant_config = MODEL_VARIANTS[variant]
    config['model_config'].update(variant_config)
    
    return config

def list_available_variants() -> List[str]:
    """
    List all available model variants.
    
    Returns:
    --------
    List[str]
        List of variant names
    """
    return list(MODEL_VARIANTS.keys())

def print_config_summary(variant: str = 'baseline_linear'):
    """
    Print a summary of the configuration for a specific variant.
    
    Parameters:
    -----------
    variant : str
        Model variant name
    """
    config = get_config(variant)
    
    print(f"Configuration for variant: {variant}")
    print("=" * 50)
    print(f"Model Type: {config['model_config']['model_type']}")
    print(f"Rolling Window: {config['model_config']['rolling_window_days']} days")
    print(f"Irradiance Threshold: {config['model_config']['min_irradiance_threshold']} W/m²")
    print(f"Output Directory: {config['output_config']['output_dir']}")
    
    if config['model_config']['model_type'] == 'ridge':
        print(f"Ridge Alpha: {config['ridge_config']['alpha']}")
    
    print(f"Features: {len([k for k, v in config['feature_config'].items() if v])} enabled")
    print(f"Evaluation Metrics: {', '.join(config['evaluation_config']['metrics'])}")

if __name__ == "__main__":
    # Print available variants
    print("Available model variants:")
    for variant in list_available_variants():
        print(f"  - {variant}")
    
    print("\n" + "=" * 50)
    
    # Print default configuration
    print_config_summary('baseline_linear')
