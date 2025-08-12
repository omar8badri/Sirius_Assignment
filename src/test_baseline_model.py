#!/usr/bin/env python3
"""
Test script for the baseline irradiance model.
==============================================

This script tests the baseline model with synthetic data to ensure
all components work correctly before running on real data.
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent))

from baseline_irradiance_model import BaselineIrradianceModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data(days: int = 30) -> tuple:
    """
    Generate synthetic test data for the baseline model.
    
    Parameters:
    -----------
    days : int
        Number of days to generate
        
    Returns:
    --------
    tuple
        (irradiance_df, pv_actuals_df)
    """
    logger.info(f"Generating {days} days of test data...")
    
    # Generate timestamps (15-minute intervals)
    start_date = datetime(2020, 1, 1)
    end_date = start_date + timedelta(days=days)
    
    timestamps = []
    current = start_date
    while current < end_date:
        timestamps.append(current)
        current += timedelta(minutes=15)
    
    # Generate irradiance data (multiple grid points)
    irradiance_data = []
    for timestamp in timestamps:
        hour = timestamp.hour
        month = timestamp.month
        
        # Base irradiance pattern
        if 6 <= hour <= 18:  # Daylight hours
            # Seasonal variation
            seasonal_factor = 0.3 + 0.7 * np.sin((month - 1) * np.pi / 6)
            # Daily pattern (peak at noon)
            daily_factor = np.sin((hour - 6) * np.pi / 12)
            # Add some randomness
            noise = np.random.normal(0, 50)
            
            base_irradiance = 800 * seasonal_factor * daily_factor + noise
            base_irradiance = max(0, base_irradiance)
        else:
            base_irradiance = 0
        
        # Generate multiple grid points
        for lat in [50.0, 51.0, 52.0, 53.0, 54.0]:
            for lon in [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]:
                # Add spatial variation
                spatial_noise = np.random.normal(0, 20)
                irradiance = max(0, base_irradiance + spatial_noise)
                
                irradiance_data.append({
                    'time': timestamp,
                    'latitude': lat,
                    'longitude': lon,
                    'ssrd': round(irradiance, 2)
                })
    
    irradiance_df = pd.DataFrame(irradiance_data)
    
    # Generate PV actuals data
    pv_data = []
    for timestamp in timestamps:
        hour = timestamp.hour
        month = timestamp.month
        
        # Realistic solar generation pattern
        if 6 <= hour <= 18:  # Daylight hours
            # Seasonal variation (higher in summer)
            seasonal_factor = 0.3 + 0.7 * np.sin((month - 1) * np.pi / 6)
            # Daily pattern (peak at noon)
            daily_factor = np.sin((hour - 6) * np.pi / 12)
            # Add some randomness
            noise = np.random.normal(0, 0.1)
            
            # Base capacity for Germany (approximate)
            base_capacity = 50000  # MW
            
            generation = base_capacity * seasonal_factor * daily_factor * (1 + noise)
            generation = max(0, generation)
        else:
            generation = 0
        
        pv_data.append({
            'timestamp': timestamp,
            'solar_generation_mw': round(generation, 2)
        })
    
    pv_df = pd.DataFrame(pv_data)
    
    logger.info(f"Generated {len(irradiance_df):,} irradiance records")
    logger.info(f"Generated {len(pv_df):,} PV actuals records")
    
    return irradiance_df, pv_df

def test_baseline_model():
    """Test the baseline model with synthetic data."""
    print("Testing Baseline Irradiance Model")
    print("=" * 50)
    
    # Generate test data
    irradiance_df, pv_df = generate_test_data(days=30)
    
    # Create temporary files for testing
    test_dir = Path("test_baseline_data")
    test_dir.mkdir(exist_ok=True)
    
    irradiance_path = test_dir / "test_irradiance.parquet"
    pv_path = test_dir / "test_pv_actuals.parquet"
    
    # Save test data
    irradiance_df.to_parquet(irradiance_path)
    pv_df.to_parquet(pv_path)
    
    print(f"Test data saved to: {test_dir}")
    
    # Test different model configurations
    test_configs = [
        {'model_type': 'linear', 'rolling_window_days': 7, 'name': 'Linear-7d'},
        {'model_type': 'ridge', 'rolling_window_days': 7, 'name': 'Ridge-7d'},
        {'model_type': 'linear', 'rolling_window_days': 14, 'name': 'Linear-14d'},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\nTesting {config['name']}...")
        
        # Initialize model
        model = BaselineIrradianceModel(
            irradiance_data_path=irradiance_path,
            pv_actuals_path=pv_path,
            output_dir=Path(f"test_results_{config['name']}"),
            model_type=config['model_type'],
            rolling_window_days=config['rolling_window_days'],
            min_irradiance_threshold=10.0
        )
        
        # Run pipeline
        start_time = time.time()
        result = model.run_complete_pipeline()
        end_time = time.time()
        
        if result['success']:
            results[config['name']] = {
                'rmse': result['metrics']['overall_metrics']['rmse'],
                'r2': result['metrics']['overall_metrics']['r2'],
                'time': end_time - start_time
            }
            print(f"✅ {config['name']} completed successfully!")
            print(f"   RMSE: {result['metrics']['overall_metrics']['rmse']:.2f} MW")
            print(f"   R²: {result['metrics']['overall_metrics']['r2']:.3f}")
            print(f"   Time: {result['total_time_seconds']:.2f} seconds")
        else:
            print(f"❌ {config['name']} failed: {result['error']}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if results:
        print(f"{'Model':<15} {'RMSE (MW)':<12} {'R²':<8} {'Time (s)':<10}")
        print("-" * 50)
        
        for name, metrics in results.items():
            print(f"{name:<15} {metrics['rmse']:<12.2f} {metrics['r2']:<8.3f} {metrics['time']:<10.2f}")
        
        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        print(f"\n🏆 Best model: {best_model[0]} (RMSE: {best_model[1]['rmse']:.2f} MW)")
    
    else:
        print("❌ All tests failed!")
    
    # Cleanup
    print(f"\n🧹 Cleaning up test data...")
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    print("✅ Test completed!")

def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting Data Loading...")
    print("-" * 30)
    
    # Test irradiance data loading
    try:
        model = BaselineIrradianceModel()
        irradiance_df = model.load_irradiance_data()
        print(f"✅ Irradiance data loading: {len(irradiance_df):,} records")
    except Exception as e:
        print(f"⚠️  Irradiance data loading: {e}")
    
    # Test PV actuals data loading
    try:
        pv_df = model.load_pv_actuals_data()
        print(f"✅ PV actuals data loading: {len(pv_df):,} records")
    except Exception as e:
        print(f"⚠️  PV actuals data loading: {e}")

def test_feature_engineering():
    """Test feature engineering functionality."""
    print("\nTesting Feature Engineering...")
    print("-" * 30)
    
    # Generate small test dataset
    irradiance_df, pv_df = generate_test_data(days=7)
    
    # Create model
    model = BaselineIrradianceModel()
    model.irradiance_data = irradiance_df
    model.pv_actuals_data = pv_df
    
    # Test preprocessing
    try:
        combined_df = model.preprocess_data()
        print(f"✅ Preprocessing: {len(combined_df):,} records")
        
        # Test feature creation
        features_df = model.create_features(combined_df)
        print(f"✅ Feature engineering: {features_df.shape[1]} features")
        print(f"   Features: {list(features_df.columns)}")
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")

if __name__ == "__main__":
    print("Baseline Model Test Suite")
    print("=" * 50)
    
    # Test data loading
    test_data_loading()
    
    # Test feature engineering
    test_feature_engineering()
    
    # Test full pipeline
    test_baseline_model()
    
    print("\n🎉 All tests completed!")
