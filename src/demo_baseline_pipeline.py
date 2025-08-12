#!/usr/bin/env python3
"""
Demo script for the baseline irradiance model pipeline (Fixed Version).
=====================================================================

This script demonstrates the baseline model with synthetic data
to verify the pipeline is working correctly.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent))

from baseline_irradiance_model import BaselineIrradianceModel

def generate_synthetic_irradiance_data():
    """
    Generate synthetic irradiance data for demonstration.
    """
    print("📊 Generating synthetic irradiance data...")
    
    # Generate 3 years of 15-minute data (2018-2020)
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2021, 1, 1)
    
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
    
    df = pd.DataFrame(irradiance_data)
    print(f"✅ Generated {len(df):,} irradiance records")
    return df

def generate_synthetic_pv_data():
    """
    Generate synthetic PV actuals data for demonstration.
    """
    print("📊 Generating synthetic PV actuals data...")
    
    # Generate 3 years of 15-minute data (2018-2020)
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2021, 1, 1)
    
    timestamps = []
    current = start_date
    while current < end_date:
        timestamps.append(current)
        current += timedelta(minutes=15)
    
    data = []
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
        
        data.append({
            'timestamp': timestamp,
            'solar_generation_mw': round(generation, 2)
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Generated {len(df):,} PV actuals records")
    return df

def main():
    """Run the baseline pipeline demo with synthetic data."""
    print("Baseline Irradiance Model - Pipeline Demo (Fixed)")
    print("=" * 55)
    
    # Create temporary directory for synthetic data
    temp_dir = Path("temp_synthetic_data")
    temp_dir.mkdir(exist_ok=True)
    
    # Generate synthetic data
    irradiance_df = generate_synthetic_irradiance_data()
    pv_df = generate_synthetic_pv_data()
    
    # Save synthetic data to temporary files
    irradiance_path = temp_dir / "synthetic_irradiance.parquet"
    pv_path = temp_dir / "synthetic_pv_actuals.parquet"
    
    irradiance_df.to_parquet(irradiance_path)
    pv_df.to_parquet(pv_path)
    
    print(f"💾 Synthetic data saved to: {temp_dir}")
    
    # Create model with synthetic data paths
    model = BaselineIrradianceModel(
        irradiance_data_path=irradiance_path,
        pv_actuals_path=pv_path,
        output_dir=Path('baseline_results_demo'),
        model_type='linear',
        rolling_window_days=30,
        min_irradiance_threshold=10.0
    )
    
    print("\n🚀 Starting baseline model pipeline...")
    print("📊 Using synthetic data for demonstration")
    print("⏱️  This will take a few minutes...")
    print()
    
    # Run the complete pipeline
    results = model.run_complete_pipeline()
    
    # Display results
    if results['success']:
        print("\n" + "=" * 55)
        print("✅ BASELINE PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 55)
        
        metrics = results['metrics']['overall_metrics']
        print(f"📊 Overall RMSE: {metrics['rmse']:.2f} MW")
        print(f"📊 Overall MAE: {metrics['mae']:.2f} MW")
        print(f"📈 Overall R²: {metrics['r2']:.3f}")
        print(f"�� MAPE: {metrics['mape']:.2f}%")
        print(f"📊 Bias: {metrics['bias']:.2f} MW")
        print(f"⏱️  Total Time: {results['total_time_seconds']:.2f} seconds")
        print(f"📁 Results saved to: {results['output_directory']}")
        
        # Show hourly performance summary
        print("\n�� HOURLY PERFORMANCE SUMMARY")
        print("-" * 30)
        hourly_metrics = results['metrics']['hourly_metrics']
        
        # Show best and worst hours
        worst_hours = sorted(hourly_metrics.items(), key=lambda x: x[1]['rmse'], reverse=True)[:3]
        best_hours = sorted(hourly_metrics.items(), key=lambda x: x[1]['rmse'])[:3]
        
        print("Worst performing hours:")
        for hour, metrics in worst_hours:
            print(f"  {hour:02d}:00 - RMSE: {metrics['rmse']:.2f} MW, R²: {metrics['r2']:.3f}")
        
        print("\nBest performing hours:")
        for hour, metrics in best_hours:
            print(f"  {hour:02d}:00 - RMSE: {metrics['rmse']:.2f} MW, R²: {metrics['r2']:.3f}")
        
        print(f"\n📈 Model Performance Summary:")
        print(f"   - The baseline model achieved R² = {metrics['r2']:.3f}")
        print(f"   - RMSE of {metrics['rmse']:.2f} MW indicates prediction accuracy")
        print(f"   - The model successfully captures the irradiance-generation relationship")
        
        print(f"\n📁 Generated Files:")
        output_dir = Path(results['output_directory'])
        for file in output_dir.glob("*"):
            print(f"   - {file.name}")
        
        print(f"\n🎉 Pipeline demo completed successfully!")
        print(f"   You can now upload your real data and run the same pipeline!")
        
    else:
        print(f"\n❌ Pipeline failed: {results['error']}")
        print("Please check the error message and try again.")
    
    # Cleanup
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"\n🧹 Cleaned up temporary synthetic data files")

if __name__ == "__main__":
    main()