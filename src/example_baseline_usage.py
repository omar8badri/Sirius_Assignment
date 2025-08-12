#!/usr/bin/env python3
"""
Example usage of the baseline irradiance model.
===============================================

This script demonstrates how to use the baseline model with different
configurations and shows the key outputs and analysis.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent))

from baseline_irradiance_model import BaselineIrradianceModel
from config.baseline_config import get_config, list_available_variants

def example_basic_usage():
    """Example 1: Basic usage with default settings."""
    print("Example 1: Basic Usage with Default Settings")
    print("=" * 50)
    
    # Initialize model with default settings
    model = BaselineIrradianceModel(
        output_dir=Path("example_results_basic")
    )
    
    # Run the complete pipeline
    results = model.run_complete_pipeline()
    
    if results['success']:
        print(f"✅ Pipeline completed successfully!")
        print(f"📊 RMSE: {results['metrics']['overall_metrics']['rmse']:.2f} MW")
        print(f"📈 R²: {results['metrics']['overall_metrics']['r2']:.3f}")
        print(f"⏱️  Time: {results['total_time_seconds']:.2f} seconds")
    else:
        print(f"❌ Pipeline failed: {results['error']}")

def example_ridge_regression():
    """Example 2: Using ridge regression with different parameters."""
    print("\nExample 2: Ridge Regression")
    print("=" * 50)
    
    # Initialize ridge regression model
    model = BaselineIrradianceModel(
        model_type='ridge',
        rolling_window_days=60,  # Longer window for ridge
        min_irradiance_threshold=20.0,  # Higher threshold
        output_dir=Path("example_results_ridge")
    )
    
    # Run the pipeline
    results = model.run_complete_pipeline()
    
    if results['success']:
        print(f"✅ Ridge regression completed!")
        print(f"📊 RMSE: {results['metrics']['overall_metrics']['rmse']:.2f} MW")
        print(f"📈 R²: {results['metrics']['overall_metrics']['r2']:.3f}")
        
        # Show coefficient analysis
        if hasattr(model, 'model_coefficients') and not model.model_coefficients.empty:
            print(f"📊 Average coefficient β: {model.model_coefficients['coefficient'].mean():.4f}")
            print(f"📊 Coefficient std: {model.model_coefficients['coefficient'].std():.4f}")
    else:
        print(f"❌ Ridge regression failed: {results['error']}")

def example_configuration_variants():
    """Example 3: Using different configuration variants."""
    print("\nExample 3: Configuration Variants")
    print("=" * 50)
    
    # List available variants
    variants = list_available_variants()
    print(f"Available variants: {variants}")
    
    # Test a few variants
    test_variants = ['short_window', 'long_window', 'high_threshold']
    
    results_comparison = {}
    
    for variant in test_variants:
        print(f"\nTesting variant: {variant}")
        
        # Get configuration
        config = get_config(variant)
        
        # Initialize model with variant config
        model = BaselineIrradianceModel(
            model_type=config['model_config']['model_type'],
            rolling_window_days=config['model_config']['rolling_window_days'],
            min_irradiance_threshold=config['model_config']['min_irradiance_threshold'],
            output_dir=Path(f"example_results_{variant}")
        )
        
        # Run pipeline
        results = model.run_complete_pipeline()
        
        if results['success']:
            results_comparison[variant] = {
                'rmse': results['metrics']['overall_metrics']['rmse'],
                'r2': results['metrics']['overall_metrics']['r2'],
                'time': results['total_time_seconds']
            }
            print(f"  ✅ RMSE: {results['metrics']['overall_metrics']['rmse']:.2f} MW")
            print(f"  📈 R²: {results['metrics']['overall_metrics']['r2']:.3f}")
        else:
            print(f"  ❌ Failed: {results['error']}")
    
    # Print comparison
    if results_comparison:
        print(f"\n📊 Variant Comparison:")
        print(f"{'Variant':<15} {'RMSE (MW)':<12} {'R²':<8} {'Time (s)':<10}")
        print("-" * 50)
        
        for variant, metrics in results_comparison.items():
            print(f"{variant:<15} {metrics['rmse']:<12.2f} {metrics['r2']:<8.3f} {metrics['time']:<10.2f}")
        
        # Find best variant
        best_variant = min(results_comparison.items(), key=lambda x: x[1]['rmse'])
        print(f"\n🏆 Best variant: {best_variant[0]} (RMSE: {best_variant[1]['rmse']:.2f} MW)")

def example_analysis_and_interpretation():
    """Example 4: Detailed analysis and interpretation."""
    print("\nExample 4: Analysis and Interpretation")
    print("=" * 50)
    
    # Run a model for analysis
    model = BaselineIrradianceModel(
        model_type='linear',
        rolling_window_days=30,
        output_dir=Path("example_results_analysis")
    )
    
    results = model.run_complete_pipeline()
    
    if not results['success']:
        print(f"❌ Analysis failed: {results['error']}")
        return
    
    # Load results for analysis
    predictions_path = model.output_dir / "baseline_linear_predictions.csv"
    coefficients_path = model.output_dir / "baseline_linear_coefficients.csv"
    
    if predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path)
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        
        print(f"📊 Analysis of {len(predictions_df):,} predictions")
        
        # Time-based analysis
        predictions_df['hour'] = predictions_df['timestamp'].dt.hour
        predictions_df['month'] = predictions_df['timestamp'].dt.month
        
        # Hourly performance
        hourly_performance = predictions_df.groupby('hour').agg({
            'actual': 'mean',
            'predicted': 'mean',
            'timestamp': 'count'
        }).rename(columns={'timestamp': 'samples'})
        
        hourly_performance['error'] = hourly_performance['predicted'] - hourly_performance['actual']
        hourly_performance['error_pct'] = (hourly_performance['error'] / hourly_performance['actual']) * 100
        
        print(f"\n🕐 Hourly Performance Summary:")
        print(f"{'Hour':<6} {'Actual (MW)':<12} {'Predicted (MW)':<15} {'Error (MW)':<12} {'Error (%)':<10}")
        print("-" * 60)
        
        for hour in range(6, 19):  # Daylight hours
            if hour in hourly_performance.index:
                row = hourly_performance.loc[hour]
                print(f"{hour:<6} {row['actual']:<12.1f} {row['predicted']:<15.1f} {row['error']:<12.1f} {row['error_pct']:<10.1f}")
        
        # Coefficient analysis
        if coefficients_path.exists():
            coeff_df = pd.read_csv(coefficients_path)
            coeff_df['timestamp'] = pd.to_datetime(coeff_df['timestamp'])
            
            print(f"\n📈 Model Coefficient Analysis:")
            print(f"Average intercept (α): {coeff_df['intercept'].mean():.2f}")
            print(f"Average coefficient (β): {coeff_df['coefficient'].mean():.4f}")
            print(f"Coefficient std: {coeff_df['coefficient'].std():.4f}")
            print(f"Coefficient range: {coeff_df['coefficient'].min():.4f} to {coeff_df['coefficient'].max():.4f}")
            
            # Seasonal coefficient analysis
            coeff_df['month'] = coeff_df['timestamp'].dt.month
            monthly_coeff = coeff_df.groupby('month')['coefficient'].mean()
            
            print(f"\n📅 Seasonal Coefficient Pattern:")
            for month in range(1, 13):
                if month in monthly_coeff.index:
                    print(f"Month {month:2d}: β = {monthly_coeff[month]:.4f}")

def example_custom_data_paths():
    """Example 5: Using custom data paths."""
    print("\nExample 5: Custom Data Paths")
    print("=" * 50)
    
    # Example with specific data paths
    irradiance_path = Path("data_3years_2018_2020_final/ssrd_germany_2018_2020_combined_15min")
    pv_path = Path("data_german_solar_generation/processed")
    
    print(f"Irradiance data: {irradiance_path}")
    print(f"PV actuals data: {pv_path}")
    
    # Check if paths exist
    if irradiance_path.exists():
        print(f"✅ Irradiance data found")
    else:
        print(f"⚠️  Irradiance data not found, will use auto-detection")
    
    if pv_path.exists():
        print(f"✅ PV actuals data found")
    else:
        print(f"⚠️  PV actuals data not found, will use auto-detection")
    
    # Initialize model with custom paths
    model = BaselineIrradianceModel(
        irradiance_data_path=irradiance_path if irradiance_path.exists() else None,
        pv_actuals_path=pv_path if pv_path.exists() else None,
        output_dir=Path("example_results_custom_paths")
    )
    
    # Run pipeline
    results = model.run_complete_pipeline()
    
    if results['success']:
        print(f"✅ Custom paths pipeline completed!")
        print(f"📊 RMSE: {results['metrics']['overall_metrics']['rmse']:.2f} MW")
    else:
        print(f"❌ Custom paths pipeline failed: {results['error']}")

def main():
    """Run all examples."""
    print("Baseline Irradiance Model - Usage Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    example_ridge_regression()
    example_configuration_variants()
    example_analysis_and_interpretation()
    example_custom_data_paths()
    
    print("\n" + "=" * 60)
    print("🎉 All examples completed!")
    print("\n📁 Check the 'example_results_*' directories for outputs")
    print("📊 Review the generated reports and visualizations")
    print("🔧 Experiment with different configurations using the config system")

if __name__ == "__main__":
    main()
