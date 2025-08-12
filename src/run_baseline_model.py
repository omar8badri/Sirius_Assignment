#!/usr/bin/env python3
"""
Runner script for the baseline irradiance model.
===============================================

This script provides a convenient interface to run the baseline model
with different configurations and data sources.
"""

import sys
import argparse
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from baseline_irradiance_model import BaselineIrradianceModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Run baseline irradiance model for German solar PV production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_baseline_model.py
  
  # Run with specific data paths
  python run_baseline_model.py --irradiance data_3years_2018_2020_final/ssrd_germany_2018_2020_combined_15min --pv-actuals data_german_solar_generation/processed
  
  # Run ridge regression with 60-day window
  python run_baseline_model.py --model-type ridge --window-days 60
  
  # Run with custom output directory
  python run_baseline_model.py --output-dir my_baseline_results
        """
    )
    
    # Data paths
    parser.add_argument(
        '--irradiance',
        type=Path,
        help='Path to irradiance data (Parquet directory or file)'
    )
    
    parser.add_argument(
        '--pv-actuals',
        type=Path,
        help='Path to PV actuals data (Parquet or CSV file)'
    )
    
    # Model parameters
    parser.add_argument(
        '--model-type',
        choices=['linear', 'ridge'],
        default='linear',
        help='Model type: linear or ridge regression (default: linear)'
    )
    
    parser.add_argument(
        '--window-days',
        type=int,
        default=30,
        help='Rolling window size in days (default: 30)'
    )
    
    parser.add_argument(
        '--irradiance-threshold',
        type=float,
        default=10.0,
        help='Minimum irradiance threshold for daytime detection in W/m² (default: 10.0)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('baseline_results'),
        help='Output directory for results (default: baseline_results)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print configuration
    print("Baseline Irradiance Model Configuration")
    print("=" * 50)
    print(f"Model Type: {args.model_type}")
    print(f"Rolling Window: {args.window_days} days")
    print(f"Irradiance Threshold: {args.irradiance_threshold} W/m²")
    print(f"Output Directory: {args.output_dir}")
    if args.irradiance:
        print(f"Irradiance Data: {args.irradiance}")
    if args.pv_actuals:
        print(f"PV Actuals Data: {args.pv_actuals}")
    print()
    
    # Initialize model
    model = BaselineIrradianceModel(
        irradiance_data_path=args.irradiance,
        pv_actuals_path=args.pv_actuals,
        output_dir=args.output_dir,
        model_type=args.model_type,
        rolling_window_days=args.window_days,
        min_irradiance_threshold=args.irradiance_threshold
    )
    
    # Run pipeline
    results = model.run_complete_pipeline()
    
    # Print results
    if results['success']:
        print("\n" + "=" * 50)
        print("✅ BASELINE MODEL COMPLETED SUCCESSFULLY")
        print("=" * 50)
        
        metrics = results['metrics']['overall_metrics']
        print(f"📊 Overall RMSE: {metrics['rmse']:.2f} MW")
        print(f"📊 Overall MAE: {metrics['mae']:.2f} MW")
        print(f"📈 Overall R²: {metrics['r2']:.3f}")
        print(f"📊 MAPE: {metrics['mape']:.2f}%")
        print(f"📊 Bias: {metrics['bias']:.2f} MW")
        print(f"⏱️  Total Time: {results['total_time_seconds']:.2f} seconds")
        print(f"📁 Results saved to: {results['output_directory']}")
        
        # Print hourly performance summary
        print("\n🕐 HOURLY PERFORMANCE SUMMARY")
        print("-" * 30)
        hourly_metrics = results['metrics']['hourly_metrics']
        worst_hours = sorted(hourly_metrics.items(), key=lambda x: x[1]['rmse'], reverse=True)[:5]
        best_hours = sorted(hourly_metrics.items(), key=lambda x: x[1]['rmse'])[:5]
        
        print("Worst performing hours:")
        for hour, metrics in worst_hours:
            print(f"  {hour:02d}:00 - RMSE: {metrics['rmse']:.2f} MW, R²: {metrics['r2']:.3f}")
        
        print("\nBest performing hours:")
        for hour, metrics in best_hours:
            print(f"  {hour:02d}:00 - RMSE: {metrics['rmse']:.2f} MW, R²: {metrics['r2']:.3f}")
        
    else:
        print(f"\n❌ Pipeline failed: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
