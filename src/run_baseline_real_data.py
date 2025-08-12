#!/usr/bin/env python3
"""
Run baseline model with real data from all three sources.
=======================================================

This script handles the partitioned parquet format and runs the baseline
model with the user's real irradiance and PV generation data.
"""

import sys
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

def load_partitioned_irradiance_data(base_path: Path) -> pd.DataFrame:
    """
    Load partitioned irradiance data properly.
    
    Parameters:
    -----------
    base_path : Path
        Path to the partitioned data directory
        
    Returns:
    --------
    pd.DataFrame
        Combined irradiance data
    """
    logger.info(f"Loading partitioned irradiance data from: {base_path}")
    
    all_data = []
    
    # Iterate through years
    for year_dir in sorted(base_path.glob("year=*")):
        year = int(year_dir.name.split("=")[1])
        logger.info(f"Processing year {year}...")
        
        # Iterate through months
        for month_dir in sorted(year_dir.glob("month=*")):
            month = int(month_dir.name.split("=")[1])
            logger.info(f"  Processing month {month}...")
            
            # Load the parquet file for this month
            parquet_files = list(month_dir.glob("*.parquet"))
            if parquet_files:
                file_path = parquet_files[0]
                try:
                    # Load with specific engine to handle partitioning
                    df = pd.read_parquet(file_path, engine='pyarrow')
                    
                    # Add year and month columns if they don't exist
                    if 'year' not in df.columns:
                        df['year'] = year
                    if 'month' not in df.columns:
                        df['month'] = month
                    
                    all_data.append(df)
                    logger.info(f"    Loaded {len(df):,} records from {file_path.name}")
                    
                except Exception as e:
                    logger.warning(f"    Error loading {file_path}: {e}")
                    continue
    
    if not all_data:
        raise ValueError("No irradiance data could be loaded")
    
    # Combine all data
    logger.info("Combining all partitioned data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure required columns exist
    required_cols = ['time', 'ssrd']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        logger.info(f"Available columns: {list(combined_df.columns)}")
        
        # Try to map common column names
        column_mapping = {
            'timestamp': 'time',
            'solar_radiation': 'ssrd',
            'radiation': 'ssrd'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in combined_df.columns and new_col not in combined_df.columns:
                combined_df = combined_df.rename(columns={old_col: new_col})
                logger.info(f"Renamed '{old_col}' to '{new_col}'")
    
    # Convert time to datetime if needed
    if 'time' in combined_df.columns and not pd.api.types.is_datetime64_any_dtype(combined_df['time']):
        combined_df['time'] = pd.to_datetime(combined_df['time'])
    
    # Sort by time
    combined_df = combined_df.sort_values('time').reset_index(drop=True)
    
    logger.info(f"Successfully loaded {len(combined_df):,} total irradiance records")
    logger.info(f"Time range: {combined_df['time'].min()} to {combined_df['time'].max()}")
    logger.info(f"Columns: {list(combined_df.columns)}")
    
    return combined_df

def load_pv_actuals_data(pv_path: Path) -> pd.DataFrame:
    """
    Load PV actuals data.
    
    Parameters:
    -----------
    pv_path : Path
        Path to PV actuals data
        
    Returns:
    --------
    pd.DataFrame
        PV actuals data
    """
    logger.info(f"Loading PV actuals data from: {pv_path}")
    
    try:
        # Load data
        if pv_path.suffix == '.parquet':
            df = pd.read_parquet(pv_path)
        else:
            df = pd.read_csv(pv_path)
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'solar_generation_mw']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            logger.info(f"Available columns: {list(df.columns)}")
            
            # Try to map common column names
            column_mapping = {
                'time': 'timestamp',
                'generation_mw': 'solar_generation_mw',
                'solar_generation': 'solar_generation_mw'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})
                    logger.info(f"Renamed '{old_col}' to '{new_col}'")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Successfully loaded {len(df):,} PV actuals records")
        logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading PV actuals data: {e}")
        raise

def run_baseline_with_real_data():
    """
    Run the baseline model with real data from all sources.
    """
    print("Baseline Model with Real Data from All Sources")
    print("=" * 55)
    
    # Define data paths
    irradiance_path = Path("data_3years_2018_2020_final/ssrd_germany_2018_2020_combined")
    pv_path = Path("data_german_solar_generation/processed/german_solar_generation_v3_2018_2020.parquet")
    output_dir = Path("baseline_results_real_data")
    
    print(f"�� Irradiance data: {irradiance_path}")
    print(f"�� PV actuals data: {pv_path}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Check if data exists
    if not irradiance_path.exists():
        print(f"❌ Irradiance data not found: {irradiance_path}")
        return
    
    if not pv_path.exists():
        print(f"❌ PV actuals data not found: {pv_path}")
        return
    
    print("✅ Data sources found!")
    print("🚀 Loading and processing data...")
    print()
    
    try:
        # Load data
        irradiance_df = load_partitioned_irradiance_data(irradiance_path)
        pv_df = load_pv_actuals_data(pv_path)
        
        # Create temporary files for the model
        temp_dir = Path("temp_real_data")
        temp_dir.mkdir(exist_ok=True)
        
        irradiance_temp_path = temp_dir / "irradiance_combined.parquet"
        pv_temp_path = temp_dir / "pv_actuals.parquet"
        
        # Save processed data
        irradiance_df.to_parquet(irradiance_temp_path)
        pv_df.to_parquet(pv_temp_path)
        
        print(f"💾 Processed data saved to: {temp_dir}")
        
        # Create model with real data
        model = BaselineIrradianceModel(
            irradiance_data_path=irradiance_temp_path,
            pv_actuals_path=pv_temp_path,
            output_dir=output_dir,
            model_type='linear',
            rolling_window_days=30,
            min_irradiance_threshold=10.0
        )
        
        print("\n🚀 Starting baseline model pipeline...")
        print("📊 Using real data from all sources")
        print("⏱️  This will take several minutes...")
        print()
        
        # Run the complete pipeline
        results = model.run_complete_pipeline()
        
        # Display results
        if results['success']:
            print("\n" + "=" * 55)
            print("✅ BASELINE MODEL WITH REAL DATA COMPLETED!")
            print("=" * 55)
            
            metrics = results['metrics']['overall_metrics']
            print(f"📊 Overall RMSE: {metrics['rmse']:.2f} MW")
            print(f"📊 Overall MAE: {metrics['mae']:.2f} MW")
            print(f"📈 Overall R²: {metrics['r2']:.3f}")
            print(f"�� MAPE: {metrics['mape']:.2f}%")
            print(f"📊 Bias: {metrics['bias']:.2f} MW")
            print(f"⏱️  Total Time: {results['total_time_seconds']:.2f} seconds")
            print(f"📁 Results saved to: {results['output_directory']}")
            
            # Data summary
            print(f"\n📊 Data Summary:")
            print(f"   Irradiance records: {len(irradiance_df):,}")
            print(f"   PV actuals records: {len(pv_df):,}")
            print(f"   Time range: {irradiance_df['time'].min()} to {irradiance_df['time'].max()}")
            
            # Show hourly performance summary
            print(f"\n�� HOURLY PERFORMANCE SUMMARY")
            print("-" * 30)
            hourly_metrics = results['metrics']['hourly_metrics']
            
            # Show best and worst hours
            worst_hours = sorted(hourly_metrics.items(), key=lambda x: x[1]['rmse'], reverse=True)[:5]
            best_hours = sorted(hourly_metrics.items(), key=lambda x: x[1]['rmse'])[:5]
            
            print("Worst performing hours:")
            for hour, metrics in worst_hours:
                print(f"  {hour:02d}:00 - RMSE: {metrics['rmse']:.2f} MW, R²: {metrics['r2']:.3f}")
            
            print("\nBest performing hours:")
            for hour, metrics in best_hours:
                print(f"  {hour:02d}:00 - RMSE: {metrics['rmse']:.2f} MW, R²: {metrics['r2']:.3f}")
            
            print(f"\n�� Real data baseline model completed successfully!")
            print(f"   Check the output directory for detailed results and visualizations!")
            
        else:
            print(f"\n❌ Pipeline failed: {results['error']}")
            print("Please check the error message and try again.")
        
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary files")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check the data format and try again.")

def main():
    """Main function."""
    run_baseline_with_real_data()

if __name__ == "__main__":
    main()