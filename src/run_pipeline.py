#!/usr/bin/env python3
"""
Simple script to run the solar radiation pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from solar_radiation_pipeline import SolarRadiationPipeline

def run_sample_pipeline():
    """Run a sample pipeline with a small dataset for testing."""
    print("Running sample pipeline (2018-2019)...")
    
    pipeline = SolarRadiationPipeline(
        output_dir="data_sample",
        use_polars=True,
        enable_profiling=True
    )
    
    try:
        results = pipeline.run_pipeline(
            start_year=2018, 
            end_year=2019,  # Small sample
            resample=True
        )
        
        print("\n" + "="*50)
        print("SAMPLE PIPELINE RESULTS")
        print("="*50)
        print(f"Hourly data shape: {results['hourly_shape']}")
        print(f"Hourly Parquet directory: {results['hourly_parquet_dir']}")
        
        if 'resampled_shape' in results:
            print(f"15-min data shape: {results['resampled_shape']}")
            print(f"15-min Parquet directory: {results['resampled_parquet_dir']}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Sample pipeline failed: {e}")
        return False
    
    return True

def run_full_pipeline():
    """Run the full pipeline for 2018-2025."""
    print("Running full pipeline (2018-2025)...")
    
    pipeline = SolarRadiationPipeline(
        output_dir="data_full",
        use_polars=True,
        enable_profiling=True
    )
    
    try:
        results = pipeline.run_pipeline(
            start_year=2018, 
            end_year=2025,
            resample=True
        )
        
        print("\n" + "="*50)
        print("FULL PIPELINE RESULTS")
        print("="*50)
        print(f"Hourly data shape: {results['hourly_shape']}")
        print(f"Hourly Parquet directory: {results['hourly_parquet_dir']}")
        
        if 'resampled_shape' in results:
            print(f"15-min data shape: {results['resampled_shape']}")
            print(f"15-min Parquet directory: {results['resampled_parquet_dir']}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Full pipeline failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run solar radiation pipeline")
    parser.add_argument(
        "--sample", 
        action="store_true", 
        help="Run sample pipeline (2018-2019)"
    )
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="Run full pipeline (2018-2025)"
    )
    
    args = parser.parse_args()
    
    if args.sample:
        success = run_sample_pipeline()
    elif args.full:
        success = run_full_pipeline()
    else:
        print("Please specify --sample or --full")
        print("Example: python run_pipeline.py --sample")
        sys.exit(1)
    
    if success:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline failed!")
        sys.exit(1)
