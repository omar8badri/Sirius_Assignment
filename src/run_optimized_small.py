#!/usr/bin/env python3
"""
Small optimized pipeline for testing with minimal data.
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from solar_radiation_pipeline_optimized import OptimizedSolarRadiationPipeline
import logging
logger = logging.getLogger(__name__)

def run_optimized_tiny_sample():
    """Run an optimized tiny sample with just a few days of data."""
    print("Running OPTIMIZED tiny sample pipeline (January 2018, first 5 days)...")
    
    pipeline = OptimizedSolarRadiationPipeline(
        output_dir="data_optimized_tiny",
        use_polars=True,
        enable_profiling=True,
        n_workers=4,  # Use 4 workers for parallel processing
        batch_size=500  # Process 500 locations at a time
    )
    
    # Override the fetch_data method to get smaller data
    def fetch_small_data(self, start_year=2018, end_year=2019):
        """Fetch a small amount of ERA5 solar radiation data."""
        logger.info(f"Fetching small ERA5 solar radiation data sample")
        
        # CDS request parameters for small sample
        request = {
            "variable": "surface_solar_radiation_downwards",
            "year": "2018",
            "month": "01",  # Only January
            "day": [f"{i:02d}" for i in range(1, 6)],  # Only first 5 days
            "time": [f"{i:02d}:00" for i in range(24)],  # All hours
            "area": self.germany_area,
            "format": "netcdf"
        }
        
        # Download file
        output_file = self.output_dir / "ssrd_germany_optimized_tiny_sample.nc"
        
        try:
            logger.info("Starting small data download...")
            result = self.client.retrieve("reanalysis-era5-land", request)
            result.download(str(output_file))
            logger.info(f"Small data downloaded successfully to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Failed to download small data: {e}")
            raise
    
    # Replace the fetch_data method
    pipeline.fetch_data = fetch_small_data.__get__(pipeline, OptimizedSolarRadiationPipeline)
    
    start_time = time.time()
    
    try:
        results = pipeline.run_pipeline(
            start_year=2018, 
            end_year=2018,  # Same year
            resample=True
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*50)
        print("OPTIMIZED TINY SAMPLE PIPELINE RESULTS")
        print("="*50)
        print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Hourly data shape: {results['hourly_shape']}")
        print(f"Hourly Parquet directory: {results['hourly_parquet_dir']}")
        
        if 'resampled_shape' in results:
            print(f"15-min data shape: {results['resampled_shape']}")
            print(f"15-min Parquet directory: {results['resampled_parquet_dir']}")
        
        # Compare with original timing (11+ minutes)
        original_time = 11 * 60  # 11 minutes in seconds
        speedup = original_time / total_time
        time_saved = original_time - total_time
        
        print(f"\nPERFORMANCE COMPARISON:")
        print(f"Original pipeline: ~{original_time/60:.1f} minutes")
        print(f"Optimized pipeline: {total_time/60:.2f} minutes")
        print(f"Speedup: {speedup:.1f}x faster")
        print(f"Time saved: {time_saved/60:.1f} minutes")
        
        print("="*50)
        
    except Exception as e:
        print(f"Optimized tiny sample pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    run_optimized_tiny_sample()
