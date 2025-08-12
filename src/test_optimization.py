#!/usr/bin/env python3
"""
Test script to compare performance between original and optimized pipelines.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from solar_radiation_pipeline import SolarRadiationPipeline
from solar_radiation_pipeline_optimized import OptimizedSolarRadiationPipeline

def test_original_pipeline():
    """Test the original pipeline with small data."""
    print("Testing ORIGINAL pipeline...")
    
    pipeline = SolarRadiationPipeline(
        output_dir="test_original",
        use_polars=True,
        enable_profiling=False  # Disable profiling for fair comparison
    )
    
    # Override fetch_data for small sample
    def fetch_small_data(self, start_year=2018, end_year=2019):
        """Fetch a small amount of ERA5 solar radiation data."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Fetching small ERA5 solar radiation data sample")
        
        request = {
            "variable": "surface_solar_radiation_downwards",
            "year": "2018",
            "month": "01",
            "day": [f"{i:02d}" for i in range(1, 6)],  # First 5 days
            "time": [f"{i:02d}:00" for i in range(24)],
            "area": self.germany_area,
            "format": "netcdf"
        }
        
        output_file = self.output_dir / "ssrd_germany_test_original.nc"
        
        try:
            logger.info("Starting small data download...")
            result = self.client.retrieve("reanalysis-era5-land", request)
            result.download(str(output_file))
            logger.info(f"Small data downloaded successfully to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Failed to download small data: {e}")
            raise
    
    pipeline.fetch_data = fetch_small_data.__get__(pipeline, SolarRadiationPipeline)
    
    start_time = time.time()
    
    try:
        results = pipeline.run_pipeline(
            start_year=2018, 
            end_year=2018,
            resample=True
        )
        
        end_time = time.time()
        original_time = end_time - start_time
        
        print(f"Original pipeline completed in {original_time:.2f} seconds")
        print(f"Original pipeline results: {results['resampled_shape']}")
        
        return original_time, results
        
    except Exception as e:
        print(f"Original pipeline failed: {e}")
        return None, None

def test_optimized_pipeline():
    """Test the optimized pipeline with small data."""
    print("Testing OPTIMIZED pipeline...")
    
    pipeline = OptimizedSolarRadiationPipeline(
        output_dir="test_optimized",
        use_polars=True,
        enable_profiling=False,  # Disable profiling for fair comparison
        n_workers=4,
        batch_size=500
    )
    
    # Override fetch_data for small sample
    def fetch_small_data(self, start_year=2018, end_year=2019):
        """Fetch a small amount of ERA5 solar radiation data."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Fetching small ERA5 solar radiation data sample")
        
        request = {
            "variable": "surface_solar_radiation_downwards",
            "year": "2018",
            "month": "01",
            "day": [f"{i:02d}" for i in range(1, 6)],  # First 5 days
            "time": [f"{i:02d}:00" for i in range(24)],
            "area": self.germany_area,
            "format": "netcdf"
        }
        
        output_file = self.output_dir / "ssrd_germany_test_optimized.nc"
        
        try:
            logger.info("Starting small data download...")
            result = self.client.retrieve("reanalysis-era5-land", request)
            result.download(str(output_file))
            logger.info(f"Small data downloaded successfully to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Failed to download small data: {e}")
            raise
    
    pipeline.fetch_data = fetch_small_data.__get__(pipeline, OptimizedSolarRadiationPipeline)
    
    start_time = time.time()
    
    try:
        results = pipeline.run_pipeline(
            start_year=2018, 
            end_year=2018,
            resample=True
        )
        
        end_time = time.time()
        optimized_time = end_time - start_time
        
        print(f"Optimized pipeline completed in {optimized_time:.2f} seconds")
        print(f"Optimized pipeline results: {results['resampled_shape']}")
        
        return optimized_time, results
        
    except Exception as e:
        print(f"Optimized pipeline failed: {e}")
        return None, None

def main():
    """Compare performance between original and optimized pipelines."""
    print("="*60)
    print("PERFORMANCE COMPARISON: ORIGINAL vs OPTIMIZED PIPELINE")
    print("="*60)
    print("Dataset: January 1-5, 2018 (5 days)")
    print("="*60)
    
    # Test original pipeline
    original_time, original_results = test_original_pipeline()
    
    print("\n" + "-"*60)
    
    # Test optimized pipeline
    optimized_time, optimized_results = test_optimized_pipeline()
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*60)
    
    if original_time and optimized_time:
        speedup = original_time / optimized_time
        time_saved = original_time - optimized_time
        
        print(f"Original pipeline time: {original_time:.2f} seconds")
        print(f"Optimized pipeline time: {optimized_time:.2f} seconds")
        print(f"Speedup: {speedup:.2f}x faster")
        print(f"Time saved: {time_saved:.2f} seconds ({time_saved/60:.1f} minutes)")
        
        if original_results and optimized_results:
            print(f"\nData shape comparison:")
            print(f"Original: {original_results['resampled_shape']}")
            print(f"Optimized: {optimized_results['resampled_shape']}")
            
            # Estimate full dataset time
            days_in_full_dataset = 365 * 8  # 8 years
            days_in_test = 5
            
            original_full_time = (original_time / days_in_test) * days_in_full_dataset
            optimized_full_time = (optimized_time / days_in_test) * days_in_full_dataset
            
            print(f"\nEstimated time for full dataset (2018-2025):")
            print(f"Original pipeline: {original_full_time/3600:.1f} hours ({original_full_time/86400:.1f} days)")
            print(f"Optimized pipeline: {optimized_full_time/3600:.1f} hours ({optimized_full_time/86400:.1f} days)")
            print(f"Time saved for full dataset: {(original_full_time - optimized_full_time)/3600:.1f} hours")
    
    print("="*60)

if __name__ == "__main__":
    main()
