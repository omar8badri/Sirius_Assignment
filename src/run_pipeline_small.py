#!/usr/bin/env python3
"""
Small sample pipeline for testing with minimal data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from solar_radiation_pipeline import SolarRadiationPipeline
import logging
logger = logging.getLogger(__name__)

def run_tiny_sample():
    """Run a tiny sample with just a few days of data."""
    print("Running tiny sample pipeline (January 2018, first 5 days)...")
    
    pipeline = SolarRadiationPipeline(
        output_dir="data_tiny",
        use_polars=True,
        enable_profiling=True
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
        output_file = self.output_dir / "ssrd_germany_tiny_sample.nc"
        
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
    pipeline.fetch_data = fetch_small_data.__get__(pipeline, SolarRadiationPipeline)
    
    try:
        results = pipeline.run_pipeline(
            start_year=2018, 
            end_year=2018,  # Same year
            resample=True
        )
        
        print("\n" + "="*50)
        print("TINY SAMPLE PIPELINE RESULTS")
        print("="*50)
        print(f"Hourly data shape: {results['hourly_shape']}")
        print(f"Hourly Parquet directory: {results['hourly_parquet_dir']}")
        
        if 'resampled_shape' in results:
            print(f"15-min data shape: {results['resampled_shape']}")
            print(f"15-min Parquet directory: {results['resampled_parquet_dir']}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Tiny sample pipeline failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    run_tiny_sample()
