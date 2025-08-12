#!/usr/bin/env python3
"""
3-Year Solar Radiation Analysis: 2018-2020
==========================================

This script processes 3 years of ERA5 surface solar radiation data for Germany
using the optimized pipeline. It fetches data month by month to avoid CDS API
limits, converts it to Parquet format, and resamples it to 15-minute intervals
using solar-geometry-aware interpolation.

Output:
- Hourly data in partitioned Parquet format
- 15-minute resampled data in partitioned Parquet format
- Performance metrics and processing statistics
"""

import sys
import time
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from solar_radiation_pipeline_optimized import OptimizedSolarRadiationPipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_data_month_by_month(pipeline, start_year=2018, end_year=2020):
    """Fetch data month by month to avoid CDS API limits."""
    all_dataframes = []
    total_months = (end_year - start_year + 1) * 12
    current_month = 0
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            current_month += 1
            print(f"\n📥 Downloading data for {year}-{month:02d} ({current_month}/{total_months})...")
            
            # Create a temporary pipeline for single month
            temp_pipeline = OptimizedSolarRadiationPipeline(
                output_dir=f"data_temp_{year}_{month:02d}",
                use_polars=False,  # Use pandas for single month
                enable_profiling=False,
                n_workers=2,
                batch_size=500
            )
            
            try:
                # Override the fetch_data method for single month
                def fetch_single_month(self, start_year, end_year):
                    """Fetch ERA5 solar radiation data for a single month."""
                    logger.info(f"Fetching ERA5 solar radiation data for {year}-{month:02d}")
                    
                    # Prepare request for single month
                    request = {
                        "variable": "surface_solar_radiation_downwards",
                        "year": str(year),
                        "month": f"{month:02d}",
                        "day": [f"{i:02d}" for i in range(1, 32)],  # All days
                        "time": [f"{i:02d}:00" for i in range(24)],  # All hours
                        "area": self.germany_area,
                        "format": "netcdf"
                    }
                    
                    # Download file
                    output_file = self.output_dir / f"ssrd_germany_{year}_{month:02d}.nc"
                    
                    try:
                        logger.info(f"Starting data download for {year}-{month:02d}...")
                        result = self.client.retrieve("reanalysis-era5-land", request)
                        result.download(str(output_file))
                        logger.info(f"Data downloaded successfully to {output_file}")
                        return output_file
                    except Exception as e:
                        logger.error(f"Failed to download data for {year}-{month:02d}: {e}")
                        raise
                
                # Replace the fetch_data method
                temp_pipeline.fetch_data = fetch_single_month.__get__(temp_pipeline, OptimizedSolarRadiationPipeline)
                
                # Fetch single month data
                nc_file = temp_pipeline.fetch_data(year, year)
                extracted_file = temp_pipeline.extract_netcdf(nc_file)
                df = temp_pipeline.load_netcdf_data(extracted_file)
                
                all_dataframes.append(df)
                print(f"✅ {year}-{month:02d} downloaded successfully: {df.shape[0]:,} records")
                
            except Exception as e:
                print(f"❌ Failed to download {year}-{month:02d}: {e}")
                raise
    
    # Combine all months
    print(f"\n Combining {len(all_dataframes)} months of data...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"✅ Combined data shape: {combined_df.shape}")
    
    return combined_df

def run_3year_analysis():
    """Run the optimized pipeline on 2018-2020 dataset."""
    print("="*70)
    print("3-YEAR SOLAR RADIATION ANALYSIS: 2018-2020")
    print("="*70)
    print("Dataset: 3 years of ERA5 solar radiation data")
    print("Region: Germany")
    print("Output: Hourly + 15-minute resampled data")
    print("Pipeline: Optimized with parallel processing")
    print("Strategy: Month-by-month download to avoid API limits")
    print("="*70)
    
    # Create optimized pipeline instance
    pipeline = OptimizedSolarRadiationPipeline(
        output_dir="data_3years_2018_2020",
        use_polars=True,
        enable_profiling=True,
        n_workers=6,  # Balanced for 3-year dataset
        batch_size=750  # Optimized batch size
    )
    
    start_time = time.time()
    
    try:
        print("\n🚀 Starting 3-year analysis...")
        print(" Fetching ERA5 data month by month (2018-2020)...")
        
        # Step 1: Fetch data month by month
        df = fetch_data_month_by_month(pipeline, 2018, 2020)
        
        print(f"\n🔄 Processing combined data...")
        
        # Step 2: Convert to partitioned Parquet
        print("📊 Converting to partitioned Parquet format...")
        parquet_dir = pipeline.convert_to_parquet_partitioned(
            df, 
            f"ssrd_germany_2018_2020"
        )
        
        # Step 3: Optimized resampling to 15-minute intervals
        print("⏱️  Resampling to 15-minute intervals...")
        df_resampled = pipeline.optimized_solar_geometry_aware_interpolation(df)
        
        # Step 4: Save resampled data
        print("💾 Saving resampled data...")
        resampled_dir = pipeline.convert_to_parquet_partitioned(
            df_resampled,
            f"ssrd_germany_2018_2020_15min"
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate performance metrics
        days_processed = 3 * 365  # 3 years
        time_per_day = total_time / days_processed
        time_per_year = time_per_day * 365 / 3600
        
        print("\n" + "="*70)
        print("📊 ANALYSIS RESULTS")
        print("="*70)
        
        print(f"⏱️  Processing Time:")
        print(f"   Total: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"   Total: {total_time/3600:.2f} hours")
        print(f"   Per day: {time_per_day:.2f} seconds")
        print(f"   Per year: {time_per_year:.2f} hours")
        
        print(f"\n📈 Data Statistics:")
        print(f"   Days processed: {days_processed:,}")
        print(f"   Hourly records: {df.shape[0]:,}")
        print(f"   Hourly columns: {df.shape[1]}")
        print(f"   15-min records: {df_resampled.shape[0]:,}")
        print(f"   15-min columns: {df_resampled.shape[1]}")
        
        print(f"\n💾 Output Files:")
        print(f"   Hourly data: {parquet_dir}")
        print(f"   15-min data: {resampled_dir}")
        
        print(f"\n✅ Performance Assessment:")
        if total_time < 3600:  # Less than 1 hour
            print(f"   🎉 EXCELLENT: Completed in under 1 hour")
        elif total_time < 7200:  # Less than 2 hours
            print(f"   GOOD: Completed in under 2 hours")
        elif total_time < 14400:  # Less than 4 hours
            print(f"   ⚠️  ACCEPTABLE: Completed in under 4 hours")
        else:
            print(f"   NEEDS OPTIMIZATION: Took longer than 4 hours")
        
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        logger.error(f"3-year analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the 3-year analysis."""
    print("🔬 3-Year Solar Radiation Analysis Pipeline")
    print("Processing ERA5 data for Germany (2018-2020)")
    print("Strategy: Month-by-month download to avoid CDS API limits")
    print("This will take 60-120 minutes depending on your system...")
    print("Total downloads: 36 months (3 years × 12 months)")
    
    success = run_3year_analysis()
    
    if success:
        print("\n🎉 3-year analysis completed successfully!")
        print("📁 Check the 'data_3years_2018_2020' directory for results")
    else:
        print("\n❌ 3-year analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()