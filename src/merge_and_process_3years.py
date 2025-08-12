#!/usr/bin/env python3
"""
Merge and Process 3-Year Solar Radiation Data
=============================================

This script merges all the monthly ERA5 data that has been downloaded and
processes it through the optimized pipeline to convert to Parquet format
and resample to 15-minute intervals using solar-geometry-aware interpolation.

Input: 36 monthly data directories (data_temp_YYYY_MM)
Output: Combined dataset in Parquet format with 15-minute resampling
"""

import sys
import time
import pandas as pd
import xarray as xr
from pathlib import Path
import glob

# Add src to path
sys.path.append(str(Path(__file__).parent))

from solar_radiation_pipeline_optimized import OptimizedSolarRadiationPipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_monthly_data_directories():
    """Find all monthly data directories."""
    pattern = "data_temp_*_*"
    directories = glob.glob(pattern)
    directories = [d for d in directories if Path(d).is_dir()]
    directories.sort()  # Sort chronologically
    return directories

def load_monthly_data(directory):
    """Load data from a monthly directory."""
    print(f"📂 Loading data from {directory}...")
    
    # Look for NetCDF files
    nc_files = list(Path(directory).glob("*.nc"))
    if not nc_files:
        print(f"⚠️  No NetCDF files found in {directory}")
        return None
    
    nc_file = nc_files[0]
    print(f"📄 Found NetCDF file: {nc_file.name}")
    
    try:
        # Check if this is a zip archive (original download)
        if nc_file.name.startswith("ssrd_germany_"):
            print(f"🔧 Extracting zip archive: {nc_file.name}")
            import zipfile
            import tempfile
            
            # Extract the zip file
            with zipfile.ZipFile(nc_file, 'r') as zip_ref:
                # Extract to the same directory
                zip_ref.extractall(directory)
            
            # Look for the extracted file
            extracted_files = list(Path(directory).glob("data_*.nc"))
            if extracted_files:
                nc_file = extracted_files[0]
                print(f"✅ Extracted: {nc_file.name}")
            else:
                print(f"❌ No extracted NetCDF file found in {directory}")
                return None
        
        # Load with xarray
        print(f"📖 Loading NetCDF file: {nc_file}")
        ds = xr.open_dataset(nc_file, engine='h5netcdf')
        
        # Get the ssrd variable
        ssrd = ds['ssrd']
        
        # Convert to DataFrame
        df = ssrd.to_dataframe().reset_index()
        
        # Clean up the data
        df = df.dropna()
        
        # Handle time column (could be 'time' or 'valid_time')
        time_col = None
        for col in ['time', 'valid_time']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col:
            # Rename to 'time' for consistency
            df = df.rename(columns={time_col: 'time'})
            df['time'] = pd.to_datetime(df['time'])
        else:
            raise ValueError("No time column found in the data")
        
        print(f"✅ Loaded {df.shape[0]:,} records from {directory}")
        return df
        
    except Exception as e:
        print(f"❌ Failed to load {directory}: {e}")
        # Try alternative approach for zip files
        if nc_file.name.startswith("ssrd_germany_"):
            print(f"🔄 Trying alternative extraction method...")
            try:
                import zipfile
                import tempfile
                import os
                
                # Create a temporary directory for extraction
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Extract to temp directory
                    with zipfile.ZipFile(nc_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Find the extracted file
                    extracted_files = list(Path(temp_dir).glob("*.nc"))
                    if extracted_files:
                        temp_nc_file = extracted_files[0]
                        print(f"📄 Found extracted file: {temp_nc_file.name}")
                        
                        # Load with xarray
                        ds = xr.open_dataset(temp_nc_file, engine='h5netcdf')
                        ssrd = ds['ssrd']
                        df = ssrd.to_dataframe().reset_index()
                        df = df.dropna()
                        
                        # Handle time column
                        time_col = None
                        for col in ['time', 'valid_time']:
                            if col in df.columns:
                                time_col = col
                                break
                        
                        if time_col:
                            df = df.rename(columns={time_col: 'time'})
                            df['time'] = pd.to_datetime(df['time'])
                        else:
                            raise ValueError("No time column found in the data")
                        
                        print(f"✅ Loaded {df.shape[0]:,} records from {directory} (alternative method)")
                        return df
                    else:
                        print(f"❌ No extracted files found in temp directory")
                        return None
                        
            except Exception as e2:
                print(f"❌ Alternative method also failed: {e2}")
                return None
        return None

def merge_monthly_data():
    """Merge all monthly data into a single dataset."""
    print("="*70)
    print("MERGE AND PROCESS 3-YEAR SOLAR RADIATION DATA")
    print("="*70)
    print("Step 1: Finding monthly data directories...")
    
    # Find all monthly directories
    directories = find_monthly_data_directories()
    print(f"📁 Found {len(directories)} monthly data directories")
    
    if len(directories) == 0:
        print("❌ No monthly data directories found!")
        return None
    
    # Load data from each directory
    all_dataframes = []
    successful_loads = 0
    
    for directory in directories:
        df = load_monthly_data(directory)
        if df is not None:
            all_dataframes.append(df)
            successful_loads += 1
    
    print(f"\n📊 Successfully loaded {successful_loads}/{len(directories)} monthly datasets")
    
    if len(all_dataframes) == 0:
        print("❌ No data could be loaded!")
        return None
    
    # Combine all data
    print(f"\n�� Combining {len(all_dataframes)} monthly datasets...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Sort by time
    combined_df = combined_df.sort_values('time').reset_index(drop=True)
    
    print(f"✅ Combined dataset shape: {combined_df.shape}")
    print(f"📅 Time range: {combined_df['time'].min()} to {combined_df['time'].max()}")
    print(f"📍 Unique locations: {combined_df[['latitude', 'longitude']].drop_duplicates().shape[0]}")
    
    return combined_df

def process_combined_data(df):
    """Process the combined dataset through the optimized pipeline."""
    print("\n" + "="*70)
    print("PROCESSING COMBINED DATASET")
    print("="*70)
    
    # Create optimized pipeline instance
    pipeline = OptimizedSolarRadiationPipeline(
        output_dir="data_3years_2018_2020_final",
        use_polars=True,
        enable_profiling=True,
        n_workers=4,  # Reduced workers for memory efficiency
        batch_size=500  # Smaller batch size
    )
    
    start_time = time.time()
    
    try:
        # Step 1: Convert to partitioned Parquet
        print("📊 Converting to partitioned Parquet format...")
        parquet_dir = pipeline.convert_to_parquet_partitioned(
            df, 
            f"ssrd_germany_2018_2020_combined"
        )
        
        # Step 2: Memory-efficient resampling to 15-minute intervals
        print("⏱️  Resampling to 15-minute intervals (memory-efficient)...")
        
        # Process in chunks by year to avoid memory issues
        years = df['time'].dt.year.unique()
        all_resampled_chunks = []
        
        for year in sorted(years):
            print(f"🔄 Processing year {year}...")
            year_df = df[df['time'].dt.year == year].copy()
            
            # Process year in smaller chunks if it's too large
            if len(year_df) > 50000000:  # 50M records threshold
                print(f"📦 Year {year} is large ({len(year_df):,} records), processing in quarters...")
                quarters = year_df['time'].dt.quarter.unique()
                
                for quarter in sorted(quarters):
                    print(f"  📊 Processing Q{quarter} of {year}...")
                    quarter_df = year_df[year_df['time'].dt.quarter == quarter].copy()
                    
                    # Resample quarter
                    quarter_resampled = pipeline.optimized_solar_geometry_aware_interpolation(quarter_df)
                    all_resampled_chunks.append(quarter_resampled)
                    
                    # Clear memory
                    del quarter_df, quarter_resampled
                    import gc
                    gc.collect()
            else:
                # Resample entire year
                year_resampled = pipeline.optimized_solar_geometry_aware_interpolation(year_df)
                all_resampled_chunks.append(year_resampled)
                
                # Clear memory
                del year_df, year_resampled
                import gc
                gc.collect()
        
        # Combine all resampled chunks
        print("🔄 Combining resampled chunks...")
        df_resampled = pd.concat(all_resampled_chunks, ignore_index=True)
        df_resampled = df_resampled.sort_values('time').reset_index(drop=True)
        
        # Clear memory
        del all_resampled_chunks
        import gc
        gc.collect()
        
        # Step 3: Save resampled data
        print("💾 Saving resampled data...")
        resampled_dir = pipeline.convert_to_parquet_partitioned(
            df_resampled,
            f"ssrd_germany_2018_2020_combined_15min"
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate performance metrics
        days_processed = (df['time'].max() - df['time'].min()).days
        time_per_day = total_time / days_processed if days_processed > 0 else 0
        
        print("\n" + "="*70)
        print("📊 PROCESSING RESULTS")
        print("="*70)
        
        print(f"⏱️  Processing Time:")
        print(f"   Total: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"   Total: {total_time/3600:.2f} hours")
        print(f"   Per day: {time_per_day:.2f} seconds")
        
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
        if total_time < 1800:  # Less than 30 minutes
            print(f"   🎉 EXCELLENT: Completed in under 30 minutes")
        elif total_time < 3600:  # Less than 1 hour
            print(f"   GOOD: Completed in under 1 hour")
        elif total_time < 7200:  # Less than 2 hours
            print(f"   ⚠️  ACCEPTABLE: Completed in under 2 hours")
        else:
            print(f"   NEEDS OPTIMIZATION: Took longer than 2 hours")
        
        print("="*70)
        
        return {
            'hourly_parquet_dir': parquet_dir,
            'resampled_parquet_dir': resampled_dir,
            'hourly_shape': df.shape,
            'resampled_shape': df_resampled.shape,
            'processing_time': total_time
        }
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to merge and process the data."""
    print("�� Merge and Process 3-Year Solar Radiation Data")
    print("Combining 36 monthly datasets and processing through pipeline")
    print("This will take 15-45 minutes depending on your system...")
    
    # Step 1: Merge monthly data
    combined_df = merge_monthly_data()
    
    if combined_df is None:
        print("\n❌ Failed to merge monthly data!")
        sys.exit(1)
    
    # Step 2: Process combined data
    results = process_combined_data(combined_df)
    
    if results is None:
        print("\n❌ Failed to process combined data!")
        sys.exit(1)
    
    print("\n🎉 3-year data processing completed successfully!")
    print("📁 Check the 'data_3years_2018_2020_final' directory for results")
    print(f"📊 Final dataset: {results['hourly_shape'][0]:,} hourly records")
    print(f"�� Resampled dataset: {results['resampled_shape'][0]:,} 15-minute records")

if __name__ == "__main__":
    main()