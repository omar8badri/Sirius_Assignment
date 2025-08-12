#!/usr/bin/env python3
"""
Hybrid 15-Minute Resampling - Import 2018 + Process 2019-2020
============================================================

This script imports the already processed 2018 data and only processes 2019-2020
to save time and avoid reprocessing.

Input: 
- Existing 2018 data from monthly_15min_results/
- Hourly data for 2019-2020 from ssrd_germany_2018_2020_combined/
Output: Complete 3-year 15-minute dataset
"""

import sys
import time
import pandas as pd
import polars as pl
from pathlib import Path
import logging
import glob
import json
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent))

from solar_radiation_pipeline_optimized import OptimizedSolarRadiationPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_existing_2018_data(base_dir, output_dir):
    """Import existing 2018 data from monthly results."""
    print("�� Importing existing 2018 data...")
    
    source_dir = base_dir / "monthly_15min_results"
    if not source_dir.exists():
        print("❌ No existing 2018 data found")
        return False
    
    # Find all 2018 monthly directories
    pattern = str(source_dir / "ssrd_germany_2018_*_15min")
    monthly_dirs = glob.glob(pattern)
    
    if not monthly_dirs:
        print("❌ No 2018 monthly data found")
        return False
    
    print(f" Found {len(monthly_dirs)} existing 2018 monthly datasets")
    
    # Create 2018 directory in output
    year_2018_dir = output_dir / "ssrd_germany_2018_15min"
    year_2018_dir.mkdir(exist_ok=True)
    
    # Copy all 2018 data
    for monthly_dir in monthly_dirs:
        monthly_path = Path(monthly_dir)
        dest_path = year_2018_dir / monthly_path.name
        if not dest_path.exists():
            shutil.copytree(monthly_path, dest_path)
            print(f"✅ Copied {monthly_path.name}")
    
    print(f"✅ Successfully imported 2018 data to {year_2018_dir}")
    return True

def load_quarterly_data(year, quarter, parquet_dir):
    """Load hourly data for a specific quarter using Polars."""
    print(f"�� Loading data for {year} Q{quarter}...")
    
    # Calculate months for this quarter
    if quarter == 1:
        months = [1, 2, 3]
    elif quarter == 2:
        months = [4, 5, 6]
    elif quarter == 3:
        months = [7, 8, 9]
    else:  # quarter == 4
        months = [10, 11, 12]
    
    all_parquet_files = []
    
    # Collect all parquet files for the quarter
    for month in months:
        month_path = parquet_dir / f"year={year}/month={month}"
        if month_path.exists():
            parquet_files = list(month_path.glob("*.parquet"))
            all_parquet_files.extend(parquet_files)
    
    if not all_parquet_files:
        print(f"❌ No data found for {year} Q{quarter}")
        return None
    
    print(f"�� Found {len(all_parquet_files)} parquet files for {year} Q{quarter}")
    
    try:
        # Load all quarter data at once
        pl_df = pl.scan_parquet([str(f) for f in all_parquet_files])
        df = pl_df.collect()
        
        # Convert to pandas for processing
        pandas_df = df.to_pandas()
        
        print(f"✅ Loaded {len(pandas_df):,} records for {year} Q{quarter}")
        return pandas_df
        
    except Exception as e:
        print(f"❌ Failed to load {year} Q{quarter}: {e}")
        return None

def process_quarter(year, quarter, pipeline, input_dir, output_dir):
    """Process one quarter and save the resampled data."""
    quarter_key = f"{year}-Q{quarter}"
    
    print(f"\n🔄 Processing {quarter_key}...")
    
    # Load quarterly data
    df = load_quarterly_data(year, quarter, input_dir)
    if df is None:
        return False
    
    try:
        # Resample to 15-minute intervals
        print(f"⏱️  Resampling {quarter_key} to 15-minute intervals...")
        df_resampled = pipeline.optimized_solar_geometry_aware_interpolation(df)
        
        # Save the resampled data for this quarter
        print(f"💾 Saving resampled data for {quarter_key}...")
        quarter_output_dir = pipeline.convert_to_parquet_partitioned(
            df_resampled,
            f"ssrd_germany_{year}_Q{quarter}_15min"
        )
        
        print(f"✅ {quarter_key} completed: {len(df_resampled):,} 15-minute records")
        
        # Clear memory
        del df, df_resampled
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to process {quarter_key}: {e}")
        return False

def combine_all_results(base_dir, output_dir):
    """Combine all results into a final dataset."""
    print("\n🔄 Combining all results...")
    
    # Create final combined directory
    combined_dir = base_dir / "ssrd_germany_2018_2020_combined_15min"
    combined_dir.mkdir(exist_ok=True)
    
    # Find all result directories
    all_dirs = []
    
    # Add 2018 data
    year_2018_dir = output_dir / "ssrd_germany_2018_15min"
    if year_2018_dir.exists():
        all_dirs.extend(list(year_2018_dir.glob("ssrd_germany_2018_*_15min")))
    
    # Add 2019-2020 data
    pattern_2019_2020 = str(output_dir / "ssrd_germany_*_Q*_15min")
    all_dirs.extend([Path(d) for d in glob.glob(pattern_2019_2020)])
    
    print(f"📊 Found {len(all_dirs)} total result directories")
    
    # Create summary
    summary_file = combined_dir / "processing_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("15-Minute Resampled Data Summary\n")
        f.write("================================\n\n")
        f.write(f"Total datasets: {len(all_dirs)}\n")
        f.write("Datasets:\n")
        for result_dir in sorted(all_dirs):
            f.write(f"  - {result_dir.name}\n")
    
    print(f"✅ Summary saved to {summary_file}")
    return True

def main():
    """Main function to import 2018 and process 2019-2020."""
    print("="*70)
    print("HYBRID 15-MINUTE RESAMPLING - IMPORT 2018 + PROCESS 2019-2020")
    print("="*70)
    
    # Setup paths
    base_dir = Path("data_3years_2018_2020_final")
    hourly_dir = base_dir / "ssrd_germany_2018_2020_combined"
    output_dir = base_dir / "hybrid_15min_results"
    
    if not hourly_dir.exists():
        print(f"❌ Hourly data directory not found: {hourly_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    # Step 1: Import existing 2018 data
    print("\n" + "="*50)
    print("STEP 1: IMPORTING EXISTING 2018 DATA")
    print("="*50)
    
    success_2018 = import_existing_2018_data(base_dir, output_dir)
    
    # Step 2: Process 2019-2020 data
    print("\n" + "="*50)
    print("STEP 2: PROCESSING 2019-2020 DATA")
    print("="*50)
    
    # Create optimized pipeline instance
    pipeline = OptimizedSolarRadiationPipeline(
        output_dir=str(output_dir),
        use_polars=True,
        enable_profiling=False,
        n_workers=6,
        batch_size=1000
    )
    
    successful_quarters = 0
    total_quarters = 8  # 2 years × 4 quarters
    
    # Process 2019-2020 quarters
    for year in [2019, 2020]:
        for quarter in range(1, 5):  # 4 quarters
            success = process_quarter(year, quarter, pipeline, hourly_dir, output_dir)
            if success:
                successful_quarters += 1
            
            # Progress update
            progress = (successful_quarters / total_quarters) * 100
            elapsed = time.time() - start_time
            if successful_quarters > 0:
                estimated_total = elapsed * total_quarters / successful_quarters
                remaining = estimated_total - elapsed
                print(f"�� Progress: {successful_quarters}/{total_quarters} ({progress:.1f}%) - Est. remaining: {remaining/60:.1f} min")
    
    # Step 3: Combine all results
    print("\n" + "="*50)
    print("STEP 3: COMBINING ALL RESULTS")
    print("="*50)
    
    combine_all_results(base_dir, output_dir)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    
    print(f"✅ 2018 data imported: {'Yes' if success_2018 else 'No'}")
    print(f"✅ 2019-2020 quarters processed: {successful_quarters}/{total_quarters}")
    print(f"⏱️  Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"📁 Output directory: {output_dir}")
    
    if successful_quarters == total_quarters and success_2018:
        print("\n�� All data processed successfully!")
        print("�� Complete 3-year 15-minute dataset ready!")
    else:
        print(f"\n⚠️  Some data processing failed")
    
    print("="*70)

if __name__ == "__main__":
    main()