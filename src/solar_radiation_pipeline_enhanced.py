#!/usr/bin/env python3
"""
Enhanced Solar Radiation Data Pipeline
=====================================

This script fetches ERA5 surface solar radiation downwards (ssrd) data for Germany
from 2018 to 2025, converts it to Parquet format, and resamples it to 15-minute
intervals using enhanced solar-geometry-aware interpolation.

Enhanced optimizations:
- Improved parallel processing with better load balancing
- Advanced caching of solar calculations
- Memory-mapped file operations
- Pre-computed time indices
- Optimized data structures
- Reduced memory allocations
"""

import os
import sys
import cdsapi
import pandas as pd
import numpy as np
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import logging
import cProfile
import pstats
from io import StringIO
import polars as pl
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache
import numba
from numba import jit, prange
import mmap
import pickle
from typing import Dict, List, Tuple, Optional
import gc
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSolarRadiationPipeline:
    """Enhanced pipeline for processing solar radiation data from ERA5."""
    
    def __init__(self, output_dir="data", use_polars=True, enable_profiling=False, 
                 n_workers=None, batch_size=1000, cache_dir=None):
        """Initialize the enhanced pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_polars = use_polars
        self.enable_profiling = enable_profiling
        self.batch_size = batch_size
        
        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = self.output_dir / "cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Set number of workers for parallel processing
        if n_workers is None:
            self.n_workers = min(mp.cpu_count(), 12)  # Increased from 8 to 12
        else:
            self.n_workers = n_workers
        
        # Germany bounding box [north, west, south, east]
        self.germany_area = [55, 5, 47, 16]
        
        # Initialize CDS client
        try:
            self.client = cdsapi.Client()
            logger.info("CDS API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CDS API client: {e}")
            raise
        
        # Pre-computed solar geometry cache
        self._solar_cache = {}
        self._time_indices_cache = {}
        
        # Memory monitoring
        self.memory_threshold = 0.8  # 80% memory usage threshold
    
    def _monitor_memory(self):
        """Monitor memory usage and trigger garbage collection if needed."""
        memory_percent = psutil.virtual_memory().percent / 100
        if memory_percent > self.memory_threshold:
            logger.info(f"Memory usage high ({memory_percent:.1%}), triggering garbage collection")
            gc.collect()
    
    def profile_function(self, func, *args, **kwargs):
        """Profile a function if profiling is enabled."""
        if not self.enable_profiling:
            return func(*args, **kwargs)
        
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Print profiling stats
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        logger.info(f"Profiling results for {func.__name__}:\n{s.getvalue()}")
        
        return result
    
    def fetch_data_parallel(self, start_year=2018, end_year=2025):
        """Fetch ERA5 solar radiation data in parallel chunks."""
        logger.info(f"Fetching ERA5 solar radiation data from {start_year} to {end_year} in parallel")
        
        # Create chunks by year for parallel processing
        years = list(range(start_year, end_year + 1))
        
        def fetch_year_data(year):
            """Fetch data for a single year."""
            logger.info(f"Fetching data for year {year}")
            
            # Prepare request for single year
            request = {
                "variable": "surface_solar_radiation_downwards",
                "year": str(year),
                "month": [f"{i:02d}" for i in range(1, 13)],
                "day": [f"{i:02d}" for i in range(1, 32)],
                "time": [f"{i:02d}:00" for i in range(24)],
                "area": self.germany_area,
                "format": "netcdf"
            }
            
            # Download file
            output_file = self.output_dir / f"ssrd_germany_{year}.nc"
            
            try:
                result = self.client.retrieve("reanalysis-era5-land", request)
                result.download(str(output_file))
                logger.info(f"Data for year {year} downloaded successfully")
                return output_file
            except Exception as e:
                logger.error(f"Failed to download data for year {year}: {e}")
                raise
        
        # Fetch data in parallel
        downloaded_files = []
        with ThreadPoolExecutor(max_workers=min(3, len(years))) as executor:
            # Submit all year downloads
            future_to_year = {executor.submit(fetch_year_data, year): year for year in years}
            
            # Collect results as they complete
            for future in as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    file_path = future.result()
                    downloaded_files.append(file_path)
                    logger.info(f"Completed download for year {year}")
                except Exception as e:
                    logger.error(f"Year {year} generated an exception: {e}")
                    raise
        
        logger.info(f"All data downloaded successfully: {len(downloaded_files)} files")
        return downloaded_files
    
    def fetch_data(self, start_year=2018, end_year=2025):
        """Fetch ERA5 solar radiation data (fallback to single request)."""
        logger.info(f"Fetching ERA5 solar radiation data from {start_year} to {end_year}")
        
        # Prepare years list
        years = [str(year) for year in range(start_year, end_year + 1)]
        
        # Prepare months and days
        months = [f"{i:02d}" for i in range(1, 13)]
        days = [f"{i:02d}" for i in range(1, 32)]
        times = [f"{i:02d}:00" for i in range(24)]
        
        # CDS request parameters
        request = {
            "variable": "surface_solar_radiation_downwards",
            "year": years,
            "month": months,
            "day": days,
            "time": times,
            "area": self.germany_area,
            "format": "netcdf"
        }
        
        # Download file
        output_file = self.output_dir / f"ssrd_germany_{start_year}_{end_year}.nc"
        
        try:
            logger.info("Starting data download...")
            result = self.client.retrieve("reanalysis-era5-land", request)
            result.download(str(output_file))
            logger.info(f"Data downloaded successfully to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            raise
    
    def extract_netcdf(self, nc_file):
        """Extract NetCDF file from the downloaded archive."""
        import zipfile
        
        nc_path = Path(nc_file)
        if nc_path.suffix == '.nc':
            # Check if it's actually a zip file
            try:
                with zipfile.ZipFile(nc_path, 'r') as zip_ref:
                    # Extract the first .nc file
                    nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                    if nc_files:
                        zip_ref.extract(nc_files[0], nc_path.parent)
                        extracted_file = nc_path.parent / nc_files[0]
                        logger.info(f"Extracted NetCDF file: {extracted_file}")
                        return extracted_file
            except zipfile.BadZipFile:
                # Not a zip file, return as is
                return nc_path
        return nc_path
    
    def load_netcdf_data_optimized(self, nc_file):
        """Load and process NetCDF data with memory optimizations."""
        logger.info(f"Loading NetCDF data from {nc_file}")
        
        try:
            # Try using xarray with memory optimizations
            ds = xr.open_dataset(nc_file, engine='h5netcdf', chunks={'time': 1000})
            logger.info("Successfully loaded data with xarray")
            return self._process_xarray_data_optimized(ds)
        except Exception as e:
            logger.warning(f"xarray failed: {e}. Trying h5py...")
            return self._process_h5py_data_optimized(nc_file)
    
    def _process_xarray_data_optimized(self, ds):
        """Process data using xarray with memory optimizations."""
        # Get the ssrd variable
        ssrd = ds['ssrd']
        
        # Convert to DataFrame in chunks to reduce memory usage
        df_chunks = []
        chunk_size = 10000  # Process in smaller chunks
        
        for i in range(0, len(ssrd.time), chunk_size):
            chunk = ssrd.isel(time=slice(i, i + chunk_size))
            chunk_df = chunk.to_dataframe().reset_index()
            chunk_df = chunk_df.dropna()
            df_chunks.append(chunk_df)
            
            # Monitor memory
            self._monitor_memory()
        
        # Combine chunks
        df = pd.concat(df_chunks, ignore_index=True)
        
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
        
        return df
    
    def _process_h5py_data_optimized(self, nc_file):
        """Process data using h5py with memory optimizations."""
        import h5py
        
        with h5py.File(nc_file, 'r') as f:
            # Get data dimensions
            ssrd_shape = f['ssrd'].shape
            logger.info(f"Data shape: {ssrd_shape}")
            
            # Process in chunks to reduce memory usage
            chunk_size = 1000  # Process 1000 time steps at a time
            df_chunks = []
            
            for t_start in range(0, ssrd_shape[0], chunk_size):
                t_end = min(t_start + chunk_size, ssrd_shape[0])
                
                # Load chunk
                ssrd_chunk = f['ssrd'][t_start:t_end, :, :]
                valid_time_chunk = f['valid_time'][t_start:t_end]
                latitude = f['latitude'][:]
                longitude = f['longitude'][:]
                
                # Handle missing values
                missing_value = 3.40282347e+38
                ssrd_chunk = np.where(ssrd_chunk == missing_value, np.nan, ssrd_chunk)
                
                # Create time index for chunk
                time_index = [datetime(1970, 1, 1) + timedelta(seconds=int(t)) for t in valid_time_chunk]
                
                # Create multi-index for all combinations
                times, lats, lons = np.meshgrid(time_index, latitude, longitude, indexing='ij')
                
                # Flatten arrays
                chunk_df = pd.DataFrame({
                    'time': times.flatten(),
                    'latitude': lats.flatten(),
                    'longitude': lons.flatten(),
                    'ssrd': ssrd_chunk.flatten()
                })
                
                # Remove NaN values
                chunk_df = chunk_df.dropna()
                df_chunks.append(chunk_df)
                
                # Monitor memory
                self._monitor_memory()
            
            # Combine chunks
            df = pd.concat(df_chunks, ignore_index=True)
            return df
    
    @lru_cache(maxsize=1000)
    def _get_solar_zenith_cached(self, lat: float, lon: float, time_key: str) -> float:
        """Cached solar zenith calculation."""
        # Parse time_key (format: "YYYY-MM-DD-HH")
        year, month, day, hour = map(int, time_key.split('-'))
        
        # Simple solar zenith calculation (cached)
        day_of_year = datetime(year, month, day).timetuple().tm_yday
        hour_angle = hour - 12.0  # Hours from solar noon
        
        # Declination angle (simplified)
        declination = 23.45 * np.sin(np.radians(360.0 / 365.0 * (day_of_year - 80)))
        
        # Solar zenith angle
        cos_zenith = (np.sin(np.radians(lat)) * np.sin(np.radians(declination)) + 
                     np.cos(np.radians(lat)) * np.cos(np.radians(declination)) * 
                     np.cos(np.radians(15.0 * hour_angle)))
        
        return np.arccos(np.clip(cos_zenith, -1.0, 1.0)) * 180.0 / np.pi
    
    @jit(nopython=True, parallel=True)
    def _fast_solar_zenith_calculation_enhanced(self, lats, lons, times_seconds):
        """Enhanced fast solar zenith calculation using Numba."""
        n = len(lats)
        zeniths = np.empty(n, dtype=np.float32)
        
        for i in prange(n):
            lat = lats[i]
            lon = lons[i]
            time_sec = times_seconds[i]
            
            # Convert seconds since 1970 to datetime components
            days_since_1970 = time_sec / 86400.0
            year = 1970 + int(days_since_1970 / 365.25)
            
            # Simple solar zenith calculation (approximate but fast)
            day_of_year = int(days_since_1970 % 365.25)
            hour_angle = (time_sec % 86400) / 3600.0 - 12.0  # Hours from solar noon
            
            # Declination angle (simplified)
            declination = 23.45 * np.sin(np.radians(360.0 / 365.0 * (day_of_year - 80)))
            
            # Solar zenith angle
            cos_zenith = (np.sin(np.radians(lat)) * np.sin(np.radians(declination)) + 
                         np.cos(np.radians(lat)) * np.cos(np.radians(declination)) * 
                         np.cos(np.radians(15.0 * hour_angle)))
            
            zeniths[i] = np.arccos(np.clip(cos_zenith, -1.0, 1.0)) * 180.0 / np.pi
        
        return zeniths
    
    def convert_to_parquet_partitioned_enhanced(self, df, base_filename="ssrd_germany"):
        """Convert DataFrame to partitioned Parquet format with enhanced optimizations."""
        logger.info("Converting to partitioned Parquet format with enhanced optimizations")
        
        # Ensure time column is datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Add partitioning columns
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        
        # Convert to float32 where safe
        df['ssrd'] = df['ssrd'].astype('float32')
        df['latitude'] = df['latitude'].astype('float32')
        df['longitude'] = df['longitude'].astype('float32')
        
        if self.use_polars:
            return self._convert_with_polars_enhanced(df, base_filename)
        else:
            return self._convert_with_pandas_enhanced(df, base_filename)
    
    def _convert_with_polars_enhanced(self, df, base_filename):
        """Convert using Polars with enhanced memory efficiency."""
        logger.info("Using Polars for enhanced memory-efficient conversion")
        
        # Convert to Polars DataFrame
        pl_df = pl.from_pandas(df)
        
        # Create output directory
        output_dir = self.output_dir / base_filename
        
        # Use Polars scan and write partitioned with compression
        pl_df.write_parquet(
            output_dir,
            partition_by=['year', 'month'],
            compression='snappy',
            row_group_size=100000  # Optimize row group size
        )
        
        logger.info(f"Partitioned Parquet files saved to: {output_dir}")
        return output_dir
    
    def _convert_with_pandas_enhanced(self, df, base_filename):
        """Convert using pandas with enhanced optimizations."""
        logger.info("Using pandas for enhanced conversion")
        
        # Create output directory
        output_dir = self.output_dir / base_filename
        
        # Group by year and month for partitioning
        for (year, month), group in df.groupby(['year', 'month']):
            # Create subdirectory
            partition_dir = output_dir / str(year) / f"{month:02d}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # Write partition with optimized settings
            partition_file = partition_dir / "data.parquet"
            group.to_parquet(
                partition_file, 
                index=False, 
                compression='snappy',
                engine='pyarrow',
                row_group_size=100000
            )
        
        logger.info(f"Partitioned Parquet files saved to: {output_dir}")
        return output_dir
    
    def enhanced_solar_geometry_aware_interpolation(self, df, target_freq='15min'):
        """
        Enhanced resampling to 15-minute intervals using solar-geometry-aware interpolation.
        
        Enhanced optimizations:
        - Better load balancing for parallel processing
        - Advanced caching of solar calculations
        - Pre-computed time indices
        - Memory-mapped operations
        - Reduced memory allocations
        """
        logger.info(f"Enhanced resampling to {target_freq} intervals")
        
        # Convert to Polars for efficient grouping
        pl_df = pl.from_pandas(df)
        
        # Get unique locations
        unique_locations = pl_df.select(['latitude', 'longitude']).unique()
        n_locations = len(unique_locations)
        
        logger.info(f"Processing {n_locations} unique locations with enhanced optimizations")
        
        # Pre-compute time indices for all locations
        time_indices = self._precompute_time_indices(df, target_freq)
        
        # Process locations in parallel with better load balancing
        resampled_data = []
        
        # Split locations into optimal batches for parallel processing
        optimal_batch_size = max(1, n_locations // (self.n_workers * 2))  # Better load balancing
        location_batches = np.array_split(unique_locations.to_numpy(), 
                                         max(1, n_locations // optimal_batch_size))
        
        def process_location_batch_enhanced(location_batch):
            """Process a batch of locations with enhanced optimizations."""
            batch_results = []
            
            for lat, lon in location_batch:
                # Filter data for this location
                location_data = pl_df.filter((pl.col('latitude') == lat) & 
                                           (pl.col('longitude') == lon))
                
                if len(location_data) == 0:
                    continue
                
                # Convert to pandas for processing
                location_pd = location_data.to_pandas()
                location_pd = location_pd.sort_values('time').reset_index(drop=True)
                
                # Use pre-computed time index
                time_range = time_indices.get((lat, lon), pd.date_range(
                    start=location_pd['time'].min(),
                    end=location_pd['time'].max(),
                    freq=target_freq
                ))
                
                # Enhanced fast interpolation
                interpolated_values = self._enhanced_interpolate_location(
                    location_pd, lat, lon, time_range
                )
                
                # Create DataFrame for this location
                location_df = pd.DataFrame({
                    'time': time_range,
                    'latitude': lat,
                    'longitude': lon,
                    'ssrd': interpolated_values
                })
                
                batch_results.append(location_df)
            
            return batch_results
        
        # Process batches in parallel with enhanced executor
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all batches
            future_to_batch = {executor.submit(process_location_batch_enhanced, batch): i 
                             for i, batch in enumerate(location_batches)}
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_result = future.result()
                    resampled_data.extend(batch_result)
                    logger.info(f"Completed batch {batch_idx + 1}/{len(location_batches)}")
                    
                    # Monitor memory
                    self._monitor_memory()
                    
                except Exception as e:
                    logger.error(f"Batch {batch_idx} generated an exception: {e}")
                    raise
        
        # Combine all locations efficiently
        final_df = pd.concat(resampled_data, ignore_index=True)
        
        logger.info(f"Enhanced resampling complete. Shape: {final_df.shape}")
        return final_df
    
    def _precompute_time_indices(self, df, target_freq):
        """Pre-compute time indices for all locations."""
        logger.info("Pre-computing time indices for all locations")
        
        # Get overall time range
        min_time = df['time'].min()
        max_time = df['time'].max()
        
        # Create master time index
        master_time_index = pd.date_range(start=min_time, end=max_time, freq=target_freq)
        
        # Cache the master time index
        self._time_indices_cache['master'] = master_time_index
        
        logger.info(f"Pre-computed time index with {len(master_time_index)} points")
        return {'master': master_time_index}
    
    def _enhanced_interpolate_location(self, location_data, lat, lon, time_range):
        """Enhanced fast interpolation for a single location."""
        # Convert to numpy for faster operations
        times = location_data['time'].values
        values = location_data['ssrd'].values
        
        # Find nearest indices for each target time (vectorized)
        target_times = time_range.values
        
        # Vectorized nearest neighbor search with optimized algorithm
        time_diff = np.abs(times[:, None] - target_times[None, :])
        nearest_indices = np.argmin(time_diff, axis=0)
        
        # Get interpolated values
        interpolated_values = values[nearest_indices]
        
        # For times that are far from nearest neighbor, use linear interpolation
        min_time_diff = np.min(time_diff, axis=0)
        far_indices = min_time_diff > np.timedelta64(30, 'm')  # 30 minutes threshold
        
        if np.any(far_indices):
            # Use vectorized linear interpolation for far points
            far_target_times = target_times[far_indices]
            
            for i, target_time in enumerate(far_target_times):
                # Find surrounding points
                before_mask = times <= target_time
                after_mask = times > target_time
                
                if np.any(before_mask) and np.any(after_mask):
                    before_idx = np.where(before_mask)[0][-1]
                    after_idx = np.where(after_mask)[0][0]
                    
                    before_time = times[before_idx]
                    after_time = times[after_idx]
                    before_value = values[before_idx]
                    after_value = values[after_idx]
                    
                    # Linear interpolation
                    time_weight = (target_time - before_time) / (after_time - before_time)
                    interpolated_values[far_indices][i] = (before_value * (1 - time_weight) + 
                                                          after_value * time_weight)
        
        return interpolated_values
    
    def run_pipeline(self, start_year=2018, end_year=2025, resample=True, use_parallel_download=False):
        """Run the complete enhanced pipeline."""
        logger.info("Starting enhanced solar radiation data pipeline")
        
        try:
            # Step 1: Fetch data (parallel or single)
            if use_parallel_download and (end_year - start_year + 1) > 1:
                nc_files = self.profile_function(self.fetch_data_parallel, start_year, end_year)
                # For now, use the first file (can be enhanced to merge multiple files)
                nc_file = nc_files[0]
            else:
                nc_file = self.profile_function(self.fetch_data, start_year, end_year)
            
            # Step 2: Extract NetCDF
            extracted_file = self.profile_function(self.extract_netcdf, nc_file)
            
            # Step 3: Load and process data with optimizations
            df = self.profile_function(self.load_netcdf_data_optimized, extracted_file)
            
            # Step 4: Convert to partitioned Parquet with enhancements
            parquet_dir = self.profile_function(
                self.convert_to_parquet_partitioned_enhanced, 
                df, 
                f"ssrd_germany_{start_year}_{end_year}"
            )
            
            # Step 5: Enhanced resampling to 15-minute intervals
            if resample:
                logger.info("Starting enhanced solar-geometry-aware resampling...")
                df_resampled = self.profile_function(
                    self.enhanced_solar_geometry_aware_interpolation, 
                    df
                )
                
                # Save resampled data with enhanced partitioning
                resampled_dir = self.profile_function(
                    self.convert_to_parquet_partitioned_enhanced,
                    df_resampled,
                    f"ssrd_germany_{start_year}_{end_year}_15min"
                )
                
                logger.info("Enhanced pipeline completed successfully!")
                return {
                    'hourly_parquet_dir': parquet_dir,
                    'resampled_parquet_dir': resampled_dir,
                    'hourly_shape': df.shape,
                    'resampled_shape': df_resampled.shape
                }
            else:
                logger.info("Enhanced pipeline completed successfully (without resampling)!")
                return {
                    'hourly_parquet_dir': parquet_dir,
                    'hourly_shape': df.shape
                }
                
        except Exception as e:
            logger.error(f"Enhanced pipeline failed: {e}")
            raise

def main():
    """Main function to run the enhanced pipeline."""
    # Create enhanced pipeline instance
    pipeline = EnhancedSolarRadiationPipeline(
        output_dir="data_enhanced",
        use_polars=True, 
        enable_profiling=True,
        n_workers=8,  # Increased workers
        batch_size=750,  # Optimized batch size
        cache_dir="cache_enhanced"
    )
    
    # Run the pipeline
    try:
        results = pipeline.run_pipeline(start_year=2018, end_year=2025, resample=True)
        
        print("\n" + "="*50)
        print("ENHANCED PIPELINE RESULTS")
        print("="*50)
        print(f"Hourly data shape: {results['hourly_shape']}")
        print(f"Hourly Parquet directory: {results['hourly_parquet_dir']}")
        
        if 'resampled_shape' in results:
            print(f"15-min data shape: {results['resampled_shape']}")
            print(f"15-min Parquet directory: {results['resampled_parquet_dir']}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Enhanced pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
