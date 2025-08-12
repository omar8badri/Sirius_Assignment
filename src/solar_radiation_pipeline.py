#!/usr/bin/env python3
"""
Solar Radiation Data Pipeline
=============================

This script fetches ERA5 surface solar radiation downwards (ssrd) data for Germany
from 2018 to 2025, converts it to Parquet format, and resamples it to 15-minute
intervals using solar-geometry-aware interpolation.

Requirements:
- CDS API credentials configured
- Required packages: cdsapi, pandas, numpy, xarray, pyarrow, ephem, polars, pvlib
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
import ephem
import warnings
from pathlib import Path
import logging
import cProfile
import pstats
from io import StringIO
import polars as pl
from pvlib import solarposition, atmosphere
from pvlib.location import Location

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SolarRadiationPipeline:
    """Pipeline for processing solar radiation data from ERA5."""
    
    def __init__(self, output_dir="data", use_polars=True, enable_profiling=False):
        """Initialize the pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_polars = use_polars
        self.enable_profiling = enable_profiling
        
        # Germany bounding box [north, west, south, east]
        self.germany_area = [55, 5, 47, 16]
        
        # Initialize CDS client
        try:
            self.client = cdsapi.Client()
            logger.info("CDS API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CDS API client: {e}")
            raise
    
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
    
    def fetch_data(self, start_year=2018, end_year=2025):
        """Fetch ERA5 solar radiation data for the specified years."""
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
    
    def load_netcdf_data(self, nc_file):
        """Load and process NetCDF data."""
        logger.info(f"Loading NetCDF data from {nc_file}")
        
        try:
            # Try using xarray first
            ds = xr.open_dataset(nc_file, engine='h5netcdf')
            logger.info("Successfully loaded data with xarray")
            return self._process_xarray_data(ds)
        except Exception as e:
            logger.warning(f"xarray failed: {e}. Trying h5py...")
            return self._process_h5py_data(nc_file)
    
    def _process_xarray_data(self, ds):
        """Process data using xarray."""
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
        
        return df
    
    def _process_h5py_data(self, nc_file):
        """Process data using h5py."""
        import h5py
        
        with h5py.File(nc_file, 'r') as f:
            # Get data
            ssrd_data = f['ssrd'][:]
            valid_time = f['valid_time'][:]
            latitude = f['latitude'][:]
            longitude = f['longitude'][:]
            
            # Handle missing values
            missing_value = 3.40282347e+38
            ssrd_data = np.where(ssrd_data == missing_value, np.nan, ssrd_data)
            
            # Create time index
            time_index = [datetime(1970, 1, 1) + timedelta(seconds=int(t)) for t in valid_time]
            
            # Create multi-index for all combinations
            times, lats, lons = np.meshgrid(time_index, latitude, longitude, indexing='ij')
            
            # Flatten arrays
            df = pd.DataFrame({
                'time': times.flatten(),
                'latitude': lats.flatten(),
                'longitude': lons.flatten(),
                'ssrd': ssrd_data.flatten()
            })
            
            # Remove NaN values
            df = df.dropna()
            
            return df
    
    def convert_to_parquet_partitioned(self, df, base_filename="ssrd_germany"):
        """Convert DataFrame to partitioned Parquet format with year/month partitioning."""
        logger.info("Converting to partitioned Parquet format")
        
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
            return self._convert_with_polars(df, base_filename)
        else:
            return self._convert_with_pandas(df, base_filename)
    
    def _convert_with_polars(self, df, base_filename):
        """Convert using Polars for memory efficiency."""
        logger.info("Using Polars for memory-efficient conversion")
        
        # Convert to Polars DataFrame
        pl_df = pl.from_pandas(df)
        
        # Create output directory
        output_dir = self.output_dir / base_filename
        
        # Use Polars scan and write partitioned
        pl_df.write_parquet(
            output_dir,
            partition_by=['year', 'month'],
            compression='snappy'
        )
        
        logger.info(f"Partitioned Parquet files saved to: {output_dir}")
        return output_dir
    
    def _convert_with_pandas(self, df, base_filename):
        """Convert using pandas (fallback method)."""
        logger.info("Using pandas for conversion")
        
        # Create output directory
        output_dir = self.output_dir / base_filename
        
        # Group by year and month for partitioning
        for (year, month), group in df.groupby(['year', 'month']):
            # Create subdirectory
            partition_dir = output_dir / str(year) / f"{month:02d}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # Write partition
            partition_file = partition_dir / "data.parquet"
            group.to_parquet(partition_file, index=False, compression='snappy')
        
        logger.info(f"Partitioned Parquet files saved to: {output_dir}")
        return output_dir
    
    def calculate_solar_geometry_pvlib(self, lat, lon, time):
        """Calculate solar geometry using pvlib for better accuracy."""
        # Create location object
        location = Location(lat, lon, tz='UTC')
        
        # Calculate solar position
        solar_pos = solarposition.get_solarposition(
            time, 
            location.latitude, 
            location.longitude,
            location.altitude
        )
        
        # Extract angles
        solar_zenith = solar_pos['zenith'].iloc[0]
        solar_azimuth = solar_pos['azimuth'].iloc[0]
        solar_elevation = solar_pos['elevation'].iloc[0]
        
        return solar_zenith, solar_azimuth, solar_elevation
    
    def estimate_tilt_orientation(self, lat, lon):
        """Estimate tilt and orientation for Germany."""
        # Simple estimation based on latitude and country-level statistics
        # Germany average rooftop tilt: ~30-35 degrees
        # Average orientation: South-facing (180 degrees)
        
        # Latitude-based tilt estimation (optimal tilt ≈ latitude * 0.76 + 3.1)
        optimal_tilt = lat * 0.76 + 3.1
        
        # Clamp to reasonable range
        tilt = np.clip(optimal_tilt, 10, 60)
        
        # Germany average orientation (south-facing with some variation)
        orientation = 180  # South-facing
        
        return tilt, orientation
    
    def solar_geometry_aware_interpolation(self, df, target_freq='15T'):
        """
        Resample data to 15-minute intervals using solar-geometry-aware interpolation.
        
        This method considers the solar geometry to provide more accurate
        interpolation for solar radiation data.
        """
        logger.info(f"Resampling to {target_freq} intervals with solar geometry awareness")
        
        if self.use_polars:
            return self._resample_with_polars(df, target_freq)
        else:
            return self._resample_with_pandas(df, target_freq)
    
    def _resample_with_polars(self, df, target_freq):
        """Resample using Polars for memory efficiency."""
        logger.info("Using Polars for memory-efficient resampling")
        
        # Convert to Polars
        pl_df = pl.from_pandas(df)
        
        # Add time components for partitioning
        pl_df = pl_df.with_columns([
            pl.col('time').dt.year().alias('year'),
            pl.col('time').dt.month().alias('month'),
            pl.col('time').dt.hour().alias('hour'),
            pl.col('time').dt.minute().alias('minute')
        ])
        
        # Group by location and process each group
        resampled_data = []
        
        # Get unique locations
        unique_locations = pl_df.select(['latitude', 'longitude']).unique()
        
        for row in unique_locations.iter_rows():
            lat, lon = row
            logger.info(f"Processing location: {lat:.2f}°N, {lon:.2f}°E")
            
            # Filter data for this location
            group = pl_df.filter((pl.col('latitude') == lat) & (pl.col('longitude') == lon))
            
            # Convert back to pandas for solar calculations
            group_pd = group.to_pandas()
            
            # Sort by time
            group_pd = group_pd.sort_values('time').reset_index(drop=True)
            
            # Create 15-minute time index
            time_range = pd.date_range(
                start=group_pd['time'].min(),
                end=group_pd['time'].max(),
                freq=target_freq
            )
            
            # Interpolate with solar geometry awareness
            interpolated_values = []
            
            for target_time in time_range:
                value = self._interpolate_with_solar_geometry(
                    group_pd, lat, lon, target_time
                )
                interpolated_values.append(value)
            
            # Create DataFrame for this location
            location_df = pd.DataFrame({
                'time': time_range,
                'latitude': lat,
                'longitude': lon,
                'ssrd': interpolated_values
            })
            
            resampled_data.append(location_df)
        
        # Combine all locations
        final_df = pd.concat(resampled_data, ignore_index=True)
        
        logger.info(f"Resampling complete. Shape: {final_df.shape}")
        return final_df
    
    def _resample_with_pandas(self, df, target_freq):
        """Resample using pandas (original method)."""
        logger.info("Using pandas for resampling")
        
        # Group by location (latitude, longitude)
        grouped = df.groupby(['latitude', 'longitude'])
        
        resampled_data = []
        
        for (lat, lon), group in grouped:
            logger.info(f"Processing location: {lat:.2f}°N, {lon:.2f}°E")
            
            # Sort by time
            group = group.sort_values('time').reset_index(drop=True)
            
            # Create 15-minute time index for this location
            time_range = pd.date_range(
                start=group['time'].min(),
                end=group['time'].max(),
                freq=target_freq
            )
            
            # Interpolate using solar geometry awareness
            interpolated_values = []
            
            for target_time in time_range:
                value = self._interpolate_with_solar_geometry(
                    group, lat, lon, target_time
                )
                interpolated_values.append(value)
            
            # Create DataFrame for this location
            location_df = pd.DataFrame({
                'time': time_range,
                'latitude': lat,
                'longitude': lon,
                'ssrd': interpolated_values
            })
            
            resampled_data.append(location_df)
        
        # Combine all locations
        final_df = pd.concat(resampled_data, ignore_index=True)
        
        logger.info(f"Resampling complete. Shape: {final_df.shape}")
        return final_df
    
    def _interpolate_with_solar_geometry(self, group, lat, lon, target_time):
        """Interpolate a single time point using solar geometry awareness."""
        # Find the two nearest hourly values
        time_diff = abs(group['time'] - target_time)
        nearest_idx = time_diff.idxmin()
        
        if time_diff[nearest_idx] <= timedelta(minutes=30):
            # Use the nearest value if within 30 minutes
            return group.loc[nearest_idx, 'ssrd']
        
        # Interpolate between two nearest values
        before_times = group[group['time'] <= target_time]
        after_times = group[group['time'] > target_time]
        
        if len(before_times) > 0 and len(after_times) > 0:
            before_time = before_times['time'].iloc[-1]
            after_time = after_times['time'].iloc[0]
            before_value = before_times['ssrd'].iloc[-1]
            after_value = after_times['ssrd'].iloc[0]
            
            # Calculate solar geometry weights using pvlib
            try:
                before_zenith, _, _ = self.calculate_solar_geometry_pvlib(lat, lon, before_time)
                after_zenith, _, _ = self.calculate_solar_geometry_pvlib(lat, lon, after_time)
                target_zenith, _, _ = self.calculate_solar_geometry_pvlib(lat, lon, target_time)
                
                # Weight by solar zenith angle (closer zenith = higher weight)
                before_weight = 1 / (1 + abs(before_zenith - target_zenith))
                after_weight = 1 / (1 + abs(after_zenith - target_zenith))
                
                # Normalize weights
                total_weight = before_weight + after_weight
                before_weight /= total_weight
                after_weight /= total_weight
                
                # Interpolate
                return before_value * before_weight + after_value * after_weight
            except Exception as e:
                logger.warning(f"Solar geometry calculation failed: {e}. Using linear interpolation.")
                # Fallback to linear interpolation
                time_weight = (target_time - before_time) / (after_time - before_time)
                return before_value * (1 - time_weight) + after_value * time_weight
        else:
            # Use nearest value if interpolation not possible
            return group.loc[nearest_idx, 'ssrd']
    
    def run_pipeline(self, start_year=2018, end_year=2025, resample=True):
        """Run the complete pipeline."""
        logger.info("Starting solar radiation data pipeline")
        
        try:
            # Step 1: Fetch data
            nc_file = self.profile_function(self.fetch_data, start_year, end_year)
            
            # Step 2: Extract NetCDF
            extracted_file = self.profile_function(self.extract_netcdf, nc_file)
            
            # Step 3: Load and process data
            df = self.profile_function(self.load_netcdf_data, extracted_file)
            
            # Step 4: Convert to partitioned Parquet
            parquet_dir = self.profile_function(
                self.convert_to_parquet_partitioned, 
                df, 
                f"ssrd_germany_{start_year}_{end_year}"
            )
            
            # Step 5: Resample to 15-minute intervals
            if resample:
                logger.info("Starting solar-geometry-aware resampling...")
                df_resampled = self.profile_function(
                    self.solar_geometry_aware_interpolation, 
                    df
                )
                
                # Save resampled data with partitioning
                resampled_dir = self.profile_function(
                    self.convert_to_parquet_partitioned,
                    df_resampled,
                    f"ssrd_germany_{start_year}_{end_year}_15min"
                )
                
                logger.info("Pipeline completed successfully!")
                return {
                    'hourly_parquet_dir': parquet_dir,
                    'resampled_parquet_dir': resampled_dir,
                    'hourly_shape': df.shape,
                    'resampled_shape': df_resampled.shape
                }
            else:
                logger.info("Pipeline completed successfully (without resampling)!")
                return {
                    'hourly_parquet_dir': parquet_dir,
                    'hourly_shape': df.shape
                }
                
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main function to run the pipeline."""
    # Create pipeline instance with profiling enabled
    pipeline = SolarRadiationPipeline(
        output_dir="data", 
        use_polars=True, 
        enable_profiling=True
    )
    
    # Run the pipeline
    try:
        results = pipeline.run_pipeline(start_year=2018, end_year=2025, resample=True)
        
        print("\n" + "="*50)
        print("PIPELINE RESULTS")
        print("="*50)
        print(f"Hourly data shape: {results['hourly_shape']}")
        print(f"Hourly Parquet directory: {results['hourly_parquet_dir']}")
        
        if 'resampled_shape' in results:
            print(f"15-min data shape: {results['resampled_shape']}")
            print(f"15-min Parquet directory: {results['resampled_parquet_dir']}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
