#!/usr/bin/env python3
"""
Debug script to examine the data structure.
"""

import sys
from pathlib import Path
import pandas as pd
import xarray as xr

# Add src to path
sys.path.append(str(Path(__file__).parent))

def debug_data():
    """Debug the data structure."""
    print("Debugging data structure...")
    
    # Load the NetCDF file
    nc_file = "data_tiny/data_0.nc"
    
    try:
        # Try xarray
        ds = xr.open_dataset(nc_file, engine='h5netcdf')
        print("=== Xarray Dataset Info ===")
        print(ds.info())
        print("\n=== Variables ===")
        print(list(ds.data_vars.keys()))
        print("\n=== Coordinates ===")
        print(list(ds.coords.keys()))
        
        # Get the ssrd variable
        ssrd = ds['ssrd']
        print(f"\n=== SSRD Shape ===")
        print(f"Shape: {ssrd.shape}")
        print(f"Dimensions: {ssrd.dims}")
        
        # Convert to DataFrame
        df = ssrd.to_dataframe().reset_index()
        print(f"\n=== DataFrame Info ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types: {df.dtypes}")
        
        print(f"\n=== Sample Data ===")
        print(df.head())
        
        # Check for time column
        if 'time' in df.columns:
            print(f"\n=== Time Column Info ===")
            print(f"Time range: {df['time'].min()} to {df['time'].max()}")
            print(f"Time type: {df['time'].dtype}")
        else:
            print(f"\n=== No 'time' column found ===")
            print(f"Available columns: {list(df.columns)}")
            
            # Check for other time-related columns
            time_cols = [col for col in df.columns if 'time' in col.lower()]
            print(f"Time-related columns: {time_cols}")
        
        ds.close()
        
    except Exception as e:
        print(f"Error with xarray: {e}")
        
        # Try h5py
        try:
            import h5py
            with h5py.File(nc_file, 'r') as f:
                print("\n=== H5py File Structure ===")
                print("Keys:", list(f.keys()))
                
                for key in f.keys():
                    obj = f[key]
                    print(f"\n{key}: {type(obj)}")
                    if isinstance(obj, h5py.Dataset):
                        print(f"  Shape: {obj.shape}")
                        print(f"  Dtype: {obj.dtype}")
                        if hasattr(obj, 'attrs'):
                            print(f"  Attributes: {dict(obj.attrs)}")
        except Exception as e2:
            print(f"Error with h5py: {e2}")

if __name__ == "__main__":
    debug_data()
