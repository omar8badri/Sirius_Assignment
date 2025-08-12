"""
Climate Data Store API (cdsapi) wrapper.
This module provides a convenient interface to the Climate Data Store API.
"""

import cdsapi

# Create a client instance
def create_client():
    """Create and return a CDS API client."""
    return cdsapi.Client()

# Convenience function for common operations
def download_data(dataset, variable, year, month, area, output_file):
    """
    Download climate data from CDS.
    
    Parameters:
    -----------
    dataset : str
        Dataset name (e.g., 'reanalysis-era5-single-levels')
    variable : str
        Variable name (e.g., '2m_temperature')
    year : str or list
        Year(s) to download
    month : str or list
        Month(s) to download
    area : list
        Area bounds [north, west, south, east]
    output_file : str
        Output file path
    """
    client = create_client()
    
    client.retrieve(
        dataset,
        {
            'variable': variable,
            'year': year,
            'month': month,
            'day': [f"{i:02d}" for i in range(1, 32)],
            'time': [f"{i:02d}:00" for i in range(24)],
            'area': area,
            'format': 'netcdf',
        },
        output_file
    )
    
    return output_file

# Export the main client for direct use
Client = cdsapi.Client
