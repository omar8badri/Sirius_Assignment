"""
Irradiance Matcher
=================

This module provides spatial join functionality for matching PV assets
to irradiance data pixels from ERA5 or other sources.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

class IrradianceMatcher:
    """Matcher for PV assets to irradiance data pixels."""
    
    def __init__(self, irradiance_grid_resolution: float = 0.1):
        """
        Initialize the irradiance matcher.
        
        Args:
            irradiance_grid_resolution: Resolution of irradiance grid in degrees (default: 0.1° for ERA5)
        """
        self.grid_resolution = irradiance_grid_resolution
        self._grid_cache = {}
        
        logger.info(f"Initialized irradiance matcher with grid resolution {self.grid_resolution}°")
    
    def find_nearest_irradiance_pixel(self, pv_lat: float, pv_lon: float) -> Dict:
        """
        Find nearest irradiance pixel for a PV asset.
        
        Args:
            pv_lat: PV asset latitude
            pv_lon: PV asset longitude
            
        Returns:
            Dictionary with nearest pixel information
        """
        # Snap to nearest grid point
        grid_lat = self._snap_to_grid(pv_lat, 'lat')
        grid_lon = self._snap_to_grid(pv_lon, 'lon')
        
        # Calculate distance to grid point
        distance_km = self._calculate_distance(pv_lat, pv_lon, grid_lat, grid_lon)
        
        return {
            'nearest_lat': grid_lat,
            'nearest_lon': grid_lon,
            'distance_km': distance_km,
            'grid_resolution': self.grid_resolution,
            'era5_grid_id': f"germany_{grid_lat:.1f}_{grid_lon:.1f}"
        }
    
    def match_pv_to_irradiance_grid(self, pv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match all PV assets to irradiance grid.
        
        Args:
            pv_df: DataFrame with PV assets (must have 'latitude' and 'longitude' columns)
            
        Returns:
            DataFrame with added irradiance matching columns
        """
        if 'latitude' not in pv_df.columns or 'longitude' not in pv_df.columns:
            raise ValueError("DataFrame must contain 'latitude' and 'longitude' columns")
        
        logger.info(f"Matching {len(pv_df)} PV assets to irradiance grid")
        
        # Create a copy to avoid modifying the original
        matched_df = pv_df.copy()
        
        # Apply matching to each row
        matching_results = []
        
        for idx, row in matched_df.iterrows():
            result = self.find_nearest_irradiance_pixel(row['latitude'], row['longitude'])
            matching_results.append(result)
        
        # Add matching results to DataFrame
        for key in matching_results[0].keys():
            matched_df[f'irradiance_{key}'] = [result[key] for result in matching_results]
        
        logger.info(f"Successfully matched {len(matched_df)} PV assets to irradiance grid")
        return matched_df
    
    def _snap_to_grid(self, coord: float, coord_type: str) -> float:
        """
        Snap coordinate to nearest grid point.
        
        Args:
            coord: Coordinate value
            coord_type: 'lat' or 'lon'
            
        Returns:
            Snapped coordinate
        """
        if coord_type == 'lat':
            # Latitude: snap to nearest 0.1 degree
            return round(coord / self.grid_resolution) * self.grid_resolution
        elif coord_type == 'lon':
            # Longitude: snap to nearest 0.1 degree
            return round(coord / self.grid_resolution) * self.grid_resolution
        else:
            raise ValueError(f"Invalid coordinate type: {coord_type}")
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points using Haversine formula.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in kilometers
        earth_radius = 6371.0
        
        return earth_radius * c
    
    def get_grid_statistics(self, matched_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics for irradiance grid matching.
        
        Args:
            matched_df: DataFrame with irradiance matching results
            
        Returns:
            DataFrame with grid statistics
        """
        if 'irradiance_nearest_lat' not in matched_df.columns:
            raise ValueError("DataFrame must contain irradiance matching columns")
        
        # Group by grid points
        grid_stats = matched_df.groupby(['irradiance_nearest_lat', 'irradiance_nearest_lon']).agg({
            'asset_id': 'count',
            'capacity_kw': ['sum', 'mean', 'min', 'max'],
            'irradiance_distance_km': ['mean', 'max']
        }).reset_index()
        
        # Flatten column names
        grid_stats.columns = [
            'grid_lat', 'grid_lon', 'pv_count', 'total_capacity_kw',
            'avg_capacity_kw', 'min_capacity_kw', 'max_capacity_kw',
            'avg_distance_km', 'max_distance_km'
        ]
        
        logger.info(f"Generated statistics for {len(grid_stats)} irradiance grid points")
        return grid_stats
    
    def validate_matching_quality(self, matched_df: pd.DataFrame, 
                                max_distance_km: float = 10.0) -> Dict:
        """
        Validate the quality of spatial matching.
        
        Args:
            matched_df: DataFrame with irradiance matching results
            max_distance_km: Maximum acceptable distance in kilometers
            
        Returns:
            Dictionary with validation results
        """
        if 'irradiance_distance_km' not in matched_df.columns:
            raise ValueError("DataFrame must contain 'irradiance_distance_km' column")
        
        total_assets = len(matched_df)
        close_matches = len(matched_df[matched_df['irradiance_distance_km'] <= max_distance_km])
        far_matches = total_assets - close_matches
        
        avg_distance = matched_df['irradiance_distance_km'].mean()
        max_distance = matched_df['irradiance_distance_km'].max()
        
        validation_results = {
            'total_assets': total_assets,
            'close_matches': close_matches,
            'far_matches': far_matches,
            'close_match_percentage': (close_matches / total_assets) * 100,
            'avg_distance_km': avg_distance,
            'max_distance_km': max_distance,
            'quality_score': max(0, 100 - (avg_distance / max_distance_km) * 100)
        }
        
        logger.info(f"Matching quality validation: {validation_results}")
        return validation_results
    
    def get_optimal_grid_resolution(self, pv_df: pd.DataFrame, 
                                  target_avg_distance_km: float = 5.0) -> float:
        """
        Find optimal grid resolution for target average distance.
        
        Args:
            pv_df: DataFrame with PV assets
            target_avg_distance_km: Target average distance in kilometers
            
        Returns:
            Optimal grid resolution in degrees
        """
        if len(pv_df) == 0:
            return 0.1  # Default resolution
        
        # Simple heuristic: 0.1° ≈ 11km at equator
        # Adjust based on target distance
        current_resolution = 0.1
        current_distance = 11.0  # Approximate distance for 0.1° resolution
        
        optimal_resolution = current_resolution * (current_distance / target_avg_distance_km)
        
        # Clamp to reasonable range
        optimal_resolution = max(0.01, min(1.0, optimal_resolution))
        
        logger.info(f"Estimated optimal grid resolution: {optimal_resolution:.3f}° "
                   f"(target distance: {target_avg_distance_km}km)")
        
        return optimal_resolution
    
    def create_grid_lookup_table(self, matched_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a compact lookup table for grid points.
        
        Args:
            matched_df: DataFrame with irradiance matching results
            
        Returns:
            Compact lookup table
        """
        if 'irradiance_nearest_lat' not in matched_df.columns:
            raise ValueError("DataFrame must contain irradiance matching columns")
        
        # Create lookup table
        lookup_table = matched_df.groupby([
            'irradiance_nearest_lat', 'irradiance_nearest_lon', 'irradiance_era5_grid_id'
        ]).agg({
            'asset_id': list,
            'capacity_kw': 'sum',
            'irradiance_distance_km': 'mean'
        }).reset_index()
        
        # Rename columns
        lookup_table.columns = [
            'grid_lat', 'grid_lon', 'era5_grid_id', 'pv_asset_ids', 
            'total_capacity_kw', 'avg_distance_km'
        ]
        
        # Add metadata
        lookup_table['pv_count'] = lookup_table['pv_asset_ids'].apply(len)
        lookup_table['created_at'] = pd.Timestamp.now()
        
        logger.info(f"Created lookup table with {len(lookup_table)} grid points")
        return lookup_table
