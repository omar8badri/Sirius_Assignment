"""
H3 Spatial Indexer
=================

This module provides H3 hexagonal spatial indexing for PV assets,
enabling efficient spatial joins and queries.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple
import warnings

# Try to import h3, but provide fallback if not available
try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    warnings.warn("H3 library not available. Install with: pip install h3")

logger = logging.getLogger(__name__)

class H3Indexer:
    """H3 hexagonal spatial indexer for PV assets."""
    
    def __init__(self, resolution: int = 9):
        """
        Initialize the H3 indexer.
        
        Args:
            resolution: H3 resolution (0-15). Resolution 9 gives ~173m hexagons
        """
        if not H3_AVAILABLE:
            raise ImportError("H3 library is required. Install with: pip install h3")
        
        self.resolution = resolution
        self._validate_resolution()
        
        logger.info(f"Initialized H3 indexer with resolution {resolution}")
    
    def _validate_resolution(self):
        """Validate H3 resolution."""
        if not 0 <= self.resolution <= 15:
            raise ValueError(f"H3 resolution must be between 0 and 15, got {self.resolution}")
    
    def index_pv_assets(self, pv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Index PV assets with H3 hexagons.
        
        Args:
            pv_df: DataFrame with PV assets (must have 'latitude' and 'longitude' columns)
            
        Returns:
            DataFrame with added 'h3_index' column
        """
        if 'latitude' not in pv_df.columns or 'longitude' not in pv_df.columns:
            raise ValueError("DataFrame must contain 'latitude' and 'longitude' columns")
        
        logger.info(f"Indexing {len(pv_df)} PV assets with H3 resolution {self.resolution}")
        
        # Create a copy to avoid modifying the original
        indexed_df = pv_df.copy()
        
        # Add H3 indices
        indexed_df['h3_index'] = indexed_df.apply(
            lambda row: h3.latlng_to_cell(
                row['latitude'], row['longitude'], self.resolution
            ), axis=1
        )
        
        # Add H3 metadata
        indexed_df['h3_resolution'] = self.resolution
        indexed_df['h3_parent'] = indexed_df['h3_index'].apply(
            lambda x: h3.cell_to_parent(x, self.resolution - 1) if self.resolution > 0 else None
        )
        indexed_df['h3_children'] = indexed_df['h3_index'].apply(
            lambda x: h3.cell_to_children(x, self.resolution + 1) if self.resolution < 15 else []
        )
        
        logger.info(f"Successfully indexed {len(indexed_df)} PV assets")
        return indexed_df
    
    def get_h3_boundaries(self, h3_indices: List[str]) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get H3 hexagon boundaries.
        
        Args:
            h3_indices: List of H3 indices
            
        Returns:
            Dictionary mapping H3 indices to boundary coordinates
        """
        boundaries = {}
        
        for h3_index in h3_indices:
            try:
                boundary = h3.cell_to_boundary(h3_index)
                boundaries[h3_index] = boundary.tolist()
            except Exception as e:
                logger.warning(f"Failed to get boundary for H3 index {h3_index}: {e}")
                boundaries[h3_index] = []
        
        return boundaries
    
    def get_h3_centroid(self, h3_index: str) -> Tuple[float, float]:
        """
        Get H3 hexagon centroid.
        
        Args:
            h3_index: H3 index
            
        Returns:
            Tuple of (latitude, longitude)
        """
        try:
            lat, lng = h3.cell_to_latlng(h3_index)
            return (lat, lng)
        except Exception as e:
            logger.error(f"Failed to get centroid for H3 index {h3_index}: {e}")
            return (0.0, 0.0)
    
    def get_neighboring_h3(self, h3_index: str) -> List[str]:
        """
        Get neighboring H3 hexagons.
        
        Args:
            h3_index: H3 index
            
        Returns:
            List of neighboring H3 indices
        """
        try:
            neighbors = h3.grid_disk(h3_index, 1)
            return [str(n) for n in neighbors if str(n) != h3_index]
        except Exception as e:
            logger.error(f"Failed to get neighbors for H3 index {h3_index}: {e}")
            return []
    
    def group_by_h3(self, pv_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group PV assets by H3 index.
        
        Args:
            pv_df: DataFrame with PV assets (must have 'h3_index' column)
            
        Returns:
            Dictionary mapping H3 indices to DataFrames of PV assets
        """
        if 'h3_index' not in pv_df.columns:
            raise ValueError("DataFrame must contain 'h3_index' column")
        
        grouped = {}
        
        for h3_index, group in pv_df.groupby('h3_index'):
            grouped[str(h3_index)] = group.reset_index(drop=True)
        
        logger.info(f"Grouped {len(pv_df)} PV assets into {len(grouped)} H3 hexagons")
        return grouped
    
    def get_h3_statistics(self, pv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics for each H3 hexagon.
        
        Args:
            pv_df: DataFrame with PV assets (must have 'h3_index' column)
            
        Returns:
            DataFrame with H3 statistics
        """
        if 'h3_index' not in pv_df.columns:
            raise ValueError("DataFrame must contain 'h3_index' column")
        
        stats = pv_df.groupby('h3_index').agg({
            'asset_id': 'count',
            'capacity_kw': ['sum', 'mean', 'min', 'max'],
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        # Flatten column names
        stats.columns = [
            'h3_index', 'asset_count', 'total_capacity_kw', 
            'avg_capacity_kw', 'min_capacity_kw', 'max_capacity_kw',
            'centroid_lat', 'centroid_lon'
        ]
        
        logger.info(f"Generated statistics for {len(stats)} H3 hexagons")
        return stats
    
    def filter_by_h3_region(self, pv_df: pd.DataFrame, 
                           center_h3: str, radius: int = 1) -> pd.DataFrame:
        """
        Filter PV assets within a H3 region.
        
        Args:
            pv_df: DataFrame with PV assets (must have 'h3_index' column)
            center_h3: Center H3 index
            radius: Radius in H3 hexagons
            
        Returns:
            Filtered DataFrame
        """
        if 'h3_index' not in pv_df.columns:
            raise ValueError("DataFrame must contain 'h3_index' column")
        
        try:
            # Get H3 indices within radius
            region_h3 = h3.grid_disk(center_h3, radius)
            region_h3_set = set(str(h) for h in region_h3)
            
            # Filter DataFrame
            filtered_df = pv_df[pv_df['h3_index'].astype(str).isin(region_h3_set)]
            
            logger.info(f"Filtered to {len(filtered_df)} PV assets within H3 region")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Failed to filter by H3 region: {e}")
            return pd.DataFrame()
    
    def get_optimal_resolution(self, pv_df: pd.DataFrame, 
                             target_hexagons: int = 1000) -> int:
        """
        Find optimal H3 resolution for given number of target hexagons.
        
        Args:
            pv_df: DataFrame with PV assets
            target_hexagons: Target number of hexagons
            
        Returns:
            Optimal H3 resolution
        """
        if len(pv_df) == 0:
            return 9  # Default resolution
        
        # Estimate resolution based on data density
        unique_locations = pv_df[['latitude', 'longitude']].drop_duplicates()
        n_locations = len(unique_locations)
        
        # Simple heuristic: resolution 9 gives ~173m hexagons
        # Adjust based on target hexagons vs unique locations
        if n_locations <= target_hexagons:
            # Use higher resolution for fewer locations
            resolution = max(9, 15 - int(np.log2(target_hexagons / n_locations)))
        else:
            # Use lower resolution for more locations
            resolution = min(9, 9 - int(np.log2(n_locations / target_hexagons)))
        
        resolution = max(0, min(15, resolution))
        
        logger.info(f"Estimated optimal H3 resolution: {resolution} "
                   f"(target: {target_hexagons} hexagons, locations: {n_locations})")
        
        return resolution
