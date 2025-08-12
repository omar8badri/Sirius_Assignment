"""
OCF Quartz PV Data Loader
========================

This module loads PV site data from Open Climate Fix's Quartz solar forecasting
project and related repositories.
"""

import pandas as pd
import numpy as np
import requests
import logging
from pathlib import Path
from typing import Optional, Dict, List
import json

logger = logging.getLogger(__name__)

class OCFQuartzLoader:
    """Loader for PV data from Open Climate Fix Quartz project."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the OCF Quartz loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir or Path("cache_ocf_quartz")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Sample PV sites for Germany (for testing)
        self.sample_german_pv_sites = [
            {
                "asset_id": "DE_BERLIN_001",
                "latitude": 52.5200,
                "longitude": 13.4050,
                "capacity_kw": 10.5,
                "installation_date": "2020-01-15",
                "technology": "mono-Si",
                "tilt_angle": 35.0,
                "azimuth": 180.0,
                "data_source": "ocf_quartz_sample"
            },
            {
                "asset_id": "DE_MUNICH_001", 
                "latitude": 48.1351,
                "longitude": 11.5820,
                "capacity_kw": 15.2,
                "installation_date": "2019-06-20",
                "technology": "poly-Si",
                "tilt_angle": 30.0,
                "azimuth": 175.0,
                "data_source": "ocf_quartz_sample"
            },
            {
                "asset_id": "DE_HAMBURG_001",
                "latitude": 53.5511,
                "longitude": 9.9937,
                "capacity_kw": 8.7,
                "installation_date": "2021-03-10",
                "technology": "mono-Si",
                "tilt_angle": 25.0,
                "azimuth": 185.0,
                "data_source": "ocf_quartz_sample"
            },
            {
                "asset_id": "DE_COLOGNE_001",
                "latitude": 50.9375,
                "longitude": 6.9603,
                "capacity_kw": 12.3,
                "installation_date": "2018-11-05",
                "technology": "poly-Si",
                "tilt_angle": 40.0,
                "azimuth": 170.0,
                "data_source": "ocf_quartz_sample"
            },
            {
                "asset_id": "DE_FRANKFURT_001",
                "latitude": 50.1109,
                "longitude": 8.6821,
                "capacity_kw": 9.8,
                "installation_date": "2020-08-12",
                "technology": "mono-Si",
                "tilt_angle": 32.0,
                "azimuth": 180.0,
                "data_source": "ocf_quartz_sample"
            }
        ]
    
    def load_german_pv_sites(self, use_sample: bool = True) -> pd.DataFrame:
        """
        Load German PV sites from OCF Quartz database.
        
        Args:
            use_sample: If True, use sample data for testing
            
        Returns:
            DataFrame with PV site information
        """
        if use_sample:
            logger.info("Using sample German PV sites for testing")
            return self._load_sample_data()
        
        # TODO: Implement actual OCF Quartz API calls
        # For now, return sample data
        logger.warning("Actual OCF Quartz API not implemented yet, using sample data")
        return self._load_sample_data()
    
    def _load_sample_data(self) -> pd.DataFrame:
        """Load sample PV site data."""
        df = pd.DataFrame(self.sample_german_pv_sites)
        
        # Add additional metadata
        df['country'] = 'Germany'
        df['region'] = df['asset_id'].str.split('_').str[1]
        df['created_at'] = pd.Timestamp.now()
        df['updated_at'] = pd.Timestamp.now()
        
        logger.info(f"Loaded {len(df)} sample PV sites")
        return df
    
    def get_pv_sites_by_region(self, region: str) -> pd.DataFrame:
        """
        Get PV sites for a specific region.
        
        Args:
            region: Region name (e.g., 'BERLIN', 'MUNICH')
            
        Returns:
            DataFrame with PV sites in the specified region
        """
        df = self.load_german_pv_sites(use_sample=True)
        return df[df['region'] == region.upper()]
    
    def get_pv_sites_by_capacity_range(self, min_capacity: float, max_capacity: float) -> pd.DataFrame:
        """
        Get PV sites within a capacity range.
        
        Args:
            min_capacity: Minimum capacity in kW
            max_capacity: Maximum capacity in kW
            
        Returns:
            DataFrame with PV sites in the specified capacity range
        """
        df = self.load_german_pv_sites(use_sample=True)
        return df[(df['capacity_kw'] >= min_capacity) & (df['capacity_kw'] <= max_capacity)]
    
    def validate_pv_data(self, df: pd.DataFrame) -> bool:
        """
        Validate PV data for required fields and data types.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'asset_id', 'latitude', 'longitude', 'capacity_kw',
            'installation_date', 'technology', 'tilt_angle', 'azimuth'
        ]
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return False
        
        # Check data types and ranges
        try:
            # Latitude should be between -90 and 90
            if not ((df['latitude'] >= -90) & (df['latitude'] <= 90)).all():
                logger.error("Invalid latitude values found")
                return False
            
            # Longitude should be between -180 and 180
            if not ((df['longitude'] >= -180) & (df['longitude'] <= 180)).all():
                logger.error("Invalid longitude values found")
                return False
            
            # Capacity should be positive
            if not (df['capacity_kw'] > 0).all():
                logger.error("Invalid capacity values found")
                return False
            
            # Tilt angle should be between 0 and 90
            if not ((df['tilt_angle'] >= 0) & (df['tilt_angle'] <= 90)).all():
                logger.error("Invalid tilt angle values found")
                return False
            
            # Azimuth should be between 0 and 360
            if not ((df['azimuth'] >= 0) & (df['azimuth'] <= 360)).all():
                logger.error("Invalid azimuth values found")
                return False
                
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
        
        logger.info("PV data validation passed")
        return True
