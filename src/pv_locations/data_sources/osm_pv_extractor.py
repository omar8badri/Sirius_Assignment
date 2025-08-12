"""
OSM PV Data Extractor
====================

This module extracts PV (solar panel) installations from OpenStreetMap data
using the Overpass API.
"""

import pandas as pd
import numpy as np
import requests
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import time
import json

logger = logging.getLogger(__name__)

class OSMPVExtractor:
    """Extractor for PV installations from OpenStreetMap."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the OSM PV extractor.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir or Path("cache_osm_pv")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.overpass_url = "https://overpass-api.de/api/interpreter"
        
        # Sample OSM PV data for Germany (for testing)
        self.sample_osm_pv_sites = [
            {
                "osm_id": "123456789",
                "asset_id": "OSM_DE_BERLIN_001",
                "latitude": 52.5200,
                "longitude": 13.4050,
                "capacity_kw": 12.0,
                "installation_date": "2020-01-15",
                "technology": "mono-Si",
                "tilt_angle": 35.0,
                "azimuth": 180.0,
                "data_source": "osm_sample",
                "osm_tags": {
                    "power": "generator",
                    "generator:source": "solar",
                    "generator:type": "solar_photovoltaic_panel"
                }
            },
            {
                "osm_id": "987654321",
                "asset_id": "OSM_DE_MUNICH_001",
                "latitude": 48.1351,
                "longitude": 11.5820,
                "capacity_kw": 18.5,
                "installation_date": "2019-06-20",
                "technology": "poly-Si",
                "tilt_angle": 30.0,
                "azimuth": 175.0,
                "data_source": "osm_sample",
                "osm_tags": {
                    "power": "generator",
                    "generator:source": "solar",
                    "generator:type": "solar_photovoltaic_panel"
                }
            }
        ]
    
    def extract_german_pv_sites(self, use_sample: bool = True, 
                               bbox: Optional[Tuple[float, float, float, float]] = None) -> pd.DataFrame:
        """
        Extract German PV sites from OpenStreetMap.
        
        Args:
            use_sample: If True, use sample data for testing
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon) for Germany
            
        Returns:
            DataFrame with PV site information
        """
        if use_sample:
            logger.info("Using sample OSM PV sites for testing")
            return self._load_sample_data()
        
        if bbox is None:
            # Default bounding box for Germany
            bbox = (47.0, 5.0, 55.0, 15.0)
        
        logger.info(f"Extracting OSM PV sites from bbox: {bbox}")
        
        try:
            # TODO: Implement actual Overpass API calls
            # For now, return sample data
            logger.warning("Actual OSM extraction not implemented yet, using sample data")
            return self._load_sample_data()
            
        except Exception as e:
            logger.error(f"Failed to extract OSM PV sites: {e}")
            return pd.DataFrame()
    
    def _load_sample_data(self) -> pd.DataFrame:
        """Load sample OSM PV site data."""
        df = pd.DataFrame(self.sample_osm_pv_sites)
        
        # Add additional metadata
        df['country'] = 'Germany'
        df['region'] = df['asset_id'].str.split('_').str[2]
        df['created_at'] = pd.Timestamp.now()
        df['updated_at'] = pd.Timestamp.now()
        
        logger.info(f"Loaded {len(df)} sample OSM PV sites")
        return df
    
    def _build_overpass_query(self, bbox: Tuple[float, float, float, float]) -> str:
        """
        Build Overpass API query for solar PV installations.
        
        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
            
        Returns:
            Overpass API query string
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        
        query = f"""
        [out:json][timeout:300];
        (
          way["power"="generator"]["generator:source"="solar"]({min_lat},{min_lon},{max_lat},{max_lon});
          node["power"="generator"]["generator:source"="solar"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["power"="generator"]["generator:source"="solar"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """
        
        return query
    
    def _parse_osm_response(self, response_data: Dict) -> pd.DataFrame:
        """
        Parse Overpass API response into DataFrame.
        
        Args:
            response_data: JSON response from Overpass API
            
        Returns:
            DataFrame with parsed PV site data
        """
        pv_sites = []
        
        for element in response_data.get('elements', []):
            if element.get('type') in ['node', 'way', 'relation']:
                tags = element.get('tags', {})
                
                # Check if it's a solar PV installation
                if (tags.get('power') == 'generator' and 
                    tags.get('generator:source') == 'solar'):
                    
                    # Extract coordinates
                    if element.get('type') == 'node':
                        lat = element.get('lat')
                        lon = element.get('lon')
                    else:
                        # For ways and relations, we'd need to calculate centroid
                        # For now, skip these
                        continue
                    
                    if lat is not None and lon is not None:
                        pv_site = {
                            'osm_id': str(element.get('id')),
                            'asset_id': f"OSM_{element.get('id')}",
                            'latitude': lat,
                            'longitude': lon,
                            'capacity_kw': self._extract_capacity(tags),
                            'installation_date': self._extract_date(tags),
                            'technology': self._extract_technology(tags),
                            'tilt_angle': self._extract_tilt_angle(tags),
                            'azimuth': self._extract_azimuth(tags),
                            'data_source': 'osm',
                            'osm_tags': tags
                        }
                        pv_sites.append(pv_site)
        
        return pd.DataFrame(pv_sites)
    
    def _extract_capacity(self, tags: Dict) -> float:
        """Extract capacity from OSM tags."""
        # Try different capacity tags
        capacity_tags = ['generator:capacity', 'capacity', 'power']
        
        for tag in capacity_tags:
            if tag in tags:
                try:
                    capacity_str = str(tags[tag])
                    # Remove units and convert to kW
                    if 'MW' in capacity_str.upper():
                        return float(capacity_str.replace('MW', '').replace('mw', '')) * 1000
                    elif 'KW' in capacity_str.upper():
                        return float(capacity_str.replace('KW', '').replace('kw', ''))
                    else:
                        return float(capacity_str)
                except (ValueError, TypeError):
                    continue
        
        return 10.0  # Default capacity
    
    def _extract_date(self, tags: Dict) -> str:
        """Extract installation date from OSM tags."""
        date_tags = ['start_date', 'opening_date', 'construction_date']
        
        for tag in date_tags:
            if tag in tags:
                return str(tags[tag])
        
        return "2020-01-01"  # Default date
    
    def _extract_technology(self, tags: Dict) -> str:
        """Extract technology from OSM tags."""
        if 'generator:type' in tags:
            tech = tags['generator:type']
            if 'mono' in tech.lower():
                return 'mono-Si'
            elif 'poly' in tech.lower():
                return 'poly-Si'
            elif 'thin' in tech.lower():
                return 'thin-film'
        
        return 'mono-Si'  # Default technology
    
    def _extract_tilt_angle(self, tags: Dict) -> float:
        """Extract tilt angle from OSM tags."""
        if 'tilt' in tags:
            try:
                return float(tags['tilt'])
            except (ValueError, TypeError):
                pass
        
        return 35.0  # Default tilt angle
    
    def _extract_azimuth(self, tags: Dict) -> float:
        """Extract azimuth from OSM tags."""
        if 'azimuth' in tags:
            try:
                return float(tags['azimuth'])
            except (ValueError, TypeError):
                pass
        
        return 180.0  # Default azimuth (south-facing)
    
    def get_pv_sites_by_region(self, region: str) -> pd.DataFrame:
        """
        Get PV sites for a specific region.
        
        Args:
            region: Region name
            
        Returns:
            DataFrame with PV sites in the specified region
        """
        df = self.extract_german_pv_sites(use_sample=True)
        return df[df['region'] == region.upper()]
