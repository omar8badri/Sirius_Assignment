"""
PV Locations Module
==================

This module provides functionality for creating and managing PV (Photovoltaic) 
asset lookup tables with spatial indexing and irradiance data matching.

Key components:
- Data source loaders (OCF, OSM, BNetzA)
- Spatial indexing (H3, Geohash)
- Spatial join operations
- Lookup table management
"""

from .lookup_table import PVLookupTableBuilder
from .data_sources import OCFQuartzLoader, OSMPVExtractor
from .spatial_indexing import H3Indexer
from .spatial_join import IrradianceMatcher

__version__ = "0.1.0"
__all__ = [
    "PVLookupTableBuilder",
    "OCFQuartzLoader", 
    "OSMPVExtractor",
    "H3Indexer",
    "IrradianceMatcher"
]
