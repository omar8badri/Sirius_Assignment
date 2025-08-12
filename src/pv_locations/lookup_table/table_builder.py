"""
PV Lookup Table Builder
======================

This module provides the main builder for creating PV asset lookup tables
with spatial indexing and irradiance data matching.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import json

# Import our modules
from ..data_sources import OCFQuartzLoader, OSMPVExtractor
from ..spatial_indexing import H3Indexer
from ..spatial_join import IrradianceMatcher

logger = logging.getLogger(__name__)

class PVLookupTableBuilder:
    """Main builder for PV asset lookup tables."""
    
    def __init__(self, output_dir: Path, h3_resolution: int = 9, 
                 irradiance_grid_resolution: float = 0.1, n_workers: int = 4):
        """
        Initialize the PV lookup table builder.
        
        Args:
            output_dir: Output directory for lookup tables
            h3_resolution: H3 resolution for spatial indexing
            irradiance_grid_resolution: Resolution of irradiance grid in degrees
            n_workers: Number of workers for parallel processing
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.h3_resolution = h3_resolution
        self.irradiance_grid_resolution = irradiance_grid_resolution
        self.n_workers = n_workers
        
        # Initialize components
        self.ocf_loader = OCFQuartzLoader()
        self.osm_extractor = OSMPVExtractor()
        self.h3_indexer = H3Indexer(resolution=h3_resolution)
        self.irradiance_matcher = IrradianceMatcher(
            irradiance_grid_resolution=irradiance_grid_resolution
        )
        
        logger.info(f"Initialized PV lookup table builder")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"H3 resolution: {h3_resolution}")
        logger.info(f"Irradiance grid resolution: {irradiance_grid_resolution}°")
    
    def build_table(self, data_sources: Optional[List[str]] = None, 
                   use_sample: bool = True) -> Dict[str, Any]:
        """
        Build the complete PV lookup table.
        
        Args:
            data_sources: List of data sources to use ('ocf', 'osm')
            use_sample: If True, use sample data for testing
            
        Returns:
            Dictionary with build results and file paths
        """
        if data_sources is None:
            data_sources = ['ocf', 'osm']
        
        logger.info("="*60)
        logger.info("BUILDING PV LOOKUP TABLE")
        logger.info("="*60)
        logger.info(f"Data sources: {data_sources}")
        logger.info(f"Use sample data: {use_sample}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load PV data from multiple sources
            logger.info("\n📥 Step 1: Loading PV data from sources...")
            pv_data = self._load_pv_data(data_sources, use_sample)
            
            if len(pv_data) == 0:
                raise ValueError("No PV data loaded from any source")
            
            # Step 2: Validate and clean data
            logger.info("\n🔍 Step 2: Validating and cleaning data...")
            cleaned_data = self._validate_and_clean_data(pv_data)
            
            # Step 3: Apply H3 spatial indexing
            logger.info("\n📍 Step 3: Applying H3 spatial indexing...")
            indexed_data = self._apply_h3_indexing(cleaned_data)
            
            # Step 4: Spatial join with irradiance grid
            logger.info("\n🔗 Step 4: Spatial join with irradiance grid...")
            matched_data = self._spatial_join_with_irradiance(indexed_data)
            
            # Step 5: Create lookup tables
            logger.info("\n📊 Step 5: Creating lookup tables...")
            lookup_tables = self._create_lookup_tables(matched_data)
            
            # Step 6: Save results
            logger.info("\n💾 Step 6: Saving results...")
            saved_files = self._save_results(matched_data, lookup_tables)
            
            end_time = datetime.now()
            build_time = (end_time - start_time).total_seconds()
            
            # Generate summary
            summary = self._generate_summary(matched_data, lookup_tables, build_time)
            
            logger.info("\n" + "="*60)
            logger.info("PV LOOKUP TABLE BUILD COMPLETED")
            logger.info("="*60)
            logger.info(f"Build time: {build_time:.2f} seconds")
            logger.info(f"Total PV assets: {len(matched_data)}")
            logger.info(f"Unique H3 hexagons: {matched_data['h3_index'].nunique()}")
            logger.info(f"Unique grid points: {matched_data['irradiance_nearest_lat'].nunique()}")
            
            return {
                'success': True,
                'build_time_seconds': build_time,
                'summary': summary,
                'files': saved_files,
                'data': {
                    'matched_pv_data': matched_data,
                    'lookup_tables': lookup_tables
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to build PV lookup table: {e}")
            return {
                'success': False,
                'error': str(e),
                'build_time_seconds': (datetime.now() - start_time).total_seconds()
            }
    
    def _load_pv_data(self, data_sources: List[str], use_sample: bool) -> pd.DataFrame:
        """Load PV data from specified sources."""
        all_data = []
        
        for source in data_sources:
            try:
                if source == 'ocf':
                    logger.info("Loading OCF Quartz PV data...")
                    ocf_data = self.ocf_loader.load_german_pv_sites(use_sample=use_sample)
                    if len(ocf_data) > 0:
                        all_data.append(ocf_data)
                        logger.info(f"Loaded {len(ocf_data)} OCF PV sites")
                
                elif source == 'osm':
                    logger.info("Loading OSM PV data...")
                    osm_data = self.osm_extractor.extract_german_pv_sites(use_sample=use_sample)
                    if len(osm_data) > 0:
                        all_data.append(osm_data)
                        logger.info(f"Loaded {len(osm_data)} OSM PV sites")
                
                else:
                    logger.warning(f"Unknown data source: {source}")
                    
            except Exception as e:
                logger.error(f"Failed to load data from {source}: {e}")
        
        if len(all_data) == 0:
            return pd.DataFrame()
        
        # Combine all data sources
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined {len(combined_data)} PV assets from {len(data_sources)} sources")
        
        return combined_data
    
    def _validate_and_clean_data(self, pv_data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean PV data."""
        logger.info(f"Validating {len(pv_data)} PV assets...")
        
        # Validate OCF data
        if len(pv_data) > 0:
            is_valid = self.ocf_loader.validate_pv_data(pv_data)
            if not is_valid:
                logger.warning("PV data validation failed, but continuing...")
        
        # Remove duplicates based on asset_id
        initial_count = len(pv_data)
        cleaned_data = pv_data.drop_duplicates(subset=['asset_id'], keep='first')
        final_count = len(cleaned_data)
        
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count} duplicate assets")
        
        # Ensure required columns exist
        required_columns = ['asset_id', 'latitude', 'longitude', 'capacity_kw']
        missing_columns = [col for col in required_columns if col not in cleaned_data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Data validation and cleaning completed: {len(cleaned_data)} assets")
        return cleaned_data
    
    def _apply_h3_indexing(self, pv_data: pd.DataFrame) -> pd.DataFrame:
        """Apply H3 spatial indexing to PV data."""
        logger.info(f"Applying H3 indexing to {len(pv_data)} PV assets...")
        
        indexed_data = self.h3_indexer.index_pv_assets(pv_data)
        
        # Get H3 statistics
        h3_stats = self.h3_indexer.get_h3_statistics(indexed_data)
        logger.info(f"H3 indexing completed: {len(h3_stats)} unique hexagons")
        
        return indexed_data
    
    def _spatial_join_with_irradiance(self, indexed_data: pd.DataFrame) -> pd.DataFrame:
        """Perform spatial join with irradiance grid."""
        logger.info(f"Performing spatial join for {len(indexed_data)} PV assets...")
        
        matched_data = self.irradiance_matcher.match_pv_to_irradiance_grid(indexed_data)
        
        # Validate matching quality
        quality = self.irradiance_matcher.validate_matching_quality(matched_data)
        logger.info(f"Spatial join completed: {quality['close_match_percentage']:.1f}% close matches")
        
        return matched_data
    
    def _create_lookup_tables(self, matched_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create various lookup tables."""
        logger.info("Creating lookup tables...")
        
        lookup_tables = {}
        
        # 1. H3-based lookup table
        h3_stats = self.h3_indexer.get_h3_statistics(matched_data)
        lookup_tables['h3_lookup'] = h3_stats
        
        # 2. Grid-based lookup table
        grid_stats = self.irradiance_matcher.get_grid_statistics(matched_data)
        lookup_tables['grid_lookup'] = grid_stats
        
        # 3. Compact grid lookup table
        compact_lookup = self.irradiance_matcher.create_grid_lookup_table(matched_data)
        lookup_tables['compact_lookup'] = compact_lookup
        
        # 4. Asset summary table
        asset_summary = matched_data.groupby('data_source').agg({
            'asset_id': 'count',
            'capacity_kw': ['sum', 'mean'],
            'irradiance_distance_km': 'mean'
        }).reset_index()
        asset_summary.columns = ['data_source', 'asset_count', 'total_capacity_kw', 
                               'avg_capacity_kw', 'avg_distance_km']
        lookup_tables['asset_summary'] = asset_summary
        
        logger.info(f"Created {len(lookup_tables)} lookup tables")
        return lookup_tables
    
    def _save_results(self, matched_data: pd.DataFrame, 
                     lookup_tables: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Save all results to files."""
        saved_files = {}
        
        # Save main matched data
        main_file = self.output_dir / "pv_assets_matched.parquet"
        matched_data.to_parquet(main_file, index=False)
        saved_files['matched_pv_data'] = str(main_file)
        
        # Save lookup tables
        for table_name, table_data in lookup_tables.items():
            table_file = self.output_dir / f"{table_name}.parquet"
            table_data.to_parquet(table_file, index=False)
            saved_files[table_name] = str(table_file)
        
        # Save metadata
        metadata = {
            'build_timestamp': datetime.now().isoformat(),
            'h3_resolution': self.h3_resolution,
            'irradiance_grid_resolution': self.irradiance_grid_resolution,
            'n_workers': self.n_workers,
            'total_assets': len(matched_data),
            'unique_h3_hexagons': matched_data['h3_index'].nunique(),
            'unique_grid_points': matched_data['irradiance_nearest_lat'].nunique()
        }
        
        metadata_file = self.output_dir / "build_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = str(metadata_file)
        
        logger.info(f"Saved {len(saved_files)} files to {self.output_dir}")
        return saved_files
    
    def _generate_summary(self, matched_data: pd.DataFrame, 
                         lookup_tables: Dict[str, pd.DataFrame], 
                         build_time: float) -> Dict[str, Any]:
        """Generate build summary."""
        summary = {
            'build_time_seconds': build_time,
            'total_pv_assets': len(matched_data),
            'unique_h3_hexagons': matched_data['h3_index'].nunique(),
            'unique_grid_points': matched_data['irradiance_nearest_lon'].nunique(),
            'data_sources': matched_data['data_source'].value_counts().to_dict(),
            'capacity_stats': {
                'total_capacity_mw': matched_data['capacity_kw'].sum() / 1000,
                'avg_capacity_kw': matched_data['capacity_kw'].mean(),
                'min_capacity_kw': matched_data['capacity_kw'].min(),
                'max_capacity_kw': matched_data['capacity_kw'].max()
            },
            'spatial_stats': {
                'avg_distance_km': matched_data['irradiance_distance_km'].mean(),
                'max_distance_km': matched_data['irradiance_distance_km'].max(),
                'close_matches_pct': (matched_data['irradiance_distance_km'] <= 10.0).mean() * 100
            },
            'lookup_tables': {
                name: len(table) for name, table in lookup_tables.items()
            }
        }
        
        return summary
