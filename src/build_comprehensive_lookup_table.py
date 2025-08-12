#!/usr/bin/env python3
"""
Build Comprehensive PV Lookup Table with Full OSM Data
======================================================

This script builds a comprehensive lookup table using the full 162k OSM PV data
instead of the sample data currently being used.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

from pv_locations.lookup_table.table_builder import PVLookupTableBuilder
from pv_locations.spatial_indexing.h3_indexer import H3Indexer
from pv_locations.spatial_join.irradiance_matcher import IrradianceMatcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_full_osm_data():
    """Load the full 162k OSM PV data."""
    osm_file = Path("data_german_pv_scaled/german_pv_osm_scaled.parquet")
    
    if not osm_file.exists():
        raise FileNotFoundError(f"OSM data file not found: {osm_file}")
    
    # Load the full OSM data
    df = pd.read_parquet(osm_file)
    print(f"Loaded {len(df):,} OSM PV locations")
    
    # Standardize column names
    column_mapping = {
        'asset_id': 'asset_id',
        'latitude': 'latitude', 
        'longitude': 'longitude',
        'capacity_kw': 'capacity_kw',
        'installation_date': 'installation_date',
        'technology': 'technology',
        'tilt_angle': 'tilt_angle',
        'azimuth': 'azimuth',
        'data_source': 'data_source'
    }
    
    # Rename columns to match expected format
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # Add missing columns with defaults
    if 'country' not in df.columns:
        df['country'] = 'Germany'
    if 'created_at' not in df.columns:
        df['created_at'] = pd.Timestamp.now()
    if 'updated_at' not in df.columns:
        df['updated_at'] = pd.Timestamp.now()
    
    # Validate data
    print(f"Data validation:")
    print(f"  - Latitude range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
    print(f"  - Longitude range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
    print(f"  - Capacity range: {df['capacity_kw'].min():.2f} to {df['capacity_kw'].max():.2f} kW")
    print(f"  - Total capacity: {df['capacity_kw'].sum()/1000:.1f} MW")
    
    return df

def build_comprehensive_lookup_table():
    """Build comprehensive lookup table with full OSM data."""
    print("="*80)
    print("🏗️  BUILDING COMPREHENSIVE PV LOOKUP TABLE")
    print("="*80)
    print("Using full 162k OSM PV data instead of sample data")
    print("="*80)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Load full OSM data
        print("\n📥 Step 1: Loading full OSM PV data...")
        pv_data = load_full_osm_data()
        
        # Step 2: Create custom lookup table builder
        print("\n🔧 Step 2: Creating lookup table builder...")
        builder = PVLookupTableBuilder(
            output_dir="data_pv_lookup_comprehensive",
            h3_resolution=9,  # ~173m hexagons
            irradiance_grid_resolution=0.1,  # 0.1° grid
            n_workers=4
        )
        
        # Step 3: Process the data manually (bypass sample limitation)
        print("\n⚙️  Step 3: Processing PV data...")
        
        # Validate and clean data
        print("   Validating and cleaning data...")
        cleaned_data = builder._validate_and_clean_data(pv_data)
        print(f"   ✅ Cleaned data: {len(cleaned_data):,} records")
        
        # Apply H3 spatial indexing
        print("   Applying H3 spatial indexing...")
        indexed_data = builder._apply_h3_indexing(cleaned_data)
        print(f"   ✅ Indexed data: {len(indexed_data):,} records")
        print(f"   📍 Unique H3 hexagons: {indexed_data['h3_index'].nunique():,}")
        
        # Spatial join with irradiance grid
        print("   Spatial join with irradiance grid...")
        matched_data = builder._spatial_join_with_irradiance(indexed_data)
        print(f"   ✅ Matched data: {len(matched_data):,} records")
        print(f"   🌍 Unique grid points: {matched_data['irradiance_nearest_lat'].nunique():,}")
        
        # Create lookup tables
        print("   Creating lookup tables...")
        lookup_tables = builder._create_lookup_tables(matched_data)
        print(f"   ✅ Created {len(lookup_tables)} lookup tables")
        
        # Save results
        print("   Saving results...")
        saved_files = builder._save_results(matched_data, lookup_tables)
        print(f"   ✅ Saved {len(saved_files)} files")
        
        end_time = datetime.now()
        build_time = (end_time - start_time).total_seconds()
        
        # Generate summary
        summary = builder._generate_summary(matched_data, lookup_tables, build_time)
        
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE PV LOOKUP TABLE COMPLETED")
        print("="*80)
        print(f"⏱️  Build Time: {build_time:.2f} seconds")
        print(f"📊 Total PV Assets: {len(matched_data):,}")
        print(f"📍 Unique H3 Hexagons: {matched_data['h3_index'].nunique():,}")
        print(f"🌍 Unique Grid Points: {matched_data['irradiance_nearest_lat'].nunique():,}")
        print(f"🔗 Spatial Join Quality: {summary['spatial_stats']['close_matches_pct']:.1f}% close matches")
        print(f"⚡ Total Capacity: {matched_data['capacity_kw'].sum()/1000:.1f} MW")
        
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
        print(f"❌ Failed to build comprehensive lookup table: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'build_time_seconds': (datetime.now() - start_time).total_seconds()
        }

def create_compact_lookup_tables(results):
    """Create compact lookup tables for the comprehensive data."""
    if not results['success']:
        return results
    
    print("\n💾 Creating compact lookup tables...")
    
    matched_data = results['data']['matched_pv_data']
    output_dir = Path("data_pv_lookup_comprehensive")
    
    # Create primary compact lookup table
    primary_lookup = matched_data[['asset_id', 'latitude', 'longitude', 'capacity_kw', 
                                  'h3_index', 'irradiance_nearest_lat', 'irradiance_nearest_lon']].copy()
    
    primary_file = output_dir / "primary_lookup.parquet"
    primary_lookup.to_parquet(primary_file, index=False, compression='snappy')
    
    file_size_mb = primary_file.stat().st_size / (1024 * 1024)
    print(f"   ✅ Primary lookup: {len(primary_lookup):,} rows, {file_size_mb:.2f} MB")
    
    return results

if __name__ == "__main__":
    print("🔬 Comprehensive PV Lookup Table Builder")
    print("Building lookup table with full 162k OSM data...")
    
    results = build_comprehensive_lookup_table()
    
    if results['success']:
        # Create compact tables
        results = create_compact_lookup_tables(results)
        
        print(f"\n🎉 Success! Created comprehensive lookup table with {results['summary']['total_pv_assets']:,} PV assets")
        print(f"📁 Output directory: data_pv_lookup_comprehensive")
        print(f"⏱️  Build time: {results['build_time_seconds']:.2f} seconds")
        
        print(f"\n📊 Summary:")
        print(f"   Total PV Assets: {results['summary']['total_pv_assets']:,}")
        print(f"   Unique H3 Hexagons: {results['summary']['unique_h3_hexagons']:,}")
        print(f"   Unique Grid Points: {results['summary']['unique_grid_points']:,}")
        print(f"   Spatial Join Quality: {results['summary']['spatial_stats']['close_matches_pct']:.1f}% close matches")
        
        print(f"\n🚀 Ready to use in main pipeline!")
    else:
        print(f"\n❌ Failed: {results.get('error')}")
        sys.exit(1)
