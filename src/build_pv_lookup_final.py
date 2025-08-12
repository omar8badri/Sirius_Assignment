#!/usr/bin/env python3
"""
PV Lookup Table Builder - Final Version
======================================

This script builds the complete PV lookup table as specified:
1. PV locations from open datasets (OCF/OSM)
2. Spatial join to nearest irradiance pixels using H3
3. Compact lookup table for repeated use

No ERA5 integration - just the spatial relationships.
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

from pv_locations import PVLookupTableBuilder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_final_pv_lookup_table():
    """Build the final PV lookup table with spatial indexing."""
    print("="*80)
    print("🏗️  BUILDING FINAL PV LOOKUP TABLE")
    print("="*80)
    print("📋 Requirements:")
    print("   1. PV locations from open datasets (OCF/OSM)")
    print("   2. Spatial join to nearest irradiance pixels using H3")
    print("   3. Compact lookup table for repeated use")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Step 1: Build comprehensive PV lookup table
        print("\n📍 Step 1: Building PV Lookup Table with Spatial Indexing...")
        pv_results = build_comprehensive_pv_lookup()
        
        if not pv_results['success']:
            raise Exception(f"PV lookup table build failed: {pv_results.get('error')}")
        
        # Step 2: Create compact lookup tables for repeated use
        print("\n💾 Step 2: Creating Compact Lookup Tables...")
        compact_results = create_compact_lookup_tables(pv_results)
        
        # Step 3: Generate final summary and metadata
        print("\n📊 Step 3: Generating Final Summary...")
        final_results = generate_final_summary(pv_results, compact_results)
        
        end_time = time.time()
        build_time = end_time - start_time
        
        # Display comprehensive results
        print("\n" + "="*80)
        print("🎉 FINAL PV LOOKUP TABLE COMPLETED")
        print("="*80)
        print(f"⏱️  Build Time: {build_time:.2f} seconds")
        print(f"📊 Total PV Assets: {pv_results['summary']['total_pv_assets']}")
        print(f"📍 Unique H3 Hexagons: {pv_results['summary']['unique_h3_hexagons']}")
        print(f"🌍 Unique Irradiance Grid Points: {pv_results['summary']['unique_grid_points']}")
        print(f"🔗 Spatial Join Quality: {pv_results['summary']['spatial_stats']['close_matches_pct']:.1f}% close matches")
        
        print(f"\n📋 Compact Lookup Tables Created:")
        for table_name, table_info in compact_results['tables'].items():
            print(f"   {table_name}: {table_info['rows']} rows, {table_info['size_mb']:.2f} MB")
        
        print(f"\n💾 Output Directory: {final_results['output_dir']}")
        print(f"📁 Total Files: {len(final_results['files'])}")
        
        print(f"\n🚀 Ready for repeated use!")
        print("="*80)
        
        return {
            'success': True,
            'build_time_seconds': build_time,
            'summary': final_results['summary'],
            'files': final_results['files'],
            'output_dir': final_results['output_dir']
        }
        
    except Exception as e:
        print(f"\n❌ Final PV lookup table build failed: {e}")
        logger.error(f"Final PV lookup table build failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'build_time_seconds': time.time() - start_time
        }

def build_comprehensive_pv_lookup():
    """Build comprehensive PV lookup table with spatial indexing."""
    print("   Building comprehensive PV lookup table...")
    
    # Create PV lookup table builder with optimal settings
    builder = PVLookupTableBuilder(
        output_dir="data_pv_lookup_final",
        h3_resolution=9,  # ~173m hexagons for good spatial resolution
        irradiance_grid_resolution=0.1,  # 0.1° grid for irradiance matching
        n_workers=4  # Optimized for performance
    )
    
    # Build with multiple data sources
    results = builder.build_table(
        data_sources=['ocf', 'osm'],  # Both OCF and OSM data
        use_sample=True  # Use sample data for now
    )
    
    if results['success']:
        print(f"   ✅ Comprehensive PV lookup table built successfully")
        print(f"   📊 {results['summary']['total_pv_assets']} PV assets indexed")
        print(f"   📍 {results['summary']['unique_h3_hexagons']} H3 hexagons")
        print(f"   🌍 {results['summary']['unique_grid_points']} irradiance grid points")
        print(f"   🔗 {results['summary']['spatial_stats']['close_matches_pct']:.1f}% close spatial matches")
    
    return results

def create_compact_lookup_tables(pv_results):
    """Create compact lookup tables optimized for repeated use."""
    print("   Creating compact lookup tables...")
    
    # Load the main matched data
    matched_data = pv_results['data']['matched_pv_data']
    lookup_tables = pv_results['data']['lookup_tables']
    
    # Create optimized compact tables
    compact_tables = {}
    
    # 1. Primary Compact Lookup Table (most important)
    print("   Creating primary compact lookup table...")
    primary_lookup = create_primary_compact_table(matched_data)
    compact_tables['primary_lookup'] = primary_lookup
    
    # 2. H3 Spatial Index Table
    print("   Creating H3 spatial index table...")
    h3_lookup = create_h3_spatial_table(matched_data)
    compact_tables['h3_spatial_lookup'] = h3_lookup
    
    # 3. Grid Point Index Table
    print("   Creating grid point index table...")
    grid_lookup = create_grid_point_table(matched_data)
    compact_tables['grid_point_lookup'] = grid_lookup
    
    # 4. Asset Summary Table
    print("   Creating asset summary table...")
    asset_summary = create_asset_summary_table(matched_data)
    compact_tables['asset_summary'] = asset_summary
    
    # Save all compact tables
    output_dir = Path("data_pv_lookup_final")
    files = {}
    table_info = {}
    
    for table_name, table_data in compact_tables.items():
        file_path = output_dir / f"{table_name}.parquet"
        table_data.to_parquet(file_path, index=False, compression='snappy')
        
        # Calculate file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        files[table_name] = str(file_path)
        table_info[table_name] = {
            'rows': len(table_data),
            'columns': len(table_data.columns),
            'size_mb': file_size_mb
        }
        
        print(f"   ✅ {table_name}: {len(table_data)} rows, {file_size_mb:.2f} MB")
    
    # Save metadata
    metadata = {
        'build_timestamp': datetime.now().isoformat(),
        'total_assets': len(matched_data),
        'unique_h3_hexagons': matched_data['h3_index'].nunique(),
        'unique_grid_points': matched_data['irradiance_nearest_lat'].nunique(),
        'spatial_join_quality': {
            'close_matches_pct': pv_results['summary']['spatial_stats']['close_matches_pct'],
            'avg_distance_km': pv_results['summary']['spatial_stats']['avg_distance_km'],
            'max_distance_km': pv_results['summary']['spatial_stats']['max_distance_km']
        },
        'table_info': table_info
    }
    
    metadata_file = output_dir / "compact_lookup_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    files['metadata'] = str(metadata_file)
    
    print(f"   ✅ All compact lookup tables saved")
    print(f"   📁 {len(files)} files created")
    
    return {
        'success': True,
        'tables': table_info,
        'files': files,
        'data': compact_tables
    }

def create_primary_compact_table(matched_data):
    """Create the primary compact lookup table for repeated use."""
    # This is the main table that will be used most frequently
    primary_lookup = matched_data.groupby([
        'irradiance_nearest_lat', 'irradiance_nearest_lon', 'irradiance_era5_grid_id'
    ]).agg({
        'asset_id': list,
        'capacity_kw': 'sum',
        'irradiance_distance_km': 'mean',
        'h3_index': 'first',
        'data_source': lambda x: list(set(x)),  # Unique data sources
        'technology': lambda x: list(set(x)),   # Unique technologies
        'tilt_angle': 'mean',
        'azimuth': 'mean'
    }).reset_index()
    
    # Rename columns for clarity
    primary_lookup.columns = [
        'grid_lat', 'grid_lon', 'era5_grid_id', 'pv_asset_ids', 
        'total_capacity_kw', 'avg_distance_km', 'h3_index',
        'data_sources', 'technologies', 'avg_tilt_angle', 'avg_azimuth'
    ]
    
    # Add derived columns
    primary_lookup['pv_count'] = primary_lookup['pv_asset_ids'].apply(len)
    primary_lookup['data_source_count'] = primary_lookup['data_sources'].apply(len)
    primary_lookup['technology_count'] = primary_lookup['technologies'].apply(len)
    
    # Add metadata
    primary_lookup['created_at'] = datetime.now()
    primary_lookup['lookup_version'] = '1.0'
    
    return primary_lookup

def create_h3_spatial_table(matched_data):
    """Create H3 spatial index table for spatial queries."""
    h3_lookup = matched_data.groupby('h3_index').agg({
        'asset_id': list,
        'capacity_kw': 'sum',
        'latitude': 'mean',
        'longitude': 'mean',
        'irradiance_nearest_lat': 'first',
        'irradiance_nearest_lon': 'first',
        'irradiance_distance_km': 'mean',
        'data_source': lambda x: list(set(x))
    }).reset_index()
    
    # Rename columns
    h3_lookup.columns = [
        'h3_index', 'pv_asset_ids', 'total_capacity_kw', 'centroid_lat', 'centroid_lon',
        'nearest_grid_lat', 'nearest_grid_lon', 'avg_distance_km', 'data_sources'
    ]
    
    # Add derived columns
    h3_lookup['pv_count'] = h3_lookup['pv_asset_ids'].apply(len)
    h3_lookup['data_source_count'] = h3_lookup['data_sources'].apply(len)
    
    return h3_lookup

def create_grid_point_table(matched_data):
    """Create grid point index table for irradiance matching."""
    grid_lookup = matched_data.groupby([
        'irradiance_nearest_lat', 'irradiance_nearest_lon', 'irradiance_era5_grid_id'
    ]).agg({
        'asset_id': list,
        'capacity_kw': 'sum',
        'irradiance_distance_km': ['mean', 'max'],
        'h3_index': lambda x: list(set(x)),
        'data_source': lambda x: list(set(x))
    }).reset_index()
    
    # Flatten column names
    grid_lookup.columns = [
        'grid_lat', 'grid_lon', 'era5_grid_id', 'pv_asset_ids', 
        'total_capacity_kw', 'avg_distance_km', 'max_distance_km',
        'h3_indices', 'data_sources'
    ]
    
    # Add derived columns
    grid_lookup['pv_count'] = grid_lookup['pv_asset_ids'].apply(len)
    grid_lookup['h3_count'] = grid_lookup['h3_indices'].apply(len)
    
    return grid_lookup

def create_asset_summary_table(matched_data):
    """Create asset summary table for high-level statistics."""
    asset_summary = matched_data.groupby('data_source').agg({
        'asset_id': 'count',
        'capacity_kw': ['sum', 'mean', 'min', 'max'],
        'irradiance_distance_km': 'mean',
        'technology': lambda x: list(set(x)),
        'tilt_angle': 'mean',
        'azimuth': 'mean'
    }).reset_index()
    
    # Flatten column names
    asset_summary.columns = [
        'data_source', 'asset_count', 'total_capacity_kw', 'avg_capacity_kw',
        'min_capacity_kw', 'max_capacity_kw', 'avg_distance_km', 'technologies',
        'avg_tilt_angle', 'avg_azimuth'
    ]
    
    return asset_summary

def generate_final_summary(pv_results, compact_results):
    """Generate final summary and metadata."""
    print("   Generating final summary...")
    
    # Create comprehensive summary
    summary = {
        'build_info': {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'description': 'PV Lookup Table with Spatial Indexing'
        },
        'pv_assets': {
            'total_assets': pv_results['summary']['total_pv_assets'],
            'data_sources': pv_results['summary']['data_sources'],
            'capacity_stats': pv_results['summary']['capacity_stats']
        },
        'spatial_indexing': {
            'h3_resolution': 9,
            'unique_h3_hexagons': pv_results['summary']['unique_h3_hexagons'],
            'irradiance_grid_resolution': 0.1,
            'unique_grid_points': pv_results['summary']['unique_grid_points']
        },
        'spatial_join': {
            'quality_score': pv_results['summary']['spatial_stats'].get('quality_score', 0),
            'close_matches_pct': pv_results['summary']['spatial_stats']['close_matches_pct'],
            'avg_distance_km': pv_results['summary']['spatial_stats']['avg_distance_km'],
            'max_distance_km': pv_results['summary']['spatial_stats']['max_distance_km']
        },
        'lookup_tables': compact_results['tables'],
        'usage_instructions': {
            'primary_lookup': 'Main table for PV asset to irradiance grid mapping',
            'h3_spatial_lookup': 'Spatial queries using H3 hexagons',
            'grid_point_lookup': 'Irradiance grid point statistics',
            'asset_summary': 'High-level asset statistics by data source'
        }
    }
    
    # Save comprehensive summary
    output_dir = Path("data_pv_lookup_final")
    summary_file = output_dir / "comprehensive_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    compact_results['files']['comprehensive_summary'] = str(summary_file)
    
    return {
        'success': True,
        'summary': summary,
        'files': compact_results['files'],
        'output_dir': str(output_dir)
    }

def main():
    """Main function to build the final PV lookup table."""
    print("🔬 Final PV Lookup Table Builder")
    print("Building compact lookup table for repeated use...")
    print("Focus: PV locations + Spatial indexing + Compact storage")
    
    results = build_final_pv_lookup_table()
    
    if results['success']:
        print("\n🎉 Final PV lookup table built successfully!")
        print(f"📁 Output directory: {results['output_dir']}")
        print(f"⏱️  Build time: {results['build_time_seconds']:.2f} seconds")
        
        print(f"\n📋 Compact Lookup Tables Ready:")
        for table_name, table_info in results['summary']['lookup_tables'].items():
            print(f"   {table_name}: {table_info['rows']} rows, {table_info['size_mb']:.2f} MB")
        
        print(f"\n🚀 Ready for repeated use in solar forecasting applications!")
        print("💡 Use the compact lookup tables for fast spatial queries")
    else:
        print(f"\n❌ Final PV lookup table build failed: {results.get('error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
