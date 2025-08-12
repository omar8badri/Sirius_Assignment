#!/usr/bin/env python3
"""
PV Lookup Table Sample Test
===========================

This script tests the PV lookup table pipeline with sample data to verify
that all components work correctly before running on larger datasets.

The pipeline:
1. Loads sample PV data from OCF and OSM sources
2. Applies H3 spatial indexing
3. Performs spatial join with irradiance grid
4. Creates compact lookup tables
5. Saves results for repeated use
"""

import sys
import time
import pandas as pd
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from pv_locations import PVLookupTableBuilder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pv_lookup_pipeline():
    """Test the PV lookup table pipeline with sample data."""
    print("="*70)
    print("PV LOOKUP TABLE SAMPLE TEST")
    print("="*70)
    print("Testing PV locations pipeline with sample data")
    print("Data sources: OCF Quartz + OSM")
    print("Spatial indexing: H3 resolution 9 (~173m hexagons)")
    print("Irradiance grid: 0.1° resolution (ERA5 compatible)")
    print("="*70)
    
    start_time = time.time()
    
    try:
        # Create PV lookup table builder
        builder = PVLookupTableBuilder(
            output_dir="data_pv_lookup_sample",
            h3_resolution=9,  # ~173m hexagons
            irradiance_grid_resolution=0.1,  # ERA5 resolution
            n_workers=2  # Conservative for testing
        )
        
        print("\n🚀 Starting PV lookup table build...")
        
        # Build the lookup table with sample data
        results = builder.build_table(
            data_sources=['ocf', 'osm'],
            use_sample=True  # Use sample data for testing
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if results['success']:
            print("\n" + "="*70)
            print("✅ PV LOOKUP TABLE TEST SUCCESSFUL")
            print("="*70)
            
            summary = results['summary']
            
            print(f"⏱️  Build Time: {total_time:.2f} seconds")
            print(f"📊 Total PV Assets: {summary['total_pv_assets']}")
            print(f"📍 Unique H3 Hexagons: {summary['unique_h3_hexagons']}")
            print(f"🌍 Unique Grid Points: {summary['unique_grid_points']}")
            
            print(f"\n📈 Data Sources:")
            for source, count in summary['data_sources'].items():
                print(f"   {source}: {count} assets")
            
            print(f"\n⚡ Capacity Statistics:")
            capacity_stats = summary['capacity_stats']
            print(f"   Total Capacity: {capacity_stats['total_capacity_mw']:.2f} MW")
            print(f"   Average Capacity: {capacity_stats['avg_capacity_kw']:.1f} kW")
            print(f"   Capacity Range: {capacity_stats['min_capacity_kw']:.1f} - {capacity_stats['max_capacity_kw']:.1f} kW")
            
            print(f"\n🌐 Spatial Statistics:")
            spatial_stats = summary['spatial_stats']
            print(f"   Average Distance: {spatial_stats['avg_distance_km']:.2f} km")
            print(f"   Maximum Distance: {spatial_stats['max_distance_km']:.2f} km")
            print(f"   Close Matches: {spatial_stats['close_matches_pct']:.1f}%")
            
            print(f"\n📋 Lookup Tables Created:")
            for table_name, row_count in summary['lookup_tables'].items():
                print(f"   {table_name}: {row_count} rows")
            
            print(f"\n💾 Output Files:")
            for file_type, file_path in results['files'].items():
                print(f"   {file_type}: {file_path}")
            
            # Test data access
            print(f"\n🔍 Testing Data Access...")
            test_data_access(results)
            
            print("\n" + "="*70)
            print("🎉 All tests passed! Pipeline is ready for production use.")
            print("="*70)
            
            return True
            
        else:
            print(f"\n❌ PV lookup table test failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        logger.error(f"PV lookup table test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_access(results):
    """Test accessing the generated data."""
    try:
        # Load main matched data
        matched_data = results['data']['matched_pv_data']
        print(f"   ✅ Loaded {len(matched_data)} matched PV assets")
        
        # Test H3 lookup table
        h3_lookup = results['data']['lookup_tables']['h3_lookup']
        print(f"   ✅ Loaded {len(h3_lookup)} H3 hexagon records")
        
        # Test compact lookup table
        compact_lookup = results['data']['lookup_tables']['compact_lookup']
        print(f"   ✅ Loaded {len(compact_lookup)} grid point records")
        
        # Test asset summary
        asset_summary = results['data']['lookup_tables']['asset_summary']
        print(f"   ✅ Loaded {len(asset_summary)} data source summaries")
        
        # Test sample queries
        print(f"   🔍 Testing sample queries...")
        
        # Query 1: Find PV assets in a specific H3 hexagon
        sample_h3 = matched_data['h3_index'].iloc[0]
        h3_assets = matched_data[matched_data['h3_index'] == sample_h3]
        print(f"      H3 {sample_h3}: {len(h3_assets)} assets")
        
        # Query 2: Find PV assets near a specific grid point
        sample_grid_lat = matched_data['irradiance_nearest_lat'].iloc[0]
        sample_grid_lon = matched_data['irradiance_nearest_lon'].iloc[0]
        grid_assets = matched_data[
            (matched_data['irradiance_nearest_lat'] == sample_grid_lat) &
            (matched_data['irradiance_nearest_lon'] == sample_grid_lon)
        ]
        print(f"      Grid ({sample_grid_lat:.1f}, {sample_grid_lon:.1f}): {len(grid_assets)} assets")
        
        # Query 3: Find high-capacity assets
        high_capacity = matched_data[matched_data['capacity_kw'] > 15]
        print(f"      High capacity (>15kW): {len(high_capacity)} assets")
        
        print(f"   ✅ All data access tests passed")
        
    except Exception as e:
        print(f"   ❌ Data access test failed: {e}")

def main():
    """Main function to run the PV lookup table test."""
    print("🔬 PV Lookup Table Sample Test Pipeline")
    print("Testing with sample data from OCF and OSM sources")
    print("This will create a compact lookup table for repeated use...")
    
    success = test_pv_lookup_pipeline()
    
    if success:
        print("\n🎉 PV lookup table test completed successfully!")
        print("📁 Check the 'data_pv_lookup_sample' directory for results")
        print("🚀 Ready to run with real data sources")
    else:
        print("\n❌ PV lookup table test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
