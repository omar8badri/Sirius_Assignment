#!/usr/bin/env python3
"""
Integrated PV Lookup Table with Solar Radiation Pipeline
=======================================================

This script demonstrates how to integrate the PV lookup table with the existing
solar radiation pipeline to create a complete system for solar forecasting.

The integrated pipeline:
1. Builds PV lookup table with spatial indexing
2. Runs solar radiation data processing
3. Creates combined dataset for solar forecasting
"""

import sys
import time
import pandas as pd
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from pv_locations import PVLookupTableBuilder
from solar_radiation_pipeline_optimized import OptimizedSolarRadiationPipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_integrated_pipeline():
    """Run the integrated PV lookup table and solar radiation pipeline."""
    print("="*70)
    print("INTEGRATED PV LOOKUP TABLE + SOLAR RADIATION PIPELINE")
    print("="*70)
    print("This demonstrates the complete solar forecasting system:")
    print("1. PV asset locations with spatial indexing")
    print("2. Solar radiation data processing")
    print("3. Combined dataset for forecasting")
    print("="*70)
    
    start_time = time.time()
    
    try:
        # Step 1: Build PV lookup table
        print("\n📍 Step 1: Building PV Lookup Table...")
        pv_results = build_pv_lookup_table()
        
        if not pv_results['success']:
            raise Exception(f"PV lookup table build failed: {pv_results.get('error')}")
        
        # Step 2: Run solar radiation pipeline (small sample)
        print("\n☀️  Step 2: Running Solar Radiation Pipeline...")
        solar_results = run_solar_radiation_pipeline()
        
        if not solar_results['success']:
            raise Exception(f"Solar radiation pipeline failed: {solar_results.get('error')}")
        
        # Step 3: Create combined dataset
        print("\n🔗 Step 3: Creating Combined Dataset...")
        combined_results = create_combined_dataset(pv_results, solar_results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Generate final summary
        print("\n" + "="*70)
        print("🎉 INTEGRATED PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"📊 PV Assets: {pv_results['summary']['total_pv_assets']}")
        print(f"📍 H3 Hexagons: {pv_results['summary']['unique_h3_hexagons']}")
        print(f"🌍 Grid Points: {pv_results['summary']['unique_grid_points']}")
        print(f"☀️  Solar Data Points: {solar_results.get('data_points', 'N/A')}")
        print(f"🔗 Combined Records: {combined_results.get('total_records', 'N/A')}")
        
        print(f"\n💾 Output Directories:")
        print(f"   PV Lookup: {pv_results['files']['metadata']}")
        print(f"   Solar Data: {solar_results.get('output_dir', 'N/A')}")
        print(f"   Combined: {combined_results.get('output_file', 'N/A')}")
        
        print("\n🚀 Ready for solar forecasting applications!")
        print("="*70)
        
        return {
            'success': True,
            'total_time': total_time,
            'pv_results': pv_results,
            'solar_results': solar_results,
            'combined_results': combined_results
        }
        
    except Exception as e:
        print(f"\n❌ Integrated pipeline failed: {e}")
        logger.error(f"Integrated pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'total_time': time.time() - start_time
        }

def build_pv_lookup_table():
    """Build the PV lookup table with sample data."""
    print("   Building PV lookup table with sample data...")
    
    builder = PVLookupTableBuilder(
        output_dir="data_pv_lookup_integrated",
        h3_resolution=9,
        irradiance_grid_resolution=0.1,
        n_workers=2
    )
    
    results = builder.build_table(
        data_sources=['ocf', 'osm'],
        use_sample=True
    )
    
    if results['success']:
        print(f"   ✅ PV lookup table built successfully")
        print(f"   📊 {results['summary']['total_pv_assets']} PV assets indexed")
        print(f"   📍 {results['summary']['unique_h3_hexagons']} H3 hexagons")
    
    return results

def run_solar_radiation_pipeline():
    """Run the solar radiation pipeline with a small sample."""
    print("   Running solar radiation pipeline with small sample...")
    
    # Create a small pipeline for testing
    pipeline = OptimizedSolarRadiationPipeline(
        output_dir="data_solar_integrated",
        use_polars=False,  # Use pandas for small sample
        enable_profiling=False,
        n_workers=2,
        batch_size=100
    )
    
    try:
        # For this demo, we'll just create a sample solar radiation dataset
        # In practice, you would run the full pipeline here
        sample_solar_data = create_sample_solar_data()
        
        # Save sample data
        output_dir = Path("data_solar_integrated")
        output_dir.mkdir(exist_ok=True)
        
        sample_file = output_dir / "sample_solar_data.parquet"
        sample_solar_data.to_parquet(sample_file, index=False)
        
        print(f"   ✅ Sample solar radiation data created")
        print(f"   📊 {len(sample_solar_data)} solar data points")
        
        return {
            'success': True,
            'output_dir': str(output_dir),
            'data_points': len(sample_solar_data),
            'sample_data': sample_solar_data
        }
        
    except Exception as e:
        print(f"   ❌ Solar radiation pipeline failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def create_sample_solar_data():
    """Create sample solar radiation data for demonstration."""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample time series
    start_time = datetime(2018, 1, 1, 0, 0)
    end_time = datetime(2018, 1, 2, 0, 0)
    time_range = pd.date_range(start=start_time, end=end_time, freq='H')
    
    # Sample grid points (matching PV lookup table)
    grid_points = [
        (52.5, 13.4),  # Berlin
        (48.1, 11.6),  # Munich
        (53.6, 10.0),  # Hamburg
        (50.9, 6.9),   # Cologne
        (50.1, 8.7),   # Frankfurt
    ]
    
    solar_data = []
    
    for lat, lon in grid_points:
        for time_point in time_range:
            # Simulate solar radiation (higher during day, zero at night)
            hour = time_point.hour
            if 6 <= hour <= 18:  # Daytime
                # Simple solar radiation model
                solar_zenith = abs(hour - 12) * 15  # Degrees from solar noon
                radiation = max(0, 800 * np.cos(np.radians(solar_zenith)) * 0.8)
            else:
                radiation = 0
            
            solar_data.append({
                'time': time_point,
                'latitude': lat,
                'longitude': lon,
                'ssrd': radiation,
                'era5_grid_id': f"germany_{lat:.1f}_{lon:.1f}"
            })
    
    return pd.DataFrame(solar_data)

def create_combined_dataset(pv_results, solar_results):
    """Create a combined dataset linking PV assets to solar radiation data."""
    print("   Creating combined PV + solar radiation dataset...")
    
    try:
        # Load PV lookup table
        pv_data = pv_results['data']['matched_pv_data']
        compact_lookup = pv_results['data']['lookup_tables']['compact_lookup']
        
        # Load solar radiation data
        solar_data = solar_results['sample_data']
        
        # Create combined dataset
        combined_records = []
        
        for _, pv_asset in pv_data.iterrows():
            # Find matching solar radiation data
            grid_lat = pv_asset['irradiance_nearest_lat']
            grid_lon = pv_asset['irradiance_nearest_lon']
            
            matching_solar = solar_data[
                (solar_data['latitude'] == grid_lat) &
                (solar_data['longitude'] == grid_lon)
            ]
            
            for _, solar_point in matching_solar.iterrows():
                combined_record = {
                    # PV asset information
                    'asset_id': pv_asset['asset_id'],
                    'pv_latitude': pv_asset['latitude'],
                    'pv_longitude': pv_asset['longitude'],
                    'capacity_kw': pv_asset['capacity_kw'],
                    'technology': pv_asset['technology'],
                    'tilt_angle': pv_asset['tilt_angle'],
                    'azimuth': pv_asset['azimuth'],
                    'h3_index': pv_asset['h3_index'],
                    'data_source': pv_asset['data_source'],
                    
                    # Solar radiation information
                    'time': solar_point['time'],
                    'solar_latitude': solar_point['latitude'],
                    'solar_longitude': solar_point['longitude'],
                    'ssrd': solar_point['ssrd'],
                    'era5_grid_id': solar_point['era5_grid_id'],
                    
                    # Spatial relationship
                    'distance_km': pv_asset['irradiance_distance_km'],
                    
                    # Estimated PV generation (simplified model)
                    'estimated_generation_kw': estimate_pv_generation(
                        pv_asset, solar_point
                    )
                }
                combined_records.append(combined_record)
        
        combined_df = pd.DataFrame(combined_records)
        
        # Save combined dataset
        output_file = Path("data_combined_pv_solar.parquet")
        combined_df.to_parquet(output_file, index=False)
        
        print(f"   ✅ Combined dataset created")
        print(f"   📊 {len(combined_df)} combined records")
        print(f"   🔗 {combined_df['asset_id'].nunique()} unique PV assets")
        print(f"   ⏰ {combined_df['time'].nunique()} unique time points")
        
        return {
            'success': True,
            'output_file': str(output_file),
            'total_records': len(combined_df),
            'unique_assets': combined_df['asset_id'].nunique(),
            'unique_times': combined_df['time'].nunique()
        }
        
    except Exception as e:
        print(f"   ❌ Combined dataset creation failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def estimate_pv_generation(pv_asset, solar_point):
    """Estimate PV generation based on solar radiation and panel characteristics."""
    # Simplified PV generation model
    ssrd = solar_point['ssrd']  # W/m²
    capacity = pv_asset['capacity_kw'] * 1000  # W
    
    # Basic efficiency factors
    panel_efficiency = 0.15  # 15% typical efficiency
    temperature_factor = 0.9  # Temperature derating
    soiling_factor = 0.95  # Soiling losses
    
    # Calculate generation
    generation = ssrd * panel_efficiency * temperature_factor * soiling_factor
    
    # Cap at panel capacity
    generation = min(generation, capacity)
    
    return max(0, generation / 1000)  # Convert back to kW

def main():
    """Main function to run the integrated pipeline."""
    print("🔬 Integrated PV Lookup Table + Solar Radiation Pipeline")
    print("Demonstrating complete solar forecasting system")
    print("This combines PV asset locations with solar radiation data...")
    
    results = run_integrated_pipeline()
    
    if results['success']:
        print("\n🎉 Integrated pipeline completed successfully!")
        print("📁 Check the output directories for results:")
        print("   - data_pv_lookup_integrated/ (PV lookup tables)")
        print("   - data_solar_integrated/ (Solar radiation data)")
        print("   - data_combined_pv_solar.parquet (Combined dataset)")
        print("\n🚀 Ready for solar forecasting applications!")
    else:
        print(f"\n❌ Integrated pipeline failed: {results.get('error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
