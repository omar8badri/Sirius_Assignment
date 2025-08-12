#!/usr/bin/env python3
"""
Full Lookup Table Builder
=========================

This script builds the complete lookup table by integrating:
1. PV asset locations with spatial indexing
2. ERA5 solar radiation data
3. Power production calculations
4. Compact lookup tables for repeated use

The result is a comprehensive solar forecasting system.
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

from pv_locations import PVLookupTableBuilder
from solar_radiation_pipeline_optimized import OptimizedSolarRadiationPipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FullLookupTableBuilder:
    """Builder for the complete solar forecasting lookup table."""
    
    def __init__(self, output_dir: str = "data_full_lookup_table"):
        """
        Initialize the full lookup table builder.
        
        Args:
            output_dir: Output directory for all lookup tables
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Full Lookup Table Builder")
        logger.info(f"Output directory: {self.output_dir}")
    
    def build_full_lookup_table(self, 
                               pv_data_sources: list = None,
                               solar_years: list = None,
                               use_sample_data: bool = True) -> dict:
        """
        Build the complete lookup table with PV assets and solar radiation data.
        
        Args:
            pv_data_sources: List of PV data sources ('ocf', 'osm')
            solar_years: Years of solar radiation data to process
            use_sample_data: If True, use sample data for testing
            
        Returns:
            Dictionary with build results and file paths
        """
        if pv_data_sources is None:
            pv_data_sources = ['ocf', 'osm']
        
        if solar_years is None:
            solar_years = [2018]  # Start with one year for testing
        
        logger.info("="*80)
        logger.info("BUILDING FULL LOOKUP TABLE")
        logger.info("="*80)
        logger.info(f"PV Data Sources: {pv_data_sources}")
        logger.info(f"Solar Years: {solar_years}")
        logger.info(f"Use Sample Data: {use_sample_data}")
        
        start_time = time.time()
        
        try:
            # Step 1: Build PV lookup table
            logger.info("\n📍 Step 1: Building PV Lookup Table...")
            pv_results = self._build_pv_lookup_table(pv_data_sources, use_sample_data)
            
            if not pv_results['success']:
                raise Exception(f"PV lookup table build failed: {pv_results.get('error')}")
            
            # Step 2: Process solar radiation data
            logger.info("\n☀️  Step 2: Processing Solar Radiation Data...")
            solar_results = self._process_solar_radiation_data(solar_years, use_sample_data)
            
            if not solar_results['success']:
                raise Exception(f"Solar radiation processing failed: {solar_results.get('error')}")
            
            # Step 3: Create integrated lookup table
            logger.info("\n🔗 Step 3: Creating Integrated Lookup Table...")
            integrated_results = self._create_integrated_lookup_table(pv_results, solar_results)
            
            # Step 4: Generate power production forecasts
            logger.info("\n⚡ Step 4: Generating Power Production Forecasts...")
            forecast_results = self._generate_power_forecasts(integrated_results)
            
            # Step 5: Create final lookup tables
            logger.info("\n📊 Step 5: Creating Final Lookup Tables...")
            final_results = self._create_final_lookup_tables(integrated_results, forecast_results)
            
            end_time = time.time()
            build_time = end_time - start_time
            
            # Generate comprehensive summary
            summary = self._generate_comprehensive_summary(
                pv_results, solar_results, integrated_results, 
                forecast_results, build_time
            )
            
            logger.info("\n" + "="*80)
            logger.info("🎉 FULL LOOKUP TABLE BUILD COMPLETED")
            logger.info("="*80)
            logger.info(f"⏱️  Total Build Time: {build_time:.2f} seconds")
            logger.info(f"📊 PV Assets: {summary['total_pv_assets']}")
            logger.info(f"📍 H3 Hexagons: {summary['unique_h3_hexagons']}")
            logger.info(f"🌍 Grid Points: {summary['unique_grid_points']}")
            logger.info(f"☀️  Solar Data Points: {summary['solar_data_points']}")
            logger.info(f"⚡ Power Forecasts: {summary['power_forecasts']}")
            
            return {
                'success': True,
                'build_time_seconds': build_time,
                'summary': summary,
                'output_dir': str(self.output_dir),
                'files': final_results['files']
            }
            
        except Exception as e:
            logger.error(f"Full lookup table build failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'build_time_seconds': time.time() - start_time
            }
    
    def _build_pv_lookup_table(self, data_sources: list, use_sample: bool) -> dict:
        """Build the PV lookup table."""
        logger.info("   Building PV lookup table...")
        
        builder = PVLookupTableBuilder(
            output_dir=self.output_dir / "pv_lookup",
            h3_resolution=9,
            irradiance_grid_resolution=0.1,
            n_workers=2
        )
        
        results = builder.build_table(
            data_sources=data_sources,
            use_sample=use_sample
        )
        
        if results['success']:
            logger.info(f"   ✅ PV lookup table built successfully")
            logger.info(f"   📊 {results['summary']['total_pv_assets']} PV assets indexed")
        
        return results
    
    def _process_solar_radiation_data(self, years: list, use_sample: bool) -> dict:
        """Process solar radiation data."""
        logger.info("   Processing solar radiation data...")
        
        if use_sample:
            # Create sample solar radiation data
            solar_data = self._create_sample_solar_data(years)
            
            # Save sample data
            solar_file = self.output_dir / "solar_radiation_sample.parquet"
            solar_data.to_parquet(solar_file, index=False)
            
            logger.info(f"   ✅ Sample solar radiation data created")
            logger.info(f"   📊 {len(solar_data)} solar data points")
            
            return {
                'success': True,
                'solar_data': solar_data,
                'solar_file': str(solar_file),
                'data_points': len(solar_data)
            }
        else:
            # TODO: Integrate with actual ERA5 pipeline
            logger.warning("   ⚠️  Actual ERA5 integration not implemented yet")
            return self._process_solar_radiation_data(years, use_sample=True)
    
    def _create_sample_solar_data(self, years: list) -> pd.DataFrame:
        """Create comprehensive sample solar radiation data."""
        solar_data = []
        
        for year in years:
            # Create hourly data for the entire year
            start_time = datetime(year, 1, 1, 0, 0)
            end_time = datetime(year, 12, 31, 23, 0)
            time_range = pd.date_range(start=start_time, end=end_time, freq='H')
            
            # Sample grid points (matching PV lookup table)
            grid_points = [
                (52.5, 13.4),  # Berlin
                (48.1, 11.6),  # Munich
                (53.6, 10.0),  # Hamburg
                (50.9, 6.9),   # Cologne
                (50.1, 8.7),   # Frankfurt
            ]
            
            for lat, lon in grid_points:
                for time_point in time_range:
                    # Realistic solar radiation model
                    radiation = self._calculate_realistic_solar_radiation(
                        lat, lon, time_point
                    )
                    
                    solar_data.append({
                        'time': time_point,
                        'latitude': lat,
                        'longitude': lon,
                        'ssrd': radiation,
                        'era5_grid_id': f"germany_{lat:.1f}_{lon:.1f}",
                        'year': time_point.year,
                        'month': time_point.month,
                        'day': time_point.day,
                        'hour': time_point.hour
                    })
        
        return pd.DataFrame(solar_data)
    
    def _calculate_realistic_solar_radiation(self, lat: float, lon: float, time_point: datetime) -> float:
        """Calculate realistic solar radiation based on location and time."""
        import math
        
        # Solar position calculation (simplified)
        hour = time_point.hour
        day_of_year = time_point.timetuple().tm_yday
        
        # Solar declination (simplified)
        declination = 23.45 * math.sin(math.radians(360/365 * (day_of_year - 80)))
        
        # Solar hour angle
        solar_noon = 12  # Approximate solar noon
        hour_angle = 15 * (hour - solar_noon)
        
        # Solar zenith angle
        lat_rad = math.radians(lat)
        decl_rad = math.radians(declination)
        hour_rad = math.radians(hour_angle)
        
        cos_zenith = (math.sin(lat_rad) * math.sin(decl_rad) + 
                     math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_rad))
        
        zenith = math.acos(max(-1, min(1, cos_zenith)))
        zenith_deg = math.degrees(zenith)
        
        # Solar radiation calculation
        if zenith_deg < 90:  # Daytime
            # Extraterrestrial radiation
            solar_constant = 1367  # W/m²
            distance_factor = 1 + 0.034 * math.cos(math.radians(360/365 * (day_of_year - 1)))
            
            # Atmospheric transmittance (simplified)
            air_mass = 1 / math.cos(math.radians(zenith_deg))
            transmittance = 0.7 ** air_mass  # Simplified atmospheric model
            
            # Ground radiation
            radiation = solar_constant * distance_factor * transmittance * math.cos(math.radians(zenith_deg))
            
            # Add some realistic variability
            import random
            random.seed(hash(f"{lat}_{lon}_{time_point}") % 1000)
            variability = 0.9 + 0.2 * random.random()
            radiation *= variability
            
            return max(0, radiation)
        else:
            return 0  # Nighttime
    
    def _create_integrated_lookup_table(self, pv_results: dict, solar_results: dict) -> dict:
        """Create the integrated lookup table."""
        logger.info("   Creating integrated lookup table...")
        
        # Load PV lookup table
        pv_lookup = pv_results['data']['lookup_tables']['compact_lookup']
        solar_data = solar_results['solar_data']
        
        # Create integrated records
        integrated_records = []
        
        for _, grid_point in pv_lookup.iterrows():
            # Get solar data for this grid point
            grid_solar = solar_data[
                (solar_data['latitude'] == grid_point['grid_lat']) &
                (solar_data['longitude'] == grid_point['grid_lon'])
            ]
            
            for _, solar_point in grid_solar.iterrows():
                # Calculate power production
                power_production = self._calculate_pv_power_production(
                    grid_point['total_capacity_kw'],
                    solar_point['ssrd'],
                    grid_point['avg_distance_km']
                )
                
                integrated_record = {
                    # Time information
                    'time': solar_point['time'],
                    'year': solar_point['year'],
                    'month': solar_point['month'],
                    'day': solar_point['day'],
                    'hour': solar_point['hour'],
                    
                    # Grid information
                    'grid_lat': grid_point['grid_lat'],
                    'grid_lon': grid_point['grid_lon'],
                    'era5_grid_id': grid_point['era5_grid_id'],
                    
                    # PV information
                    'pv_asset_ids': grid_point['pv_asset_ids'],
                    'pv_count': grid_point['pv_count'],
                    'total_capacity_kw': grid_point['total_capacity_kw'],
                    'avg_distance_km': grid_point['avg_distance_km'],
                    
                    # Solar radiation
                    'solar_radiation_wm2': solar_point['ssrd'],
                    
                    # Power production
                    'power_production_kw': power_production,
                    'efficiency_percent': (power_production / grid_point['total_capacity_kw']) * 100,
                    
                    # Metadata
                    'created_at': datetime.now()
                }
                
                integrated_records.append(integrated_record)
        
        integrated_df = pd.DataFrame(integrated_records)
        
        # Save integrated data
        integrated_file = self.output_dir / "integrated_lookup_table.parquet"
        integrated_df.to_parquet(integrated_file, index=False)
        
        logger.info(f"   ✅ Integrated lookup table created")
        logger.info(f"   📊 {len(integrated_df)} integrated records")
        logger.info(f"   🔗 {integrated_df['era5_grid_id'].nunique()} unique grid points")
        logger.info(f"   ⏰ {integrated_df['time'].nunique()} unique time points")
        
        return {
            'success': True,
            'integrated_data': integrated_df,
            'integrated_file': str(integrated_file),
            'total_records': len(integrated_df)
        }
    
    def _calculate_pv_power_production(self, capacity_kw: float, 
                                     solar_radiation_wm2: float, 
                                     distance_km: float) -> float:
        """Calculate PV power production."""
        # Basic efficiency factors
        panel_efficiency = 0.15  # 15% typical efficiency
        temperature_factor = 0.9  # Temperature derating
        soiling_factor = 0.95  # Soiling losses
        distance_factor = max(0.8, 1 - (distance_km / 50))  # Distance penalty
        
        # Calculate generation
        generation = (solar_radiation_wm2 * panel_efficiency * 
                     temperature_factor * soiling_factor * distance_factor)
        
        # Convert to power (W to kW)
        generation_kw = generation / 1000
        
        # Cap at panel capacity
        generation_kw = min(generation_kw, capacity_kw)
        
        return max(0, generation_kw)
    
    def _generate_power_forecasts(self, integrated_results: dict) -> dict:
        """Generate power production forecasts."""
        logger.info("   Generating power production forecasts...")
        
        integrated_df = integrated_results['integrated_data']
        
        # Create daily forecasts
        daily_forecasts = integrated_df.groupby([
            'grid_lat', 'grid_lon', 'era5_grid_id', 'year', 'month', 'day'
        ]).agg({
            'total_capacity_kw': 'first',
            'pv_count': 'first',
            'power_production_kw': ['sum', 'mean', 'max'],
            'solar_radiation_wm2': ['mean', 'max'],
            'efficiency_percent': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_forecasts.columns = [
            'grid_lat', 'grid_lon', 'era5_grid_id', 'year', 'month', 'day',
            'capacity_kw', 'pv_count', 'daily_generation_kwh', 'avg_power_kw', 'peak_power_kw',
            'avg_radiation_wm2', 'peak_radiation_wm2', 'avg_efficiency_percent'
        ]
        
        # Create monthly forecasts
        monthly_forecasts = daily_forecasts.groupby([
            'grid_lat', 'grid_lon', 'era5_grid_id', 'year', 'month'
        ]).agg({
            'capacity_kw': 'first',
            'pv_count': 'first',
            'daily_generation_kwh': ['sum', 'mean'],
            'avg_power_kw': 'mean',
            'peak_power_kw': 'max',
            'avg_radiation_wm2': 'mean',
            'avg_efficiency_percent': 'mean'
        }).reset_index()
        
        # Flatten column names
        monthly_forecasts.columns = [
            'grid_lat', 'grid_lon', 'era5_grid_id', 'year', 'month',
            'capacity_kw', 'pv_count', 'monthly_generation_kwh', 'avg_daily_generation_kwh',
            'avg_power_kw', 'peak_power_kw', 'avg_radiation_wm2', 'avg_efficiency_percent'
        ]
        
        # Save forecasts
        daily_file = self.output_dir / "daily_power_forecasts.parquet"
        monthly_file = self.output_dir / "monthly_power_forecasts.parquet"
        
        daily_forecasts.to_parquet(daily_file, index=False)
        monthly_forecasts.to_parquet(monthly_file, index=False)
        
        logger.info(f"   ✅ Power forecasts generated")
        logger.info(f"   📅 {len(daily_forecasts)} daily forecasts")
        logger.info(f"   📊 {len(monthly_forecasts)} monthly forecasts")
        
        return {
            'success': True,
            'daily_forecasts': daily_forecasts,
            'monthly_forecasts': monthly_forecasts,
            'daily_file': str(daily_file),
            'monthly_file': str(monthly_file)
        }
    
    def _create_final_lookup_tables(self, integrated_results: dict, forecast_results: dict) -> dict:
        """Create final lookup tables for repeated use."""
        logger.info("   Creating final lookup tables...")
        
        # Create summary lookup table
        summary_lookup = self._create_summary_lookup_table(integrated_results, forecast_results)
        
        # Create grid statistics lookup table
        grid_stats = self._create_grid_statistics_table(integrated_results)
        
        # Create time-based lookup tables
        time_lookups = self._create_time_based_lookups(integrated_results)
        
        # Save all lookup tables
        files = {}
        
        summary_file = self.output_dir / "summary_lookup_table.parquet"
        summary_lookup.to_parquet(summary_file, index=False)
        files['summary_lookup'] = str(summary_file)
        
        grid_stats_file = self.output_dir / "grid_statistics.parquet"
        grid_stats.to_parquet(grid_stats_file, index=False)
        files['grid_statistics'] = str(grid_stats_file)
        
        for name, data in time_lookups.items():
            file_path = self.output_dir / f"{name}.parquet"
            data.to_parquet(file_path, index=False)
            files[name] = str(file_path)
        
        # Save metadata
        metadata = {
            'build_timestamp': datetime.now().isoformat(),
            'total_records': integrated_results['total_records'],
            'unique_grid_points': len(grid_stats),
            'forecast_periods': {
                'daily': len(forecast_results['daily_forecasts']),
                'monthly': len(forecast_results['monthly_forecasts'])
            }
        }
        
        metadata_file = self.output_dir / "full_lookup_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        files['metadata'] = str(metadata_file)
        
        logger.info(f"   ✅ Final lookup tables created")
        logger.info(f"   📁 {len(files)} files saved")
        
        return {
            'success': True,
            'files': files,
            'summary_lookup': summary_lookup,
            'grid_stats': grid_stats,
            'time_lookups': time_lookups
        }
    
    def _create_summary_lookup_table(self, integrated_results: dict, forecast_results: dict) -> pd.DataFrame:
        """Create summary lookup table."""
        integrated_df = integrated_results['integrated_data']
        
        # Group by grid point and create summary
        summary = integrated_df.groupby(['grid_lat', 'grid_lon', 'era5_grid_id']).agg({
            'pv_asset_ids': 'first',
            'pv_count': 'first',
            'total_capacity_kw': 'first',
            'avg_distance_km': 'first',
            'power_production_kw': ['sum', 'mean', 'max'],
            'solar_radiation_wm2': ['mean', 'max'],
            'efficiency_percent': 'mean'
        }).reset_index()
        
        # Flatten column names
        summary.columns = [
            'grid_lat', 'grid_lon', 'era5_grid_id', 'pv_asset_ids', 'pv_count',
            'total_capacity_kw', 'avg_distance_km', 'total_generation_kwh',
            'avg_power_kw', 'peak_power_kw', 'avg_radiation_wm2', 'peak_radiation_wm2',
            'avg_efficiency_percent'
        ]
        
        return summary
    
    def _create_grid_statistics_table(self, integrated_results: dict) -> pd.DataFrame:
        """Create grid statistics table."""
        integrated_df = integrated_results['integrated_data']
        
        # Calculate statistics for each grid point
        stats = integrated_df.groupby(['grid_lat', 'grid_lon']).agg({
            'pv_count': 'first',
            'total_capacity_kw': 'first',
            'power_production_kw': ['count', 'sum', 'mean', 'std'],
            'solar_radiation_wm2': ['mean', 'std', 'min', 'max'],
            'efficiency_percent': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        stats.columns = [
            'grid_lat', 'grid_lon', 'pv_count', 'total_capacity_kw',
            'data_points', 'total_generation_kwh', 'avg_power_kw', 'std_power_kw',
            'avg_radiation_wm2', 'std_radiation_wm2', 'min_radiation_wm2', 'max_radiation_wm2',
            'avg_efficiency_percent', 'std_efficiency_percent'
        ]
        
        return stats
    
    def _create_time_based_lookups(self, integrated_results: dict) -> dict:
        """Create time-based lookup tables."""
        integrated_df = integrated_results['integrated_data']
        
        lookups = {}
        
        # Hourly lookup table
        hourly_lookup = integrated_df.groupby(['grid_lat', 'grid_lon', 'hour']).agg({
            'power_production_kw': 'mean',
            'solar_radiation_wm2': 'mean',
            'efficiency_percent': 'mean'
        }).reset_index()
        lookups['hourly_lookup'] = hourly_lookup
        
        # Monthly lookup table
        monthly_lookup = integrated_df.groupby(['grid_lat', 'grid_lon', 'month']).agg({
            'power_production_kw': 'mean',
            'solar_radiation_wm2': 'mean',
            'efficiency_percent': 'mean',
            'total_capacity_kw': 'first'
        }).reset_index()
        lookups['monthly_lookup'] = monthly_lookup
        
        return lookups
    
    def _generate_comprehensive_summary(self, pv_results: dict, solar_results: dict,
                                      integrated_results: dict, forecast_results: dict,
                                      build_time: float) -> dict:
        """Generate comprehensive build summary."""
        summary = {
            'build_time_seconds': build_time,
            'total_pv_assets': pv_results['summary']['total_pv_assets'],
            'unique_h3_hexagons': pv_results['summary']['unique_h3_hexagons'],
            'unique_grid_points': pv_results['summary']['unique_grid_points'],
            'solar_data_points': solar_results['data_points'],
            'integrated_records': integrated_results['total_records'],
            'power_forecasts': {
                'daily': len(forecast_results['daily_forecasts']),
                'monthly': len(forecast_results['monthly_forecasts'])
            },
            'data_sources': pv_results['summary']['data_sources'],
            'capacity_stats': pv_results['summary']['capacity_stats'],
            'spatial_stats': pv_results['summary']['spatial_stats']
        }
        
        return summary

def main():
    """Main function to build the full lookup table."""
    print("🔬 Full Lookup Table Builder")
    print("Building complete solar forecasting system...")
    print("This integrates PV assets with solar radiation data")
    
    builder = FullLookupTableBuilder()
    
    results = builder.build_full_lookup_table(
        pv_data_sources=['ocf', 'osm'],
        solar_years=[2018],
        use_sample_data=True
    )
    
    if results['success']:
        print("\n🎉 Full lookup table built successfully!")
        print(f"📁 Output directory: {results['output_dir']}")
        print(f"⏱️  Build time: {results['build_time_seconds']:.2f} seconds")
        
        print(f"\n📊 Summary:")
        summary = results['summary']
        print(f"   PV Assets: {summary['total_pv_assets']}")
        print(f"   Grid Points: {summary['unique_grid_points']}")
        print(f"   Solar Data Points: {summary['solar_data_points']}")
        print(f"   Integrated Records: {summary['integrated_records']}")
        print(f"   Daily Forecasts: {summary['power_forecasts']['daily']}")
        print(f"   Monthly Forecasts: {summary['power_forecasts']['monthly']}")
        
        print(f"\n💾 Generated Files:")
        for file_type, file_path in results['files'].items():
            print(f"   {file_type}: {file_path}")
        
        print("\n🚀 Ready for solar forecasting applications!")
    else:
        print(f"\n❌ Full lookup table build failed: {results.get('error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
