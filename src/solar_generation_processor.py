#!/usr/bin/env python3
"""
Solar Generation Data Processor
==============================

This module processes raw ENTSO-E solar generation data and converts it to
tidy Parquet format suitable for modeling.

Features:
- Data validation and quality checks
- Time series resampling and interpolation
- Missing data handling
- Data aggregation and summarization
- Export to various formats
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SolarGenerationProcessor:
    """
    Processor for ENTSO-E solar generation data.
    
    This class handles:
    - Data validation and quality checks
    - Time series processing and resampling
    - Missing data interpolation
    - Data aggregation and summarization
    - Export to modeling-ready formats
    """
    
    def __init__(self, input_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        """
        Initialize the solar generation processor.
        
        Parameters:
        -----------
        input_dir : Path, optional
            Directory containing raw data files
        output_dir : Path, optional
            Directory for processed output files
        """
        if input_dir is None:
            self.input_dir = Path("data_german_solar_generation/processed")
        else:
            self.input_dir = Path(input_dir)
            
        if output_dir is None:
            self.output_dir = Path("data_german_solar_generation/modeling")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Solar Generation Processor initialized")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_parquet_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from Parquet file.
        
        Parameters:
        -----------
        filename : str
            Name of the Parquet file
            
        Returns:
        --------
        pd.DataFrame
            Loaded DataFrame
        """
        file_path = self.input_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_parquet(file_path)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate data quality and completeness.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to validate
            
        Returns:
        --------
        Dict
            Validation results and statistics
        """
        logger.info("Validating data quality...")
        
        validation_results = {
            "total_records": len(df),
            "date_range": {
                "start": df['timestamp'].min().isoformat() if len(df) > 0 else None,
                "end": df['timestamp'].max().isoformat() if len(df) > 0 else None
            },
            "missing_values": {
                "solar_generation_mw": df['solar_generation_mw'].isna().sum(),
                "timestamp": df['timestamp'].isna().sum()
            },
            "data_quality": {
                "negative_values": (df['solar_generation_mw'] < 0).sum(),
                "zero_values": (df['solar_generation_mw'] == 0).sum(),
                "unreasonable_max": (df['solar_generation_mw'] > 50000).sum(),  # > 50 GW
                "unreasonable_min": (df['solar_generation_mw'] < -1000).sum()   # < -1 GW
            },
            "temporal_consistency": {
                "expected_15min_intervals": 0,
                "actual_intervals": 0,
                "gaps": 0
            }
        }
        
        # Check temporal consistency
        if len(df) > 1:
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff().dropna()
            
            # Count expected 15-minute intervals
            date_range = df_sorted['timestamp'].max() - df_sorted['timestamp'].min()
            expected_intervals = int(date_range.total_seconds() / (15 * 60)) + 1
            
            validation_results["temporal_consistency"]["expected_15min_intervals"] = expected_intervals
            validation_results["temporal_consistency"]["actual_intervals"] = len(df)
            validation_results["temporal_consistency"]["gaps"] = expected_intervals - len(df)
        
        # Log validation results
        logger.info(f"Validation complete:")
        logger.info(f"  Total records: {validation_results['total_records']}")
        logger.info(f"  Missing values: {validation_results['missing_values']['solar_generation_mw']}")
        logger.info(f"  Negative values: {validation_results['data_quality']['negative_values']}")
        logger.info(f"  Temporal gaps: {validation_results['temporal_consistency']['gaps']}")
        
        return validation_results
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw DataFrame to clean
            
        Returns:
        --------
        pd.DataFrame
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove negative values (replace with 0)
        negative_mask = df_clean['solar_generation_mw'] < 0
        if negative_mask.sum() > 0:
            logger.warning(f"Found {negative_mask.sum()} negative values, replacing with 0")
            df_clean.loc[negative_mask, 'solar_generation_mw'] = 0
        
        # Remove unreasonable values
        unreasonable_max_mask = df_clean['solar_generation_mw'] > 50000  # > 50 GW
        if unreasonable_max_mask.sum() > 0:
            logger.warning(f"Found {unreasonable_max_mask.sum()} unreasonable high values, replacing with NaN")
            df_clean.loc[unreasonable_max_mask, 'solar_generation_mw'] = np.nan
        
        # Sort by timestamp
        df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
        return df_clean
    
    def handle_missing_data(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing data using various methods.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with missing data
        method : str
            Method to use: 'interpolate', 'forward_fill', 'backward_fill', 'drop'
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with missing data handled
        """
        logger.info(f"Handling missing data using method: {method}")
        
        df_filled = df.copy()
        
        if method == 'interpolate':
            # Linear interpolation for missing values
            df_filled['solar_generation_mw'] = df_filled['solar_generation_mw'].interpolate(
                method='linear', limit_direction='both'
            )
        elif method == 'forward_fill':
            # Forward fill
            df_filled['solar_generation_mw'] = df_filled['solar_generation_mw'].fillna(method='ffill')
        elif method == 'backward_fill':
            # Backward fill
            df_filled['solar_generation_mw'] = df_filled['solar_generation_mw'].fillna(method='bfill')
        elif method == 'drop':
            # Drop rows with missing values
            df_filled = df_filled.dropna(subset=['solar_generation_mw'])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        missing_before = df['solar_generation_mw'].isna().sum()
        missing_after = df_filled['solar_generation_mw'].isna().sum()
        
        logger.info(f"Missing values: {missing_before} -> {missing_after}")
        return df_filled
    
    def resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 15-minute data to hourly data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 15-minute data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with hourly data
        """
        logger.info("Resampling to hourly data...")
        
        # Set timestamp as index for resampling
        df_indexed = df.set_index('timestamp')
        
        # Resample to hourly and calculate mean
        df_hourly = df_indexed['solar_generation_mw'].resample('H').mean().reset_index()
        
        # Add time components
        df_hourly['year'] = df_hourly['timestamp'].dt.year
        df_hourly['month'] = df_hourly['timestamp'].dt.month
        df_hourly['day'] = df_hourly['timestamp'].dt.day
        df_hourly['hour'] = df_hourly['timestamp'].dt.hour
        
        # Add metadata
        df_hourly['data_source'] = 'entsoe_transparency'
        df_hourly['resolution'] = 'hourly'
        df_hourly['bidding_zone'] = df['bidding_zone'].iloc[0] if len(df) > 0 else '10Y1001A1001A83F'
        
        logger.info(f"Resampled to {len(df_hourly)} hourly records")
        return df_hourly
    
    def resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 15-minute data to daily data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 15-minute data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with daily data
        """
        logger.info("Resampling to daily data...")
        
        # Set timestamp as index for resampling
        df_indexed = df.set_index('timestamp')
        
        # Resample to daily and calculate statistics
        daily_stats = df_indexed['solar_generation_mw'].resample('D').agg({
            'mean': 'mean',
            'max': 'max',
            'min': 'min',
            'sum': 'sum'  # Daily total generation
        }).reset_index()
        
        # Rename columns
        daily_stats.columns = ['timestamp', 'solar_generation_mw_mean', 'solar_generation_mw_max', 
                              'solar_generation_mw_min', 'solar_generation_mw_total']
        
        # Add time components
        daily_stats['year'] = daily_stats['timestamp'].dt.year
        daily_stats['month'] = daily_stats['timestamp'].dt.month
        daily_stats['day'] = daily_stats['timestamp'].dt.day
        daily_stats['day_of_year'] = daily_stats['timestamp'].dt.dayofyear
        
        # Add metadata
        daily_stats['data_source'] = 'entsoe_transparency'
        daily_stats['resolution'] = 'daily'
        daily_stats['bidding_zone'] = df['bidding_zone'].iloc[0] if len(df) > 0 else '10Y1001A1001A83F'
        
        logger.info(f"Resampled to {len(daily_stats)} daily records")
        return daily_stats
    
    def create_modeling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for modeling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional features
        """
        logger.info("Creating modeling features...")
        
        df_features = df.copy()
        
        # Time-based features
        df_features['hour_of_day'] = df_features['timestamp'].dt.hour
        df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
        df_features['month'] = df_features['timestamp'].dt.month
        df_features['day_of_year'] = df_features['timestamp'].dt.dayofyear
        
        # Cyclical encoding for time features
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour_of_day'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour_of_day'] / 24)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
        
        # Lag features (previous values)
        df_features['solar_generation_mw_lag1'] = df_features['solar_generation_mw'].shift(1)
        df_features['solar_generation_mw_lag4'] = df_features['solar_generation_mw'].shift(4)  # 1 hour ago
        df_features['solar_generation_mw_lag96'] = df_features['solar_generation_mw'].shift(96)  # 1 day ago
        
        # Rolling statistics
        df_features['solar_generation_mw_rolling_mean_4h'] = df_features['solar_generation_mw'].rolling(16).mean()  # 4 hours
        df_features['solar_generation_mw_rolling_std_4h'] = df_features['solar_generation_mw'].rolling(16).std()
        
        # Binary features
        df_features['is_daytime'] = ((df_features['hour_of_day'] >= 6) & (df_features['hour_of_day'] <= 18)).astype(int)
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        
        logger.info(f"Created {len(df_features.columns)} features")
        return df_features
    
    def save_modeling_data(self, df: pd.DataFrame, filename: str, format: str = 'parquet') -> Path:
        """
        Save processed data in modeling-ready format.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to save
        filename : str
            Base filename
        format : str
            Output format: 'parquet', 'csv', 'h5'
            
        Returns:
        --------
        Path
            Path to saved file
        """
        if len(df) == 0:
            logger.warning("Empty DataFrame, nothing to save")
            return None
        
        if format == 'parquet':
            output_path = self.output_dir / f"{filename}.parquet"
            df.to_parquet(output_path, engine='pyarrow', compression='snappy')
        elif format == 'csv':
            output_path = self.output_dir / f"{filename}.csv"
            df.to_csv(output_path, index=False)
        elif format == 'h5':
            output_path = self.output_dir / f"{filename}.h5"
            df.to_hdf(output_path, key='solar_generation', mode='w')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data saved to {output_path}")
        return output_path
    
    def process_pipeline(self, input_filename: str, output_filename: str) -> Dict:
        """
        Run the complete data processing pipeline.
        
        Parameters:
        -----------
        input_filename : str
            Name of input Parquet file
        output_filename : str
            Base name for output files
            
        Returns:
        --------
        Dict
            Processing results and statistics
        """
        logger.info("Starting data processing pipeline...")
        
        # Load data
        df = self.load_parquet_data(input_filename)
        
        # Validate data
        validation_results = self.validate_data(df)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Handle missing data
        df_filled = self.handle_missing_data(df_clean, method='interpolate')
        
        # Create different resolutions
        df_hourly = self.resample_to_hourly(df_filled)
        df_daily = self.resample_to_daily(df_filled)
        
        # Create modeling features
        df_features = self.create_modeling_features(df_filled)
        
        # Save processed data
        files_saved = {}
        
        # Save 15-minute data
        files_saved['quarter_hourly'] = self.save_modeling_data(
            df_filled, f"{output_filename}_15min", 'parquet'
        )
        
        # Save hourly data
        files_saved['hourly'] = self.save_modeling_data(
            df_hourly, f"{output_filename}_hourly", 'parquet'
        )
        
        # Save daily data
        files_saved['daily'] = self.save_modeling_data(
            df_daily, f"{output_filename}_daily", 'parquet'
        )
        
        # Save features data
        files_saved['features'] = self.save_modeling_data(
            df_features, f"{output_filename}_features", 'parquet'
        )
        
        # Generate summary
        summary = {
            "input_records": len(df),
            "output_records": {
                "quarter_hourly": len(df_filled),
                "hourly": len(df_hourly),
                "daily": len(df_daily),
                "features": len(df_features)
            },
            "validation": validation_results,
            "files_saved": files_saved
        }
        
        logger.info("Data processing pipeline completed successfully")
        return summary


def main():
    """Main function for testing the processor."""
    print("="*80)
    print("🔧 Solar Generation Data Processor")
    print("="*80)
    
    # Initialize processor
    processor = SolarGenerationProcessor()
    
    # Test with sample data (if available)
    test_file = "german_solar_generation_test.parquet"
    
    if (processor.input_dir / test_file).exists():
        print(f"\n📊 Processing test file: {test_file}")
        
        try:
            # Run processing pipeline
            results = processor.process_pipeline(test_file, "german_solar_generation_processed")
            
            print(f"\n✅ Processing completed successfully!")
            print(f"📈 Results:")
            print(f"   Input records: {results['input_records']}")
            print(f"   Quarter-hourly: {results['output_records']['quarter_hourly']}")
            print(f"   Hourly: {results['output_records']['hourly']}")
            print(f"   Daily: {results['output_records']['daily']}")
            print(f"   Features: {results['output_records']['features']}")
            
            print(f"\n💾 Files saved:")
            for resolution, file_path in results['files_saved'].items():
                print(f"   {resolution}: {file_path}")
                
        except Exception as e:
            print(f"❌ Error during processing: {e}")
    else:
        print(f"\n⚠️  Test file not found: {test_file}")
        print("💡 Run the ENTSO-E collector first to generate test data")


if __name__ == "__main__":
    main()
