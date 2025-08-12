#!/usr/bin/env python3
"""
Baseline Pipeline with Extended Data
====================================

This pipeline focuses only on the baseline model but uses significantly more data points
by reducing the sampling rate from 1% to 10% or more.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselinePipelineExtended:
    """Focused baseline pipeline with extended data points."""
    
    def __init__(self, output_dir: str = "baseline_results_extended"):
        """Initialize the extended baseline pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline profiler
        self.start_time = time.time()
        self.step_times = {}
        
        logger.info("="*80)
        logger.info("🚀 BASELINE PIPELINE WITH EXTENDED DATA")
        logger.info("="*80)
        logger.info("Focus: Baseline model only with maximum data points")
        logger.info("="*80)
    
    def run_pipeline(self):
        """Run the complete baseline pipeline with extended data."""
        try:
            # Step 1: Load extended radiation data
            logger.info("\n📥 STEP 1: Loading Extended Radiation Data")
            logger.info("-" * 60)
            radiation_data = self._load_extended_radiation_data()
            
            # Step 2: Load PV locations
            logger.info("\n📍 STEP 2: Loading PV Locations")
            logger.info("-" * 60)
            pv_locations = self._load_pv_locations()
            
            # Step 3: Create synthetic generation
            logger.info("\n⚡ STEP 3: Creating Synthetic Generation")
            logger.info("-" * 60)
            generation_data = self._create_synthetic_generation(radiation_data)
            
            # Step 4: Merge and preprocess data
            logger.info("\n🔄 STEP 4: Merging and Preprocessing Data")
            logger.info("-" * 60)
            merged_data = self._merge_and_preprocess_data(radiation_data, generation_data)
            
            # Step 5: Simple feature engineering for baseline
            logger.info("\n🔧 STEP 5: Simple Feature Engineering")
            logger.info("-" * 60)
            features_data = self._create_simple_features(merged_data)
            
            # Step 6: Train baseline model
            logger.info("\n🎯 STEP 6: Training Baseline Model")
            logger.info("-" * 60)
            model, X_train, y_train = self._train_baseline_model(features_data)
            
            # Step 7: Generate predictions
            logger.info("\n📊 STEP 7: Generating Predictions")
            logger.info("-" * 60)
            predictions = self._generate_predictions(model, X_train, y_train)
            
            # Step 8: Save results
            logger.info("\n💾 STEP 8: Saving Results")
            logger.info("-" * 60)
            self._save_results(predictions, model, X_train, y_train)
            
            # Final summary
            self._print_final_summary()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_extended_radiation_data(self) -> pd.DataFrame:
        """Load radiation data with increased sampling (10% instead of 1%)."""
        logger.info("Loading extended radiation data (10% sampling)...")
        
        base_path = Path("data_3years_2018_2020_final")
        all_data = []
        
        # Load monthly data (2018) with increased sampling
        monthly_path = base_path / "monthly_15min_results"
        if monthly_path.exists():
            logger.info("Loading monthly 15-minute radiation data (10% sampling)...")
            for month_dir in sorted(monthly_path.glob("ssrd_germany_2018_*_15min")):
                try:
                    month_num = int(month_dir.name.split("_")[3])
                except ValueError:
                    logger.warning(f"Could not parse month from directory name: {month_dir.name}")
                    continue
                
                logger.info(f"Processing month {month_num}...")
                
                # Find the Parquet file within the year/month structure
                parquet_files = list(month_dir.rglob("*.parquet"))
                if parquet_files:
                    df = pd.read_parquet(parquet_files[0])
                    
                    # Increased sampling: 10% instead of 1%
                    sample_size = max(50000, len(df) // 10)  # At least 50k records, or 10%
                    df = df.sample(n=sample_size, random_state=42)
                    
                    df['month'] = month_num
                    df['year'] = 2018
                    all_data.append(df)
                    logger.info(f"  Loaded {len(df):,} records for 2018-{month_num:02d} (10% sampling)")
        
        # Load quarterly data (2018) with increased sampling - fixed parsing
        quarterly_path = base_path / "quarterly_15min_results"
        if quarterly_path.exists():
            logger.info("Loading quarterly 15-minute radiation data (10% sampling)...")
            for quarter_base_dir in sorted(quarterly_path.glob("ssrd_germany_*_15min")):
                # Parse directory name correctly: ssrd_germany_2018_Q3_15min
                parts = quarter_base_dir.name.split("_")
                if len(parts) >= 3:
                    try:
                        year_from_dir = int(parts[2])  # 2018
                        quarter_from_dir = parts[3]    # Q3, Q4
                        logger.info(f"Processing {year_from_dir} {quarter_from_dir}...")
                        
                        # Iterate through year=X/month=Y subdirectories
                        for year_dir in sorted(quarter_base_dir.glob("year=*")):
                            year = int(year_dir.name.split("=")[1])
                            for month_dir in sorted(year_dir.glob("month=*")):
                                month = int(month_dir.name.split("=")[1])
                                
                                parquet_files = list(month_dir.glob("*.parquet"))
                                if parquet_files:
                                    file_path = parquet_files[0]
                                    try:
                                        df = pd.read_parquet(file_path, engine='pyarrow')
                                        
                                        # Increased sampling: 10% instead of 1%
                                        sample_size = max(50000, len(df) // 10)  # At least 50k records, or 10%
                                        df = df.sample(n=sample_size, random_state=42)
                                        
                                        if 'year' not in df.columns:
                                            df['year'] = year
                                        if 'month' not in df.columns:
                                            df['month'] = month
                                        all_data.append(df)
                                        logger.info(f"    Loaded {len(df):,} records from {file_path.name} (10% sampling)")
                                    except Exception as e:
                                        logger.warning(f"    Error loading {file_path}: {e}")
                                        continue
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse directory name {quarter_base_dir.name}: {e}")
                        continue
        
        if not all_data:
            raise ValueError("No radiation data found!")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('time').reset_index(drop=True)
        
        # Standardize column names
        if 'ssrd' in combined_df.columns:
            # Convert from J/m² to W/m² (divide by 3600 seconds)
            combined_df['irradiance_w_m2'] = combined_df['ssrd'] / 3600
        elif 'ssrd_w_m2' in combined_df.columns:
            combined_df['irradiance_w_m2'] = combined_df['ssrd_w_m2']
        else:
            raise ValueError("No irradiance column found in data")
        
        # Rename time column
        if 'time' in combined_df.columns:
            combined_df = combined_df.rename(columns={'time': 'timestamp'})
        
        # Select relevant columns
        result_df = combined_df[['timestamp', 'irradiance_w_m2']].copy()
        
        logger.info(f"Combined radiation data: {len(result_df):,} records (10% sampling)")
        logger.info(f"Time range: {result_df['timestamp'].min()} to {result_df['timestamp'].max()}")
        logger.info(f"Irradiance range: {result_df['irradiance_w_m2'].min():.2f} to {result_df['irradiance_w_m2'].max():.2f} W/m²")
        
        return result_df
    
    def _load_pv_locations(self) -> pd.DataFrame:
        """Load PV location data."""
        # Use comprehensive lookup table
        comprehensive_path = Path("data_pv_lookup_comprehensive")
        primary_lookup_path = comprehensive_path / "primary_lookup.parquet"
        
        if primary_lookup_path.exists():
            pv_data = pd.read_parquet(primary_lookup_path)
            logger.info(f"Loaded {len(pv_data):,} PV locations from comprehensive lookup")
        else:
            raise ValueError("Comprehensive PV lookup table not found!")
        
        logger.info(f"PV locations span: lat {pv_data['latitude'].min():.2f} to {pv_data['latitude'].max():.2f}")
        logger.info(f"PV locations span: lon {pv_data['longitude'].min():.2f} to {pv_data['longitude'].max():.2f}")
        logger.info(f"Total PV capacity: {pv_data['capacity_kw'].sum()/1000:.1f} MW")
        
        return pv_data
    
    def _create_synthetic_generation(self, radiation_df: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic solar generation data based on real radiation data."""
        generation_data = radiation_df.copy()
        
        # Calculate synthetic generation using realistic parameters
        system_efficiency = 0.15  # 15% typical panel efficiency
        capacity_factor = 0.12   # 12% typical capacity factor for Germany
        base_capacity = 50000    # 50,000 MW total German capacity
        
        # Calculate generation based on irradiance
        generation_data['solar_generation_mw'] = (
            generation_data['irradiance_w_m2'] * 
            system_efficiency * 
            capacity_factor * 
            base_capacity / 1000  # Convert to MW
        )
        
        # Add realistic noise and variations
        np.random.seed(42)  # For reproducibility
        noise_factor = np.random.normal(1.0, 0.1, len(generation_data))
        generation_data['solar_generation_mw'] *= noise_factor
        
        # Ensure non-negative values
        generation_data['solar_generation_mw'] = np.maximum(0, generation_data['solar_generation_mw'])
        
        # Cap at realistic maximum
        max_generation = base_capacity * 0.8  # 80% of capacity
        generation_data['solar_generation_mw'] = np.minimum(
            generation_data['solar_generation_mw'], 
            max_generation
        )
        
        logger.info(f"Created synthetic generation data: {len(generation_data):,} records")
        logger.info(f"Generation range: {generation_data['solar_generation_mw'].min():.2f} to {generation_data['solar_generation_mw'].max():.2f} MW")
        
        return generation_data
    
    def _merge_and_preprocess_data(self, radiation_df: pd.DataFrame, generation_df: pd.DataFrame) -> pd.DataFrame:
        """Merge radiation and generation data with memory-efficient processing."""
        logger.info("Merging radiation and generation datasets (memory-efficient)...")
        
        # Since both DataFrames are already aligned (generation_df was created from radiation_df),
        # we can simply add the generation column to the radiation DataFrame
        merged_data = radiation_df.copy()
        merged_data['solar_generation_mw'] = generation_df['solar_generation_mw']
        
        logger.info(f"Merged dataset: {len(merged_data):,} records")
        
        # Filter night-time data
        initial_count = len(merged_data)
        merged_data = merged_data[merged_data['irradiance_w_m2'] >= 10.0]
        final_count = len(merged_data)
        logger.info(f"Filtered night-time data: {initial_count - final_count} records removed")
        
        return merged_data
    
    def _create_simple_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create simple features for baseline model with memory-efficient processing."""
        logger.info("Creating simple features for baseline model...")
        
        # Process in chunks to avoid memory issues
        chunk_size = 1000000  # 1M records per chunk
        total_chunks = len(data) // chunk_size + 1
        
        all_features = []
        
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(data))
            
            chunk = data.iloc[start_idx:end_idx].copy()
            
            # Basic time features
            chunk['hour'] = chunk['timestamp'].dt.hour
            chunk['month'] = chunk['timestamp'].dt.month
            chunk['day_of_year'] = chunk['timestamp'].dt.dayofyear
            
            # Cyclical encoding for time features
            chunk['hour_sin'] = np.sin(2 * np.pi * chunk['hour'] / 24)
            chunk['hour_cos'] = np.cos(2 * np.pi * chunk['hour'] / 24)
            chunk['month_sin'] = np.sin(2 * np.pi * chunk['month'] / 12)
            chunk['month_cos'] = np.cos(2 * np.pi * chunk['month'] / 12)
            
            # Daytime indicator
            chunk['is_daytime'] = ((chunk['hour'] >= 6) & (chunk['hour'] <= 18)).astype(int)
            
            # Select features for baseline model
            feature_columns = ['irradiance_w_m2', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_daytime']
            chunk_features = chunk[feature_columns + ['solar_generation_mw', 'timestamp']]
            
            all_features.append(chunk_features)
            
            logger.info(f"  Processed chunk {i+1}/{total_chunks}: {len(chunk):,} records")
        
        # Combine all chunks
        result = pd.concat(all_features, ignore_index=True)
        
        logger.info(f"Created {len(result):,} records with 6 simple features")
        logger.info(f"Feature columns: {['irradiance_w_m2', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_daytime']}")
        
        return result
    
    def _train_baseline_model(self, data: pd.DataFrame):
        """Train the baseline linear model with memory-efficient processing."""
        logger.info("Training baseline linear model (memory-efficient)...")
        
        # Prepare features and target
        feature_columns = ['irradiance_w_m2', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_daytime']
        
        # For large datasets, we'll sample a representative subset for training
        # This avoids memory issues while still using more data than the original 1% sampling
        if len(data) > 5000000:  # If more than 5M records
            logger.info(f"Large dataset detected ({len(data):,} records), sampling 2M records for training...")
            train_sample = data.sample(n=2000000, random_state=42)
        else:
            train_sample = data
        
        X = train_sample[feature_columns].values
        y = train_sample['solar_generation_mw'].values
        
        # Train linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        logger.info(f"Baseline model trained on {len(X):,} samples with {len(feature_columns)} features")
        logger.info(f"Model coefficients: {model.coef_}")
        logger.info(f"Model intercept: {model.intercept_:.2f}")
        
        # For predictions, we'll use a smaller sample to avoid memory issues
        if len(data) > 1000000:  # If more than 1M records
            logger.info(f"Sampling 1M records for predictions...")
            pred_sample = data.sample(n=1000000, random_state=42)
            X_pred = pred_sample[feature_columns].values
            y_pred_actual = pred_sample['solar_generation_mw'].values
        else:
            X_pred = data[feature_columns].values
            y_pred_actual = data['solar_generation_mw'].values
        
        return model, X_pred, y_pred_actual
    
    def _generate_predictions(self, model, X, y):
        """Generate predictions with the baseline model."""
        logger.info("Generating predictions...")
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        logger.info(f"Baseline Model Performance:")
        logger.info(f"  MAE: {mae:.2f} MW")
        logger.info(f"  RMSE: {rmse:.2f} MW")
        logger.info(f"  R²: {r2:.3f}")
        
        return {
            'predictions': y_pred,
            'actual': y,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
        }
    
    def _save_results(self, predictions, model, X, y):
        """Save results and model."""
        logger.info("Saving results...")
        
        # Save predictions
        results_df = pd.DataFrame({
            'actual_mw': predictions['actual'],
            'predicted_mw': predictions['predictions'],
            'error_mw': predictions['actual'] - predictions['predictions']
        })
        
        results_file = self.output_dir / "baseline_predictions.parquet"
        results_df.to_parquet(results_file, index=False)
        
        # Save metrics
        metrics_file = self.output_dir / "baseline_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(predictions['metrics'], f, indent=2)
        
        # Save model info
        model_info = {
            'model_type': 'LinearRegression',
            'n_samples': len(X),
            'n_features': X.shape[1],
            'coefficients': model.coef_.tolist(),
            'intercept': float(model.intercept_),
            'feature_names': ['irradiance_w_m2', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_daytime']
        }
        
        model_file = self.output_dir / "baseline_model_info.json"
        with open(model_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"  - Predictions: {results_file}")
        logger.info(f"  - Metrics: {metrics_file}")
        logger.info(f"  - Model info: {model_file}")
    
    def _print_final_summary(self):
        """Print final summary."""
        total_time = time.time() - self.start_time
        
        logger.info("\n" + "="*80)
        logger.info("🎉 BASELINE PIPELINE WITH EXTENDED DATA COMPLETED")
        logger.info("="*80)
        logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info("="*80)

def main():
    """Main function to run the extended baseline pipeline."""
    pipeline = BaselinePipelineExtended()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
