#!/usr/bin/env python3
"""
Baseline Pipeline with Profiling
================================

This script runs the baseline model pipeline with comprehensive profiling
and logging for the make full-run command.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import time
import cProfile
import pstats
import io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Set up comprehensive logging
def setup_logging():
    """Set up comprehensive logging for the pipeline."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler for detailed log
    file_handler = logging.FileHandler('runtime.log', mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

class BaselinePipelineProfiled:
    """Baseline pipeline with comprehensive profiling and logging."""
    
    def __init__(self):
        """Initialize the profiled baseline pipeline."""
        self.logger = setup_logging()
        self.start_time = time.perf_counter()
        self.step_times = {}
        self.profiler = cProfile.Profile()
        
        # Create output directory
        self.output_dir = Path("pipeline_output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.info("="*80)
        self.logger.info("🚀 BASELINE PIPELINE WITH PROFILING")
        self.logger.info("="*80)
        self.logger.info("Single command execution with comprehensive logging")
        self.logger.info("="*80)
    
    def run_pipeline(self):
        """Run the complete baseline pipeline with profiling."""
        try:
            # Start profiling
            self.profiler.enable()
            
            # Step 1: Load data
            self._log_step_start("Data Loading")
            radiation_data = self._load_radiation_data()
            pv_locations = self._load_pv_locations()
            self._log_step_end("Data Loading")
            
            # Step 2: Create synthetic generation
            self._log_step_start("Synthetic Generation")
            generation_data = self._create_synthetic_generation(radiation_data)
            self._log_step_end("Synthetic Generation")
            
            # Step 3: Merge and preprocess
            self._log_step_start("Data Preprocessing")
            merged_data = self._merge_and_preprocess_data(radiation_data, generation_data)
            self._log_step_end("Data Preprocessing")
            
            # Step 4: Feature engineering
            self._log_step_start("Feature Engineering")
            features_data = self._create_features(merged_data)
            self._log_step_end("Feature Engineering")
            
            # Step 5: Train model
            self._log_step_start("Model Training")
            model, X_train, y_train = self._train_baseline_model(features_data)
            self._log_step_end("Model Training")
            
            # Step 6: Generate predictions
            self._log_step_start("Prediction Generation")
            predictions = self._generate_predictions(model, X_train, y_train)
            self._log_step_end("Prediction Generation")
            
            # Step 7: Save results
            self._log_step_start("Results Saving")
            self._save_results(predictions, model, X_train, y_train)
            self._log_step_end("Results Saving")
            
            # Stop profiling
            self.profiler.disable()
            
            # Generate profiling report
            self._generate_profiling_report()
            
            # Final summary
            self._print_final_summary()
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _log_step_start(self, step_name):
        """Log the start of a pipeline step."""
        self.step_times[step_name] = {'start': time.perf_counter()}
        self.logger.info(f"\n📋 STEP: {step_name}")
        self.logger.info("-" * 60)
    
    def _log_step_end(self, step_name):
        """Log the end of a pipeline step."""
        if step_name in self.step_times:
            end_time = time.perf_counter()
            duration = end_time - self.step_times[step_name]['start']
            self.step_times[step_name]['end'] = end_time
            self.step_times[step_name]['duration'] = duration
            self.logger.info(f"✅ {step_name} completed in {duration:.2f} seconds")
    
    def _load_radiation_data(self):
        """Load radiation data with profiling."""
        self.logger.info("Loading radiation data (sampled for efficiency)...")
        
        base_path = Path("data_3years_2018_2020_final")
        all_data = []
        
        # Load monthly data (2018) with 5% sampling for balance
        monthly_path = base_path / "monthly_15min_results"
        if monthly_path.exists():
            self.logger.info("Loading monthly 15-minute radiation data (5% sampling)...")
            for month_dir in sorted(monthly_path.glob("ssrd_germany_2018_*_15min")):
                try:
                    month_num = int(month_dir.name.split("_")[3])
                except ValueError:
                    continue
                
                self.logger.info(f"Processing month {month_num}...")
                
                parquet_files = list(month_dir.rglob("*.parquet"))
                if parquet_files:
                    df = pd.read_parquet(parquet_files[0])
                    
                    # 5% sampling for balance between speed and data volume
                    sample_size = max(25000, len(df) // 20)  # At least 25k records, or 5%
                    df = df.sample(n=sample_size, random_state=42)
                    
                    df['month'] = month_num
                    df['year'] = 2018
                    all_data.append(df)
                    self.logger.info(f"  Loaded {len(df):,} records for 2018-{month_num:02d} (5% sampling)")
        
        if not all_data:
            raise ValueError("No radiation data found!")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('time').reset_index(drop=True)
        
        # Standardize column names
        if 'ssrd' in combined_df.columns:
            combined_df['irradiance_w_m2'] = combined_df['ssrd'] / 3600
        elif 'ssrd_w_m2' in combined_df.columns:
            combined_df['irradiance_w_m2'] = combined_df['ssrd_w_m2']
        else:
            raise ValueError("No irradiance column found in data")
        
        # Rename time column
        if 'time' in combined_df.columns:
            combined_df = combined_df.rename(columns={'time': 'timestamp'})
        
        result_df = combined_df[['timestamp', 'irradiance_w_m2']].copy()
        
        self.logger.info(f"Combined radiation data: {len(result_df):,} records (5% sampling)")
        self.logger.info(f"Time range: {result_df['timestamp'].min()} to {result_df['timestamp'].max()}")
        self.logger.info(f"Irradiance range: {result_df['irradiance_w_m2'].min():.2f} to {result_df['irradiance_w_m2'].max():.2f} W/m²")
        
        return result_df
    
    def _load_pv_locations(self):
        """Load PV location data."""
        # Use comprehensive lookup table
        comprehensive_path = Path("data_pv_lookup_comprehensive")
        primary_lookup_path = comprehensive_path / "primary_lookup.parquet"
        
        if primary_lookup_path.exists():
            pv_data = pd.read_parquet(primary_lookup_path)
            self.logger.info(f"Loaded {len(pv_data):,} PV locations from comprehensive lookup")
        else:
            raise ValueError("Comprehensive PV lookup table not found!")
        
        self.logger.info(f"PV locations span: lat {pv_data['latitude'].min():.2f} to {pv_data['latitude'].max():.2f}")
        self.logger.info(f"PV locations span: lon {pv_data['longitude'].min():.2f} to {pv_data['longitude'].max():.2f}")
        self.logger.info(f"Total PV capacity: {pv_data['capacity_kw'].sum()/1000:.1f} MW")
        
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
        
        self.logger.info(f"Created synthetic generation data: {len(generation_data):,} records")
        self.logger.info(f"Generation range: {generation_data['solar_generation_mw'].min():.2f} to {generation_data['solar_generation_mw'].max():.2f} MW")
        
        return generation_data
    
    def _merge_and_preprocess_data(self, radiation_df: pd.DataFrame, generation_df: pd.DataFrame) -> pd.DataFrame:
        """Merge radiation and generation data."""
        self.logger.info("Merging radiation and generation datasets...")
        
        # Since both DataFrames are already aligned, simply add the generation column
        merged_data = radiation_df.copy()
        merged_data['solar_generation_mw'] = generation_df['solar_generation_mw']
        
        self.logger.info(f"Merged dataset: {len(merged_data):,} records")
        
        # Filter night-time data
        initial_count = len(merged_data)
        merged_data = merged_data[merged_data['irradiance_w_m2'] >= 10.0]
        final_count = len(merged_data)
        self.logger.info(f"Filtered night-time data: {initial_count - final_count} records removed")
        
        return merged_data
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for baseline model."""
        self.logger.info("Creating features for baseline model...")
        
        # Basic time features
        data['hour'] = data['timestamp'].dt.hour
        data['month'] = data['timestamp'].dt.month
        data['day_of_year'] = data['timestamp'].dt.dayofyear
        
        # Cyclical encoding for time features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Daytime indicator
        data['is_daytime'] = ((data['hour'] >= 6) & (data['hour'] <= 18)).astype(int)
        
        # Select features for baseline model
        feature_columns = ['irradiance_w_m2', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_daytime']
        
        self.logger.info(f"Created {len(feature_columns)} features")
        self.logger.info(f"Feature columns: {feature_columns}")
        
        return data[feature_columns + ['solar_generation_mw', 'timestamp']]
    
    def _train_baseline_model(self, data: pd.DataFrame):
        """Train the baseline linear model."""
        self.logger.info("Training baseline linear model...")
        
        # Prepare features and target
        feature_columns = ['irradiance_w_m2', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_daytime']
        
        # Use all available data for training
        X = data[feature_columns].values
        y = data['solar_generation_mw'].values
        
        # Train linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        self.logger.info(f"Baseline model trained on {len(X):,} samples with {len(feature_columns)} features")
        self.logger.info(f"Model coefficients: {model.coef_}")
        self.logger.info(f"Model intercept: {model.intercept_:.2f}")
        
        return model, X, y
    
    def _generate_predictions(self, model, X, y):
        """Generate predictions with the baseline model."""
        self.logger.info("Generating predictions...")
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        self.logger.info(f"Baseline Model Performance:")
        self.logger.info(f"  MAE: {mae:.2f} MW")
        self.logger.info(f"  RMSE: {rmse:.2f} MW")
        self.logger.info(f"  R²: {r2:.3f}")
        
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
        self.logger.info("Saving results...")
        
        # Save predictions to main output file
        results_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2018-01-01', periods=len(predictions['actual']), freq='15min'),
            'actual_mw': predictions['actual'],
            'predicted_mw': predictions['predictions'],
            'error_mw': predictions['actual'] - predictions['predictions']
        })
        
        # Save main predictions file
        results_df.to_parquet('predictions.parquet', index=False)
        
        # Save detailed results to pipeline_output
        detailed_results_file = self.output_dir / "baseline_predictions.parquet"
        results_df.to_parquet(detailed_results_file, index=False)
        
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
        
        self.logger.info(f"Results saved:")
        self.logger.info(f"  - Main output: predictions.parquet")
        self.logger.info(f"  - Detailed results: {detailed_results_file}")
        self.logger.info(f"  - Metrics: {metrics_file}")
        self.logger.info(f"  - Model info: {model_file}")
    
    def _generate_profiling_report(self):
        """Generate profiling report."""
        self.logger.info("Generating profiling report...")
        
        # Create profiling stats
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        # Save profiling report
        profiling_file = self.output_dir / "profiling_report.txt"
        with open(profiling_file, 'w') as f:
            f.write("Baseline Pipeline Profiling Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(s.getvalue())
        
        # Create profiling summary
        profiling_summary = {
            'total_execution_time': time.perf_counter() - self.start_time,
            'step_times': {step: data.get('duration', 0) for step, data in self.step_times.items()},
            'profiling_file': str(profiling_file),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(profiling_summary, f, indent=2)
        
        self.logger.info(f"Profiling report saved: {profiling_file}")
        self.logger.info(f"Pipeline summary saved: {summary_file}")
    
    def _print_final_summary(self):
        """Print final summary."""
        total_time = time.perf_counter() - self.start_time
        
        self.logger.info("\n" + "="*80)
        self.logger.info("🎉 BASELINE PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("="*80)
        self.logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
        self.logger.info(f"📄 Main output: predictions.parquet")
        self.logger.info(f"📋 Runtime log: runtime.log")
        self.logger.info("="*80)
        
        # Print step timing summary
        self.logger.info("\n📊 Step Timing Summary:")
        for step, data in self.step_times.items():
            duration = data.get('duration', 0)
            percentage = (duration / total_time) * 100
            self.logger.info(f"  {step}: {duration:.2f}s ({percentage:.1f}%)")

def main():
    """Main function to run the profiled baseline pipeline."""
    pipeline = BaselinePipelineProfiled()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
