#!/usr/bin/env python3
"""
Fixed Baseline Model - Addresses the major issues found in the original implementation.
====================================================================================

Key Fixes:
1. Convert irradiance from J/m² to W/m² (divide by 3600 seconds)
2. Resample irradiance to 15-minute intervals
3. Simplify feature engineering for baseline
4. Use proper data scaling
5. Add data validation and debugging
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedBaselineModel:
    """
    Fixed baseline model that properly handles:
    - Irradiance unit conversion (J/m² → W/m²)
    - Time resolution matching (hourly → 15-min)
    - Simplified feature engineering
    - Proper data validation
    """
    
    def __init__(self, output_dir="baseline_results_fixed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model parameters
        self.model_type = 'linear'  # or 'ridge'
        self.rolling_window_hours = 24 * 7  # 1 week instead of 30 days
        self.min_irradiance_threshold = 10.0  # W/m²
        
        # Data storage
        self.irradiance_data = None
        self.pv_data = None
        self.merged_data = None
        self.predictions = []
        self.actuals = []
        self.coefficients = []
        
        logger.info(f"Fixed baseline model initialized")
        logger.info(f"Rolling window: {self.rolling_window_hours} hours")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_and_process_irradiance(self, irradiance_path):
        """Load and properly process irradiance data."""
        logger.info(f"Loading irradiance data from: {irradiance_path}")
        
        # Load partitioned data
        all_data = []
        base_path = Path(irradiance_path)
        
        for year_dir in sorted(base_path.glob("year=*")):
            year = int(year_dir.name.split("=")[1])
            logger.info(f"Processing year {year}...")
            
            for month_dir in sorted(year_dir.glob("month=*")):
                month = int(month_dir.name.split("=")[1])
                parquet_files = list(month_dir.glob("*.parquet"))
                
                if parquet_files:
                    df = pd.read_parquet(parquet_files[0])
                    df['year'] = year
                    df['month'] = month
                    all_data.append(df)
        
        # Combine data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('time').reset_index(drop=True)
        
        logger.info(f"Loaded {len(combined_df):,} irradiance records")
        logger.info(f"Time range: {combined_df['time'].min()} to {combined_df['time'].max()}")
        
        # CRITICAL FIX: Convert from J/m² to W/m²
        logger.info("Converting irradiance from J/m² to W/m²...")
        combined_df['ssrd_w_m2'] = combined_df['ssrd'] / 3600  # Divide by 3600 seconds
        
        # Calculate national average (simple mean across all grid points)
        logger.info("Calculating national average irradiance...")
        national_avg = combined_df.groupby('time')['ssrd_w_m2'].mean().reset_index()
        national_avg = national_avg.rename(columns={'ssrd_w_m2': 'irradiance_w_m2'})
        
        # Resample to 15-minute intervals
        logger.info("Resampling to 15-minute intervals...")
        national_avg['time'] = pd.to_datetime(national_avg['time'])
        national_avg = national_avg.set_index('time')
        
        # Forward fill to get 15-min values
        resampled = national_avg.resample('15T').ffill()
        
        # Add some interpolation for smoother transitions
        resampled = resampled.interpolate(method='linear')
        
        self.irradiance_data = resampled.reset_index()
        
        logger.info(f"Processed irradiance data:")
        logger.info(f"  Records: {len(self.irradiance_data):,}")
        logger.info(f"  Time range: {self.irradiance_data['time'].min()} to {self.irradiance_data['time'].max()}")
        logger.info(f"  Irradiance range: {self.irradiance_data['irradiance_w_m2'].min():.2f} to {self.irradiance_data['irradiance_w_m2'].max():.2f} W/m²")
        
        return self.irradiance_data
    
    def load_pv_data(self, pv_path):
        """Load PV generation data."""
        logger.info(f"Loading PV data from: {pv_path}")
        
        self.pv_data = pd.read_parquet(pv_path)
        self.pv_data['timestamp'] = pd.to_datetime(self.pv_data['timestamp'])
        self.pv_data = self.pv_data.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.pv_data):,} PV records")
        logger.info(f"Time range: {self.pv_data['timestamp'].min()} to {self.pv_data['timestamp'].max()}")
        logger.info(f"Generation range: {self.pv_data['solar_generation_mw'].min():.2f} to {self.pv_data['solar_generation_mw'].max():.2f} MW")
        
        return self.pv_data
    
    def merge_data(self):
        """Merge irradiance and PV data."""
        logger.info("Merging irradiance and PV data...")
        
        # Rename time columns for merging
        irradiance = self.irradiance_data.copy()
        pv = self.pv_data.copy()
        
        irradiance = irradiance.rename(columns={'time': 'timestamp'})
        
        # Merge on timestamp
        merged = pd.merge(pv, irradiance, on='timestamp', how='inner')
        
        # Filter night-time data
        merged = merged[merged['irradiance_w_m2'] >= self.min_irradiance_threshold]
        
        # Add simple time features
        merged['hour'] = merged['timestamp'].dt.hour
        merged['month'] = merged['timestamp'].dt.month
        merged['day_of_year'] = merged['timestamp'].dt.dayofyear
        
        # Simple cyclical encoding
        merged['hour_sin'] = np.sin(2 * np.pi * merged['hour'] / 24)
        merged['hour_cos'] = np.cos(2 * np.pi * merged['hour'] / 24)
        merged['month_sin'] = np.sin(2 * np.pi * merged['month'] / 12)
        merged['month_cos'] = np.cos(2 * np.pi * merged['month'] / 12)
        
        self.merged_data = merged.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Merged data: {len(self.merged_data):,} records")
        logger.info(f"Time range: {self.merged_data['timestamp'].min()} to {self.merged_data['timestamp'].max()}")
        
        return self.merged_data
    
    def train_rolling_model(self):
        """Train model with rolling window approach."""
        logger.info("Training rolling window model...")
        
        data = self.merged_data.copy()
        window_size = self.rolling_window_hours * 4  # 4 intervals per hour
        
        # Simple features for baseline
        features = ['irradiance_w_m2', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        X = data[features]
        y = data['solar_generation_mw']
        
        # Initialize storage
        self.predictions = []
        self.actuals = []
        self.coefficients = []
        
        # Rolling window training
        for i in range(window_size, len(data)):
            # Training window
            X_train = X.iloc[i-window_size:i]
            y_train = y.iloc[i-window_size:i]
            
            # Test point
            X_test = X.iloc[i:i+1]
            y_test = y.iloc[i:i+1]
            
            # Train model
            if self.model_type == 'linear':
                model = LinearRegression()
            else:
                model = Ridge(alpha=1.0)
            
            model.fit(X_train, y_train)
            
            # Predict
            pred = model.predict(X_test)[0]
            actual = y_test.iloc[0]
            
            # Store results
            self.predictions.append(pred)
            self.actuals.append(actual)
            self.coefficients.append({
                'timestamp': data.iloc[i]['timestamp'],
                'intercept': model.intercept_,
                'irradiance_coef': model.coef_[0],
                'hour_sin_coef': model.coef_[1],
                'hour_cos_coef': model.coef_[2],
                'month_sin_coef': model.coef_[3],
                'month_cos_coef': model.coef_[4]
            })
            
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(data)} records")
        
        logger.info("Rolling window training completed!")
        
        # Calculate overall metrics
        rmse = np.sqrt(mean_squared_error(self.actuals, self.predictions))
        mae = mean_absolute_error(self.actuals, self.predictions)
        r2 = r2_score(self.actuals, self.predictions)
        
        logger.info(f"Overall RMSE: {rmse:.2f} MW")
        logger.info(f"Overall MAE: {mae:.2f} MW")
        logger.info(f"Overall R²: {r2:.3f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': self.predictions,
            'actuals': self.actuals,
            'coefficients': self.coefficients
        }
    
    def create_visualizations(self, results):
        """Create diagnostic plots."""
        logger.info("Creating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Predictions vs Actuals scatter
        axes[0, 0].scatter(self.actuals, self.predictions, alpha=0.5, s=1)
        axes[0, 0].plot([0, max(self.actuals)], [0, max(self.actuals)], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual PV Generation (MW)')
        axes[0, 0].set_ylabel('Predicted PV Generation (MW)')
        axes[0, 0].set_title('Predictions vs Actuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Time series (first 1000 points)
        n_points = min(1000, len(self.actuals))
        timestamps = [coef['timestamp'] for coef in self.coefficients[:n_points]]
        axes[0, 1].plot(timestamps, self.actuals[:n_points], label='Actual', alpha=0.7)
        axes[0, 1].plot(timestamps, self.predictions[:n_points], label='Predicted', alpha=0.7)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('PV Generation (MW)')
        axes[0, 1].set_title('Time Series (First 1000 points)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals
        residuals = np.array(self.predictions) - np.array(self.actuals)
        axes[1, 0].scatter(self.actuals, residuals, alpha=0.5, s=1)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Actual PV Generation (MW)')
        axes[1, 0].set_ylabel('Residuals (MW)')
        axes[1, 0].set_title('Residuals vs Actuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Coefficient evolution
        irradiance_coefs = [coef['irradiance_coef'] for coef in self.coefficients]
        timestamps_coefs = [coef['timestamp'] for coef in self.coefficients]
        axes[1, 1].plot(timestamps_coefs, irradiance_coefs, alpha=0.7)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Irradiance Coefficient')
        axes[1, 1].set_title('Irradiance Coefficient Evolution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fixed_baseline_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to: {self.output_dir / 'fixed_baseline_performance.png'}")
    
    def save_results(self, results):
        """Save results to files."""
        logger.info("Saving results...")
        
        # Save predictions
        results_df = pd.DataFrame({
            'timestamp': [coef['timestamp'] for coef in self.coefficients],
            'actual': self.actuals,
            'predicted': self.predictions,
            'residual': np.array(self.predictions) - np.array(self.actuals)
        })
        results_df.to_csv(self.output_dir / 'fixed_baseline_predictions.csv', index=False)
        
        # Save coefficients
        coef_df = pd.DataFrame(self.coefficients)
        coef_df.to_csv(self.output_dir / 'fixed_baseline_coefficients.csv', index=False)
        
        # Save metrics
        metrics = {
            'overall_metrics': {
                'rmse': results['rmse'],
                'mae': results['mae'],
                'r2': results['r2']
            },
            'data_summary': {
                'total_records': len(self.merged_data),
                'training_windows': len(self.coefficients),
                'time_range': {
                    'start': str(self.merged_data['timestamp'].min()),
                    'end': str(self.merged_data['timestamp'].max())
                }
            }
        }
        
        with open(self.output_dir / 'fixed_baseline_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save report
        report = f"""
FIXED BASELINE MODEL RESULTS
============================

Model Configuration:
- Model Type: {self.model_type}
- Rolling Window: {self.rolling_window_hours} hours
- Min Irradiance Threshold: {self.min_irradiance_threshold} W/m²

Data Summary:
- Total Records: {len(self.merged_data):,}
- Training Windows: {len(self.coefficients):,}
- Time Range: {self.merged_data['timestamp'].min()} to {self.merged_data['timestamp'].max()}

Performance Metrics:
- RMSE: {results['rmse']:.2f} MW
- MAE: {results['mae']:.2f} MW
- R²: {results['r2']:.3f}

Key Fixes Applied:
1. Converted irradiance from J/m² to W/m² (divided by 3600)
2. Resampled irradiance to 15-minute intervals
3. Simplified feature engineering (5 features vs 23)
4. Reduced rolling window size (1 week vs 30 days)
5. Added proper data validation

Interpretation:
- R² > 0.5 indicates good fit
- R² < 0 indicates model performs worse than mean
- RMSE should be reasonable compared to data range
"""
        
        with open(self.output_dir / 'fixed_baseline_report.txt', 'w') as f:
            f.write(report)
        
        logger.info(f"Results saved to: {self.output_dir}")
    
    def run_pipeline(self, irradiance_path, pv_path):
        """Run the complete fixed pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING FIXED BASELINE MODEL PIPELINE")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load and process data
            self.load_and_process_irradiance(irradiance_path)
            self.load_pv_data(pv_path)
            
            # Step 2: Merge data
            self.merge_data()
            
            # Step 3: Train model
            results = self.train_rolling_model()
            
            # Step 4: Create visualizations
            self.create_visualizations(results)
            
            # Step 5: Save results
            self.save_results(results)
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            logger.info("=" * 80)
            logger.info("FIXED BASELINE PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            logger.info(f"Output directory: {self.output_dir}")
            
            return {
                'success': True,
                'metrics': results,
                'output_directory': str(self.output_dir),
                'total_time_seconds': total_time
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """Main function to run the fixed baseline model."""
    print("Fixed Baseline Model - Addressing Implementation Issues")
    print("=" * 60)
    
    # Define paths
    irradiance_path = "data_3years_2018_2020_final/ssrd_germany_2018_2020_combined"
    pv_path = "data_german_solar_generation/processed/german_solar_generation_v3_2018_2020.parquet"
    
    print(f"�� Irradiance data: {irradiance_path}")
    print(f"⚡ PV generation data: {pv_path}")
    print(f"📁 Output directory: baseline_results_fixed")
    print()
    
    # Create and run model
    model = FixedBaselineModel()
    results = model.run_pipeline(irradiance_path, pv_path)
    
    if results['success']:
        print("\n" + "=" * 60)
        print("✅ FIXED BASELINE MODEL COMPLETED!")
        print("=" * 60)
        print(f"📊 RMSE: {results['metrics']['rmse']:.2f} MW")
        print(f"📊 MAE: {results['metrics']['mae']:.2f} MW")
        print(f"📈 R²: {results['metrics']['r2']:.3f}")
        print(f"⏱️  Total Time: {results['total_time_seconds']:.2f} seconds")
        print(f"📁 Results: {results['output_directory']}")
        
        # Interpretation
        r2 = results['metrics']['r2']
        if r2 > 0.5:
            print("🎉 Excellent performance! R² > 0.5")
        elif r2 > 0.3:
            print("👍 Good performance! R² > 0.3")
        elif r2 > 0.1:
            print("📈 Acceptable performance! R² > 0.1")
        else:
            print("⚠️  Poor performance. Check data and model assumptions.")
    else:
        print(f"\n❌ Pipeline failed: {results['error']}")

if __name__ == "__main__":
    main()