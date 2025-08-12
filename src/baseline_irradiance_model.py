#!/usr/bin/env python3
"""
Baseline Irradiance-Only Model for German Solar PV Production
=============================================================

This module implements the baseline approach for predicting quarter-hourly 
national solar PV production using only irradiance data.

Baseline Model: Pt = α + βIt (Linear) or Ridge Regression
- Pt: National PV generation (MW)
- It: National average irradiance (W/m²)
- α, β: Model parameters

Features:
- Rolling window training for temporal adaptation
- Night-time handling (It ≈ 0)
- Missing value handling
- Calendar effects (seasonal, daily patterns)
- Robust evaluation metrics
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineIrradianceModel:
    """
    Baseline model for predicting German solar PV production using irradiance only.
    
    Implements the equation: Pt = α + βIt
    where:
    - Pt: National PV generation (MW)
    - It: National average irradiance (W/m²)
    - α, β: Model parameters
    """
    
    def __init__(self, 
                 irradiance_data_path: Optional[Path] = None,
                 pv_actuals_path: Optional[Path] = None,
                 output_dir: Optional[Path] = None,
                 model_type: str = 'linear',
                 rolling_window_days: int = 30,
                 min_irradiance_threshold: float = 10.0):
        """
        Initialize the baseline irradiance model.
        
        Parameters:
        -----------
        irradiance_data_path : Path, optional
            Path to irradiance data (15-minute intervals)
        pv_actuals_path : Path, optional
            Path to PV actuals data (15-minute intervals)
        output_dir : Path, optional
            Output directory for results
        model_type : str
            Model type: 'linear' or 'ridge'
        rolling_window_days : int
            Number of days for rolling window training
        min_irradiance_threshold : float
            Minimum irradiance threshold for daytime detection (W/m²)
        """
        self.irradiance_data_path = irradiance_data_path
        self.pv_actuals_path = pv_actuals_path
        self.output_dir = output_dir or Path("baseline_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.model_type = model_type.lower()
        self.rolling_window_days = rolling_window_days
        self.min_irradiance_threshold = min_irradiance_threshold
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Data storage
        self.irradiance_data = None
        self.pv_actuals_data = None
        self.combined_data = None
        
        # Results storage
        self.predictions = None
        self.metrics = {}
        self.model_coefficients = {}
        
        logger.info(f"Baseline model initialized: {model_type} regression")
        logger.info(f"Rolling window: {rolling_window_days} days")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_irradiance_data(self, data_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load irradiance data from the processed pipeline output.
        
        Parameters:
        -----------
        data_path : Path, optional
            Path to irradiance data. If None, uses default path.
            
        Returns:
        --------
        pd.DataFrame
            Irradiance data with columns: ['time', 'ssrd', 'latitude', 'longitude']
        """
        path = data_path or self.irradiance_data_path
        if path is None:
            # Try to find the data automatically
            possible_paths = [
                Path("data_3years_2018_2020_final/ssrd_germany_2018_2020_combined_15min"),
                Path("data_enhanced_tiny/ssrd_germany_2018_2018_15min"),
                Path("data_optimized_tiny/ssrd_germany_2018_2018_15min")
            ]
            
            for p in possible_paths:
                if p.exists():
                    path = p
                    break
            else:
                raise FileNotFoundError("No irradiance data found. Please specify data_path.")
        
        logger.info(f"Loading irradiance data from: {path}")
        
        try:
            # Load from Parquet format (partitioned)
            if path.is_dir():
                df = pd.read_parquet(path)
            else:
                df = pd.read_parquet(path)
            
            # Ensure required columns exist
            required_cols = ['time', 'ssrd']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert time to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'])
            
            # Sort by time
            df = df.sort_values('time').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df):,} irradiance records")
            logger.info(f"Time range: {df['time'].min()} to {df['time'].max()}")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading irradiance data: {e}")
            raise
    
    def load_pv_actuals_data(self, data_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load PV actuals data from ENTSO-E or generated data.
        
        Parameters:
        -----------
        data_path : Path, optional
            Path to PV actuals data. If None, uses default path.
            
        Returns:
        --------
        pd.DataFrame
            PV actuals data with columns: ['timestamp', 'solar_generation_mw']
        """
        path = data_path or self.pv_actuals_path
        if path is None:
            # Try to find the data automatically
            possible_paths = [
                Path("data_german_solar_generation/processed"),
                Path("data_german_solar_generation/raw")
            ]
            
            for p in possible_paths:
                if p.exists():
                    # Find the most recent file
                    files = list(p.glob("*.parquet")) + list(p.glob("*.csv"))
                    if files:
                        path = max(files, key=lambda x: x.stat().st_mtime)
                        break
            else:
                logger.warning("No PV actuals data found. Will generate synthetic data.")
                return self._generate_synthetic_pv_data()
        
        logger.info(f"Loading PV actuals data from: {path}")
        
        try:
            # Load data
            if path.suffix == '.parquet':
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'solar_generation_mw']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df):,} PV actuals records")
            logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading PV actuals data: {e}")
            logger.info("Generating synthetic PV data as fallback")
            return self._generate_synthetic_pv_data()
    
    def _generate_synthetic_pv_data(self) -> pd.DataFrame:
        """
        Generate synthetic PV actuals data for testing when real data is unavailable.
        
        Returns:
        --------
        pd.DataFrame
            Synthetic PV actuals data
        """
        logger.info("Generating synthetic PV actuals data...")
        
        # Generate 3 years of 15-minute data (2018-2020)
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2021, 1, 1)
        
        timestamps = []
        current = start_date
        while current < end_date:
            timestamps.append(current)
            current += timedelta(minutes=15)
        
        data = []
        for timestamp in timestamps:
            hour = timestamp.hour
            month = timestamp.month
            
            # Realistic solar generation pattern
            if 6 <= hour <= 18:  # Daylight hours
                # Seasonal variation (higher in summer)
                seasonal_factor = 0.3 + 0.7 * np.sin((month - 1) * np.pi / 6)
                
                # Daily pattern (peak at noon)
                daily_factor = np.sin((hour - 6) * np.pi / 12)
                
                # Add some randomness
                noise = np.random.normal(0, 0.1)
                
                # Base capacity for Germany (approximate)
                base_capacity = 50000  # MW
                
                generation = base_capacity * seasonal_factor * daily_factor * (1 + noise)
                generation = max(0, generation)
            else:
                generation = 0
            
            data.append({
                'timestamp': timestamp,
                'solar_generation_mw': round(generation, 2)
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df):,} synthetic PV actuals records")
        return df
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess and combine irradiance and PV actuals data.
        
        Returns:
        --------
        pd.DataFrame
            Combined and preprocessed data
        """
        logger.info("Preprocessing data...")
        
        # Load data if not already loaded
        if self.irradiance_data is None:
            self.irradiance_data = self.load_irradiance_data()
        
        if self.pv_actuals_data is None:
            self.pv_actuals_data = self.load_pv_actuals_data()
        
        # Calculate national average irradiance
        logger.info("Calculating national average irradiance...")
        irradiance_avg = self.irradiance_data.groupby('time')['ssrd'].mean().reset_index()
        irradiance_avg = irradiance_avg.rename(columns={'time': 'timestamp'})
        
        # Merge irradiance and PV actuals
        logger.info("Merging irradiance and PV actuals data...")
        combined = pd.merge(
            irradiance_avg, 
            self.pv_actuals_data[['timestamp', 'solar_generation_mw']], 
            on='timestamp', 
            how='inner'
        )
        
        # Sort by timestamp
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        # Add time-based features
        logger.info("Adding time-based features...")
        combined['hour'] = combined['timestamp'].dt.hour
        combined['month'] = combined['timestamp'].dt.month
        combined['day_of_year'] = combined['timestamp'].dt.dayofyear
        combined['is_daytime'] = (combined['hour'] >= 6) & (combined['hour'] <= 18)
        combined['is_nighttime'] = ~combined['is_daytime']
        
        # Handle missing values
        logger.info("Handling missing values...")
        initial_count = len(combined)
        combined = combined.dropna()
        final_count = len(combined)
        
        if initial_count != final_count:
            logger.warning(f"Removed {initial_count - final_count} records with missing values")
        
        # Filter out very low irradiance (night-time)
        combined = combined[combined['ssrd'] >= self.min_irradiance_threshold]
        
        logger.info(f"Preprocessed data: {len(combined):,} records")
        logger.info(f"Time range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
        
        self.combined_data = combined
        return combined
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for the baseline model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data with timestamp, ssrd, solar_generation_mw
            
        Returns:
        --------
        pd.DataFrame
            Data with additional features
        """
        logger.info("Creating features...")
        
        # Copy to avoid modifying original
        df_features = df.copy()
        
        # Basic irradiance features
        df_features['irradiance_w_m2'] = df_features['ssrd']
        df_features['irradiance_kw_m2'] = df_features['ssrd'] / 1000
        
        # Time-based features
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
        df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
        
        # Binary features
        df_features['is_weekend'] = df_features['timestamp'].dt.weekday >= 5
        df_features['is_peak_hours'] = (df_features['hour'] >= 10) & (df_features['hour'] <= 16)
        
        # Lag features (previous values)
        df_features['irradiance_lag_1'] = df_features['irradiance_w_m2'].shift(1)
        df_features['irradiance_lag_4'] = df_features['irradiance_w_m2'].shift(4)  # 1 hour ago
        df_features['irradiance_lag_96'] = df_features['irradiance_w_m2'].shift(96)  # 1 day ago
        
        # Rolling statistics
        df_features['irradiance_rolling_mean_4'] = df_features['irradiance_w_m2'].rolling(4).mean()
        df_features['irradiance_rolling_std_4'] = df_features['irradiance_w_m2'].rolling(4).std()
        
        # Remove NaN values from lag features
        df_features = df_features.dropna()
        
        logger.info(f"Created features. Final shape: {df_features.shape}")
        return df_features
    
    def train_rolling_window_model(self, df: pd.DataFrame) -> Dict:
        """
        Train the baseline model using rolling window approach.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Preprocessed data with features
            
        Returns:
        --------
        Dict
            Training results and model information
        """
        logger.info(f"Training {self.model_type} model with rolling window...")
        
        # Prepare features and target
        feature_cols = ['irradiance_w_m2']  # Start with just irradiance
        target_col = 'solar_generation_mw'
        
        # Initialize results storage
        all_predictions = []
        all_actuals = []
        model_coefficients = []
        training_metrics = []
        
        # Rolling window parameters
        window_size = self.rolling_window_days * 96  # 96 15-min intervals per day
        step_size = 24 * 96  # Move forward by 1 day
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Initialize model
        if self.model_type == 'linear':
            model = LinearRegression()
        elif self.model_type == 'ridge':
            model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Rolling window training
        start_idx = window_size
        end_idx = len(df) - 1
        
        logger.info(f"Rolling window: {window_size} intervals ({self.rolling_window_days} days)")
        logger.info(f"Training from index {start_idx} to {end_idx}")
        
        for i in range(start_idx, end_idx, step_size):
            # Training window
            train_start = max(0, i - window_size)
            train_end = i
            
            # Prediction window
            pred_start = i
            pred_end = min(len(df), i + step_size)
            
            # Extract training data
            train_data = df.iloc[train_start:train_end]
            pred_data = df.iloc[pred_start:pred_end]
            
            # Skip if not enough data
            if len(train_data) < window_size * 0.8:  # At least 80% of window
                continue
            
            # Prepare features and target
            X_train = train_data[feature_cols].values
            y_train = train_data[target_col].values
            X_pred = pred_data[feature_cols].values
            y_actual = pred_data[target_col].values
            
            # Skip if no valid data
            if len(X_train) == 0 or len(X_pred) == 0:
                continue
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_pred)
            
            # Store results
            all_predictions.extend(y_pred)
            all_actuals.extend(y_actual)
            
            # Store model coefficients
            if hasattr(model, 'coef_'):
                model_coefficients.append({
                    'timestamp': pred_data['timestamp'].iloc[0],
                    'intercept': model.intercept_,
                    'coefficient': model.coef_[0] if len(model.coef_) > 0 else 0
                })
            
            # Calculate training metrics
            train_pred = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            train_r2 = r2_score(y_train, train_pred)
            
            training_metrics.append({
                'timestamp': pred_data['timestamp'].iloc[0],
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'train_samples': len(y_train)
            })
            
            # Progress update
            if i % (step_size * 7) == 0:  # Weekly progress
                logger.info(f"Processed up to {pred_data['timestamp'].iloc[-1]}")
        
        # Convert to arrays
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        # Calculate overall metrics
        overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        overall_mae = mean_absolute_error(all_actuals, all_predictions)
        overall_r2 = r2_score(all_actuals, all_predictions)
        
        # Store results
        self.predictions = pd.DataFrame({
            'timestamp': df['timestamp'].iloc[start_idx:end_idx:step_size].repeat(step_size)[:len(all_predictions)],
            'actual': all_actuals,
            'predicted': all_predictions
        })
        
        self.metrics = {
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'overall_r2': overall_r2,
            'total_predictions': len(all_predictions)
        }
        
        self.model_coefficients = pd.DataFrame(model_coefficients)
        
        logger.info("Rolling window training completed!")
        logger.info(f"Overall RMSE: {overall_rmse:.2f} MW")
        logger.info(f"Overall MAE: {overall_mae:.2f} MW")
        logger.info(f"Overall R²: {overall_r2:.3f}")
        
        return {
            'metrics': self.metrics,
            'predictions': self.predictions,
            'model_coefficients': self.model_coefficients,
            'training_metrics': training_metrics
        }
    
    def evaluate_model(self) -> Dict:
        """
        Evaluate the trained model with comprehensive metrics.
        
        Returns:
        --------
        Dict
            Evaluation results
        """
        if self.predictions is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model performance...")
        
        actual = self.predictions['actual'].values
        predicted = self.predictions['predicted'].values
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        # Additional metrics
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100  # Avoid division by zero
        bias = np.mean(predicted - actual)
        
        # Time-based analysis
        self.predictions['hour'] = self.predictions['timestamp'].dt.hour
        self.predictions['month'] = self.predictions['timestamp'].dt.month
        
        # Hourly performance
        hourly_metrics = {}
        for hour in range(24):
            hour_mask = self.predictions['hour'] == hour
            if hour_mask.sum() > 0:
                hour_actual = actual[hour_mask]
                hour_pred = predicted[hour_mask]
                hourly_metrics[hour] = {
                    'rmse': np.sqrt(mean_squared_error(hour_actual, hour_pred)),
                    'mae': mean_absolute_error(hour_actual, hour_pred),
                    'r2': r2_score(hour_actual, hour_pred),
                    'samples': len(hour_actual)
                }
        
        # Monthly performance
        monthly_metrics = {}
        for month in range(1, 13):
            month_mask = self.predictions['month'] == month
            if month_mask.sum() > 0:
                month_actual = actual[month_mask]
                month_pred = predicted[month_mask]
                monthly_metrics[month] = {
                    'rmse': np.sqrt(mean_squared_error(month_actual, month_pred)),
                    'mae': mean_absolute_error(month_actual, month_pred),
                    'r2': r2_score(month_actual, month_pred),
                    'samples': len(month_actual)
                }
        
        evaluation_results = {
            'overall_metrics': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'bias': bias
            },
            'hourly_metrics': hourly_metrics,
            'monthly_metrics': monthly_metrics,
            'total_samples': len(actual)
        }
        
        logger.info("Evaluation completed!")
        logger.info(f"Overall RMSE: {rmse:.2f} MW")
        logger.info(f"Overall MAE: {mae:.2f} MW")
        logger.info(f"Overall R²: {r2:.3f}")
        logger.info(f"MAPE: {mape:.2f}%")
        logger.info(f"Bias: {bias:.2f} MW")
        
        return evaluation_results
    
    def create_visualizations(self, evaluation_results: Dict) -> None:
        """
        Create comprehensive visualizations of model performance.
        
        Parameters:
        -----------
        evaluation_results : Dict
            Results from model evaluation
        """
        logger.info("Creating visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Baseline {self.model_type.title()} Model Performance', fontsize=16)
        
        # 1. Actual vs Predicted scatter plot
        axes[0, 0].scatter(self.predictions['actual'], self.predictions['predicted'], alpha=0.5, s=1)
        axes[0, 0].plot([0, self.predictions['actual'].max()], [0, self.predictions['actual'].max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Generation (MW)')
        axes[0, 0].set_ylabel('Predicted Generation (MW)')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Time series plot (sample)
        sample_size = min(1000, len(self.predictions))
        sample_data = self.predictions.sample(n=sample_size).sort_values('timestamp')
        
        axes[0, 1].plot(sample_data['timestamp'], sample_data['actual'], label='Actual', alpha=0.7)
        axes[0, 1].plot(sample_data['timestamp'], sample_data['predicted'], label='Predicted', alpha=0.7)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Generation (MW)')
        axes[0, 1].set_title('Time Series (Sample)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals plot
        residuals = self.predictions['predicted'] - self.predictions['actual']
        axes[0, 2].scatter(self.predictions['actual'], residuals, alpha=0.5, s=1)
        axes[0, 2].axhline(y=0, color='r', linestyle='--')
        axes[0, 2].set_xlabel('Actual Generation (MW)')
        axes[0, 2].set_ylabel('Residuals (MW)')
        axes[0, 2].set_title('Residuals vs Actual')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Hourly performance
        hours = list(evaluation_results['hourly_metrics'].keys())
        hourly_rmse = [evaluation_results['hourly_metrics'][h]['rmse'] for h in hours]
        
        axes[1, 0].bar(hours, hourly_rmse)
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('RMSE (MW)')
        axes[1, 0].set_title('RMSE by Hour of Day')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Monthly performance
        months = list(evaluation_results['monthly_metrics'].keys())
        monthly_rmse = [evaluation_results['monthly_metrics'][m]['rmse'] for m in months]
        
        axes[1, 1].bar(months, monthly_rmse)
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('RMSE (MW)')
        axes[1, 1].set_title('RMSE by Month')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Model coefficients over time
        if not self.model_coefficients.empty:
            axes[1, 2].plot(self.model_coefficients['timestamp'], self.model_coefficients['coefficient'])
            axes[1, 2].set_xlabel('Time')
            axes[1, 2].set_ylabel('Coefficient β')
            axes[1, 2].set_title('Model Coefficient β Over Time')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'baseline_{self.model_type}_performance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to: {plot_path}")
    
    def save_results(self, evaluation_results: Dict) -> None:
        """
        Save all results to files.
        
        Parameters:
        -----------
        evaluation_results : Dict
            Evaluation results
        """
        logger.info("Saving results...")
        
        # Save predictions
        predictions_path = self.output_dir / f'baseline_{self.model_type}_predictions.csv'
        self.predictions.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to: {predictions_path}")
        
        # Save model coefficients
        if not self.model_coefficients.empty:
            coeff_path = self.output_dir / f'baseline_{self.model_type}_coefficients.csv'
            self.model_coefficients.to_csv(coeff_path, index=False)
            logger.info(f"Model coefficients saved to: {coeff_path}")
        
        # Save evaluation metrics
        metrics_path = self.output_dir / f'baseline_{self.model_type}_metrics.json'
        import json
        with open(metrics_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        logger.info(f"Metrics saved to: {metrics_path}")
        
        # Save summary report
        report_path = self.output_dir / f'baseline_{self.model_type}_report.txt'
        with open(report_path, 'w') as f:
            f.write("BASELINE IRRADIANCE MODEL REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {self.model_type.title()} Regression\n")
            f.write(f"Rolling Window: {self.rolling_window_days} days\n")
            f.write(f"Training Period: {self.combined_data['timestamp'].min()} to {self.combined_data['timestamp'].max()}\n\n")
            
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            overall = evaluation_results['overall_metrics']
            f.write(f"RMSE: {overall['rmse']:.2f} MW\n")
            f.write(f"MAE: {overall['mae']:.2f} MW\n")
            f.write(f"R²: {overall['r2']:.3f}\n")
            f.write(f"MAPE: {overall['mape']:.2f}%\n")
            f.write(f"Bias: {overall['bias']:.2f} MW\n\n")
            
            f.write("HOURLY PERFORMANCE (Top 5 worst hours)\n")
            f.write("-" * 40 + "\n")
            hourly_rmse = [(h, evaluation_results['hourly_metrics'][h]['rmse']) 
                          for h in evaluation_results['hourly_metrics']]
            hourly_rmse.sort(key=lambda x: x[1], reverse=True)
            for hour, rmse in hourly_rmse[:5]:
                f.write(f"Hour {hour:02d}:00 - RMSE: {rmse:.2f} MW\n")
            
            f.write("\nMONTHLY PERFORMANCE (Top 5 worst months)\n")
            f.write("-" * 40 + "\n")
            monthly_rmse = [(m, evaluation_results['monthly_metrics'][m]['rmse']) 
                           for m in evaluation_results['monthly_metrics']]
            monthly_rmse.sort(key=lambda x: x[1], reverse=True)
            for month, rmse in monthly_rmse[:5]:
                f.write(f"Month {month:02d} - RMSE: {rmse:.2f} MW\n")
        
        logger.info(f"Report saved to: {report_path}")
    
    def run_complete_pipeline(self) -> Dict:
        """
        Run the complete baseline pipeline.
        
        Returns:
        --------
        Dict
            Complete pipeline results
        """
        logger.info("=" * 80)
        logger.info("STARTING BASELINE IRRADIANCE MODEL PIPELINE")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Preprocess data
            logger.info("Step 1: Preprocessing data...")
            df = self.preprocess_data()
            
            # Step 2: Create features
            logger.info("Step 2: Creating features...")
            df_features = self.create_features(df)
            
            # Step 3: Train model with rolling window
            logger.info("Step 3: Training model...")
            training_results = self.train_rolling_window_model(df_features)
            
            # Step 4: Evaluate model
            logger.info("Step 4: Evaluating model...")
            evaluation_results = self.evaluate_model()
            
            # Step 5: Create visualizations
            logger.info("Step 5: Creating visualizations...")
            self.create_visualizations(evaluation_results)
            
            # Step 6: Save results
            logger.info("Step 6: Saving results...")
            self.save_results(evaluation_results)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Final results
            results = {
                'success': True,
                'model_type': self.model_type,
                'rolling_window_days': self.rolling_window_days,
                'total_time_seconds': total_time,
                'metrics': evaluation_results,
                'output_directory': str(self.output_dir)
            }
            
            logger.info("=" * 80)
            logger.info("BASELINE PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            logger.info(f"Output directory: {self.output_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {'success': False, 'error': str(e)}


def main():
    """Main function to run the baseline pipeline."""
    print("Baseline Irradiance Model for German Solar PV Production")
    print("=" * 60)
    
    # Initialize model
    model = BaselineIrradianceModel(
        model_type='linear',  # or 'ridge'
        rolling_window_days=30,
        min_irradiance_threshold=10.0
    )
    
    # Run pipeline
    results = model.run_complete_pipeline()
    
    if results['success']:
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Overall RMSE: {results['metrics']['overall_metrics']['rmse']:.2f} MW")
        print(f"📈 Overall R²: {results['metrics']['overall_metrics']['r2']:.3f}")
        print(f"📁 Results saved to: {results['output_directory']}")
    else:
        print(f"\n❌ Pipeline failed: {results['error']}")


if __name__ == "__main__":
    main()
