#!/usr/bin/env python3
"""
Simplified Candidate Models for German Solar PV Production Prediction
====================================================================

This module implements simplified candidate models for predicting quarter-hourly 
national solar PV production in Germany, including:

1. Random Forest (scikit-learn) - Classical ensemble model
2. Simple Neural Network (scikit-learn) - Basic deep learning model

Features include:
- Lagged irradiance values
- Clear-sky index calculations
- Calendar effects
- Rolling statistics
- Advanced temporal features
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleFeatureEngineer:
    """Simplified feature engineering for solar PV prediction."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def calculate_clear_sky_index(self, irradiance: pd.Series, theoretical_max: pd.Series) -> pd.Series:
        """Calculate clear sky index (CSI)."""
        csi = irradiance / theoretical_max
        csi = csi.clip(0, 1)  # Clip to [0, 1]
        return csi
    
    def calculate_theoretical_max_irradiance(self, timestamps: pd.Series, latitude: float = 51.0) -> pd.Series:
        """Calculate theoretical maximum irradiance using solar geometry."""
        theoretical_max = []
        
        for timestamp in timestamps:
            # Solar position calculation (simplified)
            date = timestamp.date()
            hour = timestamp.hour + timestamp.minute / 60.0
            
            # Day of year
            day_of_year = date.timetuple().tm_yday
            
            # Solar declination (approximate)
            declination = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 80)))
            
            # Hour angle
            hour_angle = 15 * (hour - 12)  # Degrees
            
            # Solar zenith angle
            zenith_rad = np.arccos(
                np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
                np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * 
                np.cos(np.radians(hour_angle))
            )
            
            # Extraterrestrial irradiance
            solar_constant = 1367  # W/m²
            distance_factor = 1 + 0.034 * np.cos(np.radians(360/365 * (day_of_year - 2)))
            
            # Theoretical maximum (simplified)
            if zenith_rad < np.pi/2:  # Above horizon
                theoretical = solar_constant * distance_factor * np.cos(zenith_rad)
                theoretical = max(0, theoretical)
            else:
                theoretical = 0
                
            theoretical_max.append(theoretical)
        
        return pd.Series(theoretical_max, index=timestamps.index)
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for modeling."""
        logger.info("Creating advanced features...")
        logger.info(f"Input DataFrame shape: {df.shape}")
        logger.info(f"Input DataFrame columns: {list(df.columns)}")
        
        df_features = df.copy()
        
        # Basic time features
        df_features['hour'] = df_features['timestamp'].dt.hour
        df_features['month'] = df_features['timestamp'].dt.month
        df_features['day_of_year'] = df_features['timestamp'].dt.dayofyear
        df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
        df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
        
        # Binary features
        df_features['is_daytime'] = ((df_features['hour'] >= 6) & (df_features['hour'] <= 18)).astype(int)
        df_features['is_peak_hours'] = ((df_features['hour'] >= 10) & (df_features['hour'] <= 16)).astype(int)
        
        # Lag features for irradiance
        for lag in [1, 4, 8, 24, 96]:  # 15min, 1h, 2h, 6h, 1day
            df_features[f'irradiance_lag_{lag}'] = df_features['irradiance_w_m2'].shift(lag)
        
        # Lag features for generation
        for lag in [1, 4, 8, 24, 96]:
            df_features[f'generation_lag_{lag}'] = df_features['solar_generation_mw'].shift(lag)
        
        # Rolling statistics for irradiance
        for window in [4, 8, 24, 96]:  # 1h, 2h, 6h, 1day
            df_features[f'irradiance_rolling_mean_{window}'] = df_features['irradiance_w_m2'].rolling(window).mean()
            df_features[f'irradiance_rolling_std_{window}'] = df_features['irradiance_w_m2'].rolling(window).std()
        
        # Rolling statistics for generation
        for window in [4, 8, 24, 96]:
            df_features[f'generation_rolling_mean_{window}'] = df_features['solar_generation_mw'].rolling(window).mean()
            df_features[f'generation_rolling_std_{window}'] = df_features['solar_generation_mw'].rolling(window).std()
        
        # Clear sky index
        theoretical_max = self.calculate_theoretical_max_irradiance(df_features['timestamp'])
        df_features['clear_sky_index'] = self.calculate_clear_sky_index(
            df_features['irradiance_w_m2'], theoretical_max
        )
        
        # Solar position features (simplified calculation)
        declination = 23.45 * np.sin(np.radians(360/365 * (df_features['day_of_year'] - 80)))
        hour_angle = 15 * (df_features['hour'] - 12)
        
        zenith_angle = np.arccos(
            np.sin(np.radians(51.0)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(51.0)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
        )
        df_features['solar_elevation'] = np.maximum(0, 90 - np.degrees(zenith_angle))
        
        # Synthetic PV signal (simplified)
        df_features['synthetic_pv_signal'] = (
            df_features['irradiance_w_m2'] * 
            df_features['clear_sky_index'] * 
            np.cos(np.radians(df_features['solar_elevation'])) * 
            0.15  # Typical panel efficiency
        )
        
        # Holiday features (German holidays - simplified)
        df_features['is_holiday'] = (
            (df_features['month'] == 1) & (df_features['timestamp'].dt.day == 1) |  # New Year
            (df_features['month'] == 5) & (df_features['timestamp'].dt.day == 1) |  # Labor Day
            (df_features['month'] == 10) & (df_features['timestamp'].dt.day == 3)   # German Unity Day
        ).astype(int)
        
        # Remove NaN values
        initial_count = len(df_features)
        df_features = df_features.dropna()
        final_count = len(df_features)
        
        logger.info(f"Feature engineering completed: {initial_count - final_count} records removed due to NaN")
        logger.info(f"Final feature count: {len(df_features.columns)}")
        
        # Store feature columns for later use
        self.feature_columns = [col for col in df_features.columns 
                               if col not in ['timestamp', 'solar_generation_mw']]
        
        # Debug: print feature columns
        logger.info(f"Feature columns: {self.feature_columns}")
        logger.info(f"DataFrame columns: {list(df_features.columns)}")
        
        return df_features

class RandomForestModel:
    """Random Forest model for solar PV prediction."""
    
    def __init__(self, output_dir="random_forest_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model parameters
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.feature_importance = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """Train the Random Forest model."""
        logger.info("Training Random Forest model...")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Random Forest training completed!")
        return {'model': self.model, 'feature_importance': self.feature_importance}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        import joblib
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

class NeuralNetworkModel:
    """Neural Network model for solar PV prediction."""
    
    def __init__(self, output_dir="neural_network_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model parameters
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.scaler = StandardScaler()
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """Train the Neural Network model."""
        logger.info("Training Neural Network model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        logger.info("Neural Network training completed!")
        return {'model': self.model, 'scaler': self.scaler}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        import joblib
        joblib.dump((self.model, self.scaler), filepath)
        logger.info(f"Model saved to {filepath}")

class SimpleCandidateModelsPipeline:
    """Pipeline for training and evaluating simplified candidate models."""
    
    def __init__(self, output_dir="simple_candidate_models_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.feature_engineer = SimpleFeatureEngineer()
        self.rf_model = RandomForestModel(str(output_dir) + "/random_forest")
        self.nn_model = NeuralNetworkModel(str(output_dir) + "/neural_network")
        
        self.results = {}
        
    def load_and_preprocess_data(self, irradiance_path: str, pv_path: str) -> pd.DataFrame:
        """Load and preprocess data for modeling."""
        logger.info("Loading and preprocessing data...")
        
        try:
            # Load data (simplified - you can adapt from baseline_model_fixed.py)
            irradiance_data = pd.read_parquet(irradiance_path)
            pv_data = pd.read_parquet(pv_path)
            
            # Merge data
            merged_data = pd.merge(pv_data, irradiance_data, on='timestamp', how='inner')
            
            # Filter night-time data
            merged_data = merged_data[merged_data['irradiance_w_m2'] >= 10.0]
            
            # Create advanced features
            processed_data = self.feature_engineer.create_advanced_features(merged_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Create synthetic data for testing
            logger.info("Creating synthetic data for testing...")
            synthetic_data = self._create_synthetic_data()
            # Apply feature engineering to synthetic data
            return self.feature_engineer.create_advanced_features(synthetic_data)
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic data for testing when real data is unavailable."""
        # Generate 3 months of 15-minute data
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2018, 4, 1)
        
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
                seasonal_factor = 0.3 + 0.7 * np.sin((month - 1) * np.pi / 6)
                daily_factor = np.sin((hour - 6) * np.pi / 12)
                noise = np.random.normal(0, 0.1)
                base_capacity = 50000  # MW
                generation = base_capacity * seasonal_factor * daily_factor * (1 + noise)
                generation = max(0, generation)
                
                # Realistic irradiance
                irradiance = 800 * seasonal_factor * daily_factor * (1 + np.random.normal(0, 0.2))
                irradiance = max(0, irradiance)
            else:
                generation = 0
                irradiance = 0
            
            data.append({
                'timestamp': timestamp,
                'solar_generation_mw': round(generation, 2),
                'irradiance_w_m2': round(irradiance, 2)
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Created {len(df):,} synthetic records")
        return df
    
    def train_random_forest(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")
        
        # Debug: check feature columns
        logger.info(f"Feature columns: {self.feature_engineer.feature_columns}")
        logger.info(f"Data columns: {list(data.columns)}")
        
        # Prepare features and target
        X = data[self.feature_engineer.feature_columns]
        y = data['solar_generation_mw']
        
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Split data
        split_idx = int(len(data) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        train_result = self.rf_model.train(X_train, y_train, X_test, y_test)
        
        # Make predictions
        y_pred = self.rf_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Random Forest Results: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")
        
        self.results['random_forest'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actuals': y_test,
            'feature_importance': train_result['feature_importance']
        }
        
        return self.results['random_forest']
    
    def train_neural_network(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Train Neural Network model."""
        logger.info("Training Neural Network model...")
        
        # Prepare features and target
        X = data[self.feature_engineer.feature_columns]
        y = data['solar_generation_mw']
        
        # Split data
        split_idx = int(len(data) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        train_result = self.nn_model.train(X_train, y_train, X_test, y_test)
        
        # Make predictions
        y_pred = self.nn_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Neural Network Results: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")
        
        self.results['neural_network'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actuals': y_test
        }
        
        return self.results['neural_network']
    
    def create_comparison_visualizations(self):
        """Create comparison visualizations."""
        logger.info("Creating comparison visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison
        models = list(self.results.keys())
        rmse_values = [self.results[model]['metrics']['rmse'] for model in models]
        mae_values = [self.results[model]['metrics']['mae'] for model in models]
        r2_values = [self.results[model]['metrics']['r2'] for model in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        axes[0, 0].bar(x - width, rmse_values, width, label='RMSE', alpha=0.8)
        axes[0, 0].bar(x, mae_values, width, label='MAE', alpha=0.8)
        axes[0, 0].bar(x + width, [r * 10000 for r in r2_values], width, label='R² × 10000', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Error (MW) / R² × 10000')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Random Forest predictions vs actuals
        if 'random_forest' in self.results:
            axes[0, 1].scatter(self.results['random_forest']['actuals'], 
                              self.results['random_forest']['predictions'], alpha=0.5, s=1)
            axes[0, 1].plot([0, max(self.results['random_forest']['actuals'])], 
                           [0, max(self.results['random_forest']['actuals'])], 'r--', lw=2)
            axes[0, 1].set_xlabel('Actual PV Generation (MW)')
            axes[0, 1].set_ylabel('Predicted PV Generation (MW)')
            axes[0, 1].set_title('Random Forest: Predictions vs Actuals')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Neural Network predictions vs actuals
        if 'neural_network' in self.results:
            axes[1, 0].scatter(self.results['neural_network']['actuals'], 
                              self.results['neural_network']['predictions'], alpha=0.5, s=1)
            axes[1, 0].plot([0, max(self.results['neural_network']['actuals'])], 
                           [0, max(self.results['neural_network']['actuals'])], 'r--', lw=2)
            axes[1, 0].set_xlabel('Actual PV Generation (MW)')
            axes[1, 0].set_ylabel('Predicted PV Generation (MW)')
            axes[1, 0].set_title('Neural Network: Predictions vs Actuals')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature importance (Random Forest)
        if 'random_forest' in self.results:
            top_features = self.results['random_forest']['feature_importance'].head(10)
            axes[1, 1].barh(range(len(top_features)), top_features['importance'])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features['feature'])
            axes[1, 1].set_xlabel('Feature Importance')
            axes[1, 1].set_title('Random Forest: Top 10 Feature Importance')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save all results."""
        logger.info("Saving results...")
        
        # Save metrics
        metrics_df = pd.DataFrame({
            model: self.results[model]['metrics'] 
            for model in self.results.keys()
        }).T
        
        metrics_df.to_csv(self.output_dir / 'model_metrics.csv')
        
        # Save detailed results
        for model_name, result in self.results.items():
            result_df = pd.DataFrame({
                'actual': result['actuals'],
                'predicted': result['predictions']
            })
            result_df.to_csv(self.output_dir / f'{model_name}_predictions.csv', index=False)
        
        # Save feature importance
        if 'random_forest' in self.results:
            self.results['random_forest']['feature_importance'].to_csv(
                self.output_dir / 'random_forest_feature_importance.csv', index=False
            )
        
        logger.info(f"Results saved to {self.output_dir}")

def main():
    """Main function to run the simplified candidate models pipeline."""
    print("="*80)
    print("🚀 SIMPLIFIED CANDIDATE MODELS PIPELINE")
    print("="*80)
    
    # Initialize pipeline
    pipeline = SimpleCandidateModelsPipeline()
    
    # Load and preprocess data
    # Note: Update these paths to your actual data locations
    irradiance_path = "data_3years_2018_2020_final/ssrd_germany_2018_2020_combined_15min"
    pv_path = "data_german_solar_generation/processed/german_solar_generation_v3_2018_2020.parquet"
    
    try:
        data = pipeline.load_and_preprocess_data(irradiance_path, pv_path)
        print(f"✅ Data loaded: {len(data):,} records")
        
        # Train Random Forest
        print("\n🌳 Training Random Forest...")
        rf_results = pipeline.train_random_forest(data)
        
        # Train Neural Network
        print("\n🧠 Training Neural Network...")
        nn_results = pipeline.train_neural_network(data)
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        pipeline.create_comparison_visualizations()
        
        # Save results
        print("\n💾 Saving results...")
        pipeline.save_results()
        
        print("\n" + "="*80)
        print("✅ SIMPLIFIED CANDIDATE MODELS PIPELINE COMPLETED!")
        print("="*80)
        
        # Print summary
        print("\n📈 RESULTS SUMMARY:")
        for model_name, result in pipeline.results.items():
            metrics = result['metrics']
            print(f"  {model_name.upper()}:")
            print(f"    RMSE: {metrics['rmse']:.2f} MW")
            print(f"    MAE:  {metrics['mae']:.2f} MW")
            print(f"    R²:   {metrics['r2']:.3f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
