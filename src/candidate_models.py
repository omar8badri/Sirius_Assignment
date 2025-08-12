#!/usr/bin/env python3
"""
Candidate Models for German Solar PV Production Prediction
=========================================================

This module implements advanced candidate models for predicting quarter-hourly 
national solar PV production in Germany, including:

1. LightGBM (Gradient Boosting) - Classical ensemble model
2. Temporal CNN - Deep learning model for time series

Features include:
- Lagged irradiance values
- Clear-sky index calculations
- Sunrise/sunset times
- Calendar/holiday effects
- Asset orientation statistics
- Synthetic PV signals
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

# LightGBM
import lightgbm as lgb

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Astronomy for sunrise/sunset
import ephem

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Advanced feature engineering for solar PV prediction."""
    
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
    
    def calculate_sunrise_sunset(self, timestamps: pd.Series, latitude: float = 51.0, longitude: float = 10.0) -> pd.DataFrame:
        """Calculate sunrise and sunset times."""
        observer = ephem.Observer()
        observer.lat = str(latitude)
        observer.lon = str(longitude)
        
        sunrise_times = []
        sunset_times = []
        solar_noon_times = []
        
        for timestamp in timestamps:
            date = timestamp.date()
            observer.date = date
            
            # Calculate sun position
            sun = ephem.Sun()
            sun.compute(observer)
            
            # Get sunrise and sunset
            sunrise = observer.next_rising(sun).datetime()
            sunset = observer.next_setting(sun).datetime()
            
            # Solar noon (simplified)
            solar_noon = sunrise + (sunset - sunrise) / 2
            
            sunrise_times.append(sunrise)
            sunset_times.append(sunset)
            solar_noon_times.append(solar_noon)
        
        return pd.DataFrame({
            'sunrise_time': sunrise_times,
            'sunset_time': sunset_times,
            'solar_noon_time': solar_noon_times
        }, index=timestamps.index)
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for modeling."""
        logger.info("Creating advanced features...")
        
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
        
        # Sunrise/sunset features
        sun_times = self.calculate_sunrise_sunset(df_features['timestamp'])
        df_features['hours_since_sunrise'] = (
            df_features['timestamp'] - sun_times['sunrise_time']
        ).dt.total_seconds() / 3600
        df_features['hours_until_sunset'] = (
            sun_times['sunset_time'] - df_features['timestamp']
        ).dt.total_seconds() / 3600
        
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
        
        return df_features

class LightGBMModel:
    """LightGBM model for solar PV prediction."""
    
    def __init__(self, output_dir="lightgbm_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model parameters
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        self.model = None
        self.feature_importance = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """Train the LightGBM model."""
        logger.info("Training LightGBM model...")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [train_data, val_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.model.feature_name(),
            'importance': self.model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        logger.info("LightGBM training completed!")
        return {'model': self.model, 'feature_importance': self.feature_importance}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is not None:
            self.model.save_model(filepath)
            logger.info(f"Model saved to {filepath}")

class TemporalCNNDataset(Dataset):
    """Dataset for temporal CNN."""
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 96, target_col: str = 'solar_generation_mw'):
        self.data = data
        self.sequence_length = sequence_length
        self.target_col = target_col
        
        # Get feature columns (exclude timestamp and target)
        self.feature_cols = [col for col in data.columns 
                            if col not in ['timestamp', target_col]]
        
        # Normalize features
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(data[self.feature_cols])
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(sequence_length, len(data)):
            sequence = self.features_scaled[i-sequence_length:i]
            target = data.iloc[i][target_col]
            
            self.sequences.append(sequence)
            self.targets.append(target)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]), 
                torch.FloatTensor([self.targets[idx]]))

class TemporalCNN(nn.Module):
    """Temporal CNN for solar PV prediction."""
    
    def __init__(self, input_size: int, sequence_length: int = 96, hidden_size: int = 64):
        super(TemporalCNN, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        # Transpose to: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Remove last dimension
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class TemporalCNNModel:
    """Temporal CNN model wrapper."""
    
    def __init__(self, output_dir="temporal_cnn_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, train_dataset: TemporalCNNDataset, 
              val_dataset: TemporalCNNDataset = None,
              epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001) -> Dict:
        """Train the temporal CNN model."""
        logger.info(f"Training Temporal CNN on {self.device}...")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        
        # Initialize model
        input_size = len(train_dataset.feature_cols)
        self.model = TemporalCNN(input_size=input_size).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_features, batch_targets in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        outputs = self.model(batch_features)
                        loss = criterion(outputs, batch_targets)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        
        logger.info("Temporal CNN training completed!")
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def predict(self, dataset: TemporalCNNDataset) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                features, _ = dataset[i]
                features = features.unsqueeze(0).to(self.device)
                output = self.model(features)
                predictions.append(output.cpu().numpy()[0, 0])
        
        return np.array(predictions)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is not None:
            torch.save(self.model.state_dict(), filepath)
            logger.info(f"Model saved to {filepath}")

class CandidateModelsPipeline:
    """Pipeline for training and evaluating candidate models."""
    
    def __init__(self, output_dir="candidate_models_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.feature_engineer = AdvancedFeatureEngineer()
        self.lightgbm_model = LightGBMModel(output_dir / "lightgbm")
        self.cnn_model = TemporalCNNModel(output_dir / "temporal_cnn")
        
        self.results = {}
        
    def load_and_preprocess_data(self, irradiance_path: str, pv_path: str) -> pd.DataFrame:
        """Load and preprocess data for modeling."""
        logger.info("Loading and preprocessing data...")
        
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
    
    def train_lightgbm(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Train LightGBM model."""
        logger.info("Training LightGBM model...")
        
        # Prepare features and target
        X = data[self.feature_engineer.feature_columns]
        y = data['solar_generation_mw']
        
        # Split data
        split_idx = int(len(data) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        train_result = self.lightgbm_model.train(X_train, y_train, X_test, y_test)
        
        # Make predictions
        y_pred = self.lightgbm_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"LightGBM Results: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")
        
        self.results['lightgbm'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actuals': y_test,
            'feature_importance': train_result['feature_importance']
        }
        
        return self.results['lightgbm']
    
    def train_temporal_cnn(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Train Temporal CNN model."""
        logger.info("Training Temporal CNN model...")
        
        # Prepare data
        X = data[self.feature_engineer.feature_columns]
        y = data['solar_generation_mw']
        
        # Split data
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        # Create datasets
        train_dataset = TemporalCNNDataset(train_data, sequence_length=96)
        test_dataset = TemporalCNNDataset(test_data, sequence_length=96)
        
        # Train model
        train_result = self.cnn_model.train(train_dataset, test_dataset, epochs=30)
        
        # Make predictions
        y_pred = self.cnn_model.predict(test_dataset)
        y_test = test_data['solar_generation_mw'].iloc[96:].values  # Account for sequence length
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Temporal CNN Results: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")
        
        self.results['temporal_cnn'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actuals': y_test,
            'train_losses': train_result['train_losses']
        }
        
        return self.results['temporal_cnn']
    
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
        
        # 2. LightGBM predictions vs actuals
        if 'lightgbm' in self.results:
            axes[0, 1].scatter(self.results['lightgbm']['actuals'], 
                              self.results['lightgbm']['predictions'], alpha=0.5, s=1)
            axes[0, 1].plot([0, max(self.results['lightgbm']['actuals'])], 
                           [0, max(self.results['lightgbm']['actuals'])], 'r--', lw=2)
            axes[0, 1].set_xlabel('Actual PV Generation (MW)')
            axes[0, 1].set_ylabel('Predicted PV Generation (MW)')
            axes[0, 1].set_title('LightGBM: Predictions vs Actuals')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Temporal CNN predictions vs actuals
        if 'temporal_cnn' in self.results:
            axes[1, 0].scatter(self.results['temporal_cnn']['actuals'], 
                              self.results['temporal_cnn']['predictions'], alpha=0.5, s=1)
            axes[1, 0].plot([0, max(self.results['temporal_cnn']['actuals'])], 
                           [0, max(self.results['temporal_cnn']['actuals'])], 'r--', lw=2)
            axes[1, 0].set_xlabel('Actual PV Generation (MW)')
            axes[1, 0].set_ylabel('Predicted PV Generation (MW)')
            axes[1, 0].set_title('Temporal CNN: Predictions vs Actuals')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature importance (LightGBM)
        if 'lightgbm' in self.results:
            top_features = self.results['lightgbm']['feature_importance'].head(10)
            axes[1, 1].barh(range(len(top_features)), top_features['importance'])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features['feature'])
            axes[1, 1].set_xlabel('Feature Importance')
            axes[1, 1].set_title('LightGBM: Top 10 Feature Importance')
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
        if 'lightgbm' in self.results:
            self.results['lightgbm']['feature_importance'].to_csv(
                self.output_dir / 'lightgbm_feature_importance.csv', index=False
            )
        
        logger.info(f"Results saved to {self.output_dir}")

def main():
    """Main function to run the candidate models pipeline."""
    print("="*80)
    print("🚀 CANDIDATE MODELS PIPELINE")
    print("="*80)
    
    # Initialize pipeline
    pipeline = CandidateModelsPipeline()
    
    # Load and preprocess data
    # Note: Update these paths to your actual data locations
    irradiance_path = "data_3years_2018_2020_final/ssrd_germany_2018_2020_combined_15min"
    pv_path = "data_german_solar_generation/processed/german_solar_generation_v3_2018_2020.parquet"
    
    try:
        data = pipeline.load_and_preprocess_data(irradiance_path, pv_path)
        print(f"✅ Data loaded: {len(data):,} records")
        
        # Train LightGBM
        print("\n🌳 Training LightGBM...")
        lightgbm_results = pipeline.train_lightgbm(data)
        
        # Train Temporal CNN
        print("\n🧠 Training Temporal CNN...")
        cnn_results = pipeline.train_temporal_cnn(data)
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        pipeline.create_comparison_visualizations()
        
        # Save results
        print("\n💾 Saving results...")
        pipeline.save_results()
        
        print("\n" + "="*80)
        print("✅ CANDIDATE MODELS PIPELINE COMPLETED!")
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
