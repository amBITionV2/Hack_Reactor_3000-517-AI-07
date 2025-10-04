import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class ShipPerformancePredictor:
    def __init__(self, sequence_length=10, feature_dim=12):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.feature_columns = [
            'wind_speed', 'wind_direction', 'wave_height', 'wave_period',
            'current_speed', 'current_direction', 'sea_temp', 'air_temp',
            'pressure', 'visibility', 'ship_heading', 'ship_speed_prev'
        ]
        self.target_columns = ['ship_speed', 'fuel_consumption']
        
    def build_model(self):
        model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, self.feature_dim)),
            Dense(32, activation="relu"),
            Dense(2, activation="linear")  # [speed, fuel]
        ])
        model.compile(optimizer="adam", loss="mse")
        self.model = model
        return model

    def predict(self, features):
        # Ensure features is a numpy array
        features = np.array(features)
        
        # If the input is 2D (a single sequence), expand it to 3D (batch of 1)
        if features.ndim == 2:
            features = np.expand_dims(features, axis=0)
            
        result = self.model.predict(features, verbose=0)[0]
        return {
            "speed": float(result[0]),
            "fuel_consumption": float(result[1])
        }
    
    def save_model(self, filepath='models/ship_performance_model'):
        """Save trained model and scalers"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save Keras model
        self.model.save(f"{filepath}.h5")
        
        # Save scalers
        with open(f"{filepath}_scalers.pkl", 'wb') as f:
            pickle.dump({
                'scaler_features': self.scaler_features,
                'scaler_target': self.scaler_target,
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns,
                'sequence_length': self.sequence_length
            }, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/ship_performance_model'):
        """Load trained model and scalers"""
        # Load Keras model
        # FIXED: Added compile=False to prevent deserialization errors during inference
        self.model = tf.keras.models.load_model(f"{filepath}.h5", compile=False)
        
        # Load scalers (assuming they exist, otherwise handle error)
        scaler_path = f"{filepath}_scalers.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                data = pickle.load(f)
                self.scaler_features = data.get('scaler_features', self.scaler_features)
                self.scaler_target = data.get('scaler_target', self.scaler_target)
                self.feature_columns = data.get('feature_columns', self.feature_columns)
                self.target_columns = data.get('target_columns', self.target_columns)
                self.sequence_length = data.get('sequence_length', self.sequence_length)
        
        print(f"Model loaded from {filepath}")

def train_model(weather_api, bounds, step=5):
    """
    Train ML model on regional weather/ocean conditions.
    Uses fetch_region_conditions to gather training samples.
    """
    from weather_api import fetch_region_conditions

    predictor = ShipPerformancePredictor()
    predictor.build_model()

    # 📡 Get weather data grid across region
    print("Fetching regional weather data for training...")
    region_data = fetch_region_conditions(weather_api, bounds, step=step)
    if not region_data:
        print("No weather data fetched. Cannot train model.")
        return None

    X, y = [], []

    for cond in region_data:
        features = [
            cond.get("wind_speed", 10), cond.get("wind_direction", 180),
            cond.get("wave_height", 2), cond.get("wave_period", 8),
            cond.get("current_speed", 0.5), cond.get("current_direction", 90),
            cond.get("sea_temp", 22), cond.get("air_temp", 20),
            cond.get("pressure", 1013), cond.get("visibility", 10),
            180, 12
        ]

        seq = np.tile(features, (predictor.sequence_length, 1))
        X.append(seq)

        true_speed = max(5, 15 - 0.1 * cond.get("wind_speed", 0) - 0.5 * cond.get("wave_height", 0))
        true_fuel  = 20 + 0.2 * cond.get("wind_speed", 0) + 0.5 * cond.get("wave_height", 0)
        y.append([true_speed, true_fuel])

    X = np.array(X)
    y = np.array(y)

    print(f"Training dataset built: {X.shape[0]} samples")

    predictor.model.fit(X, y, epochs=5, verbose=1)

    os.makedirs("models", exist_ok=True)
    # Save using the instance method which also saves scalers
    predictor.save_model("models/ship_performance_model")

    return predictor

if __name__ == "__main__":
    from weather_api import WeatherOceanAPI

    api = WeatherOceanAPI(openweather_api_key="81da745c6171d7297c8d6943dd0d240e")
    bounds = {"lat_min": -40, "lat_max": 30, "lon_min": 20, "lon_max": 120}
    predictor = train_model(api, bounds, step=5)
    if predictor:
        print("Training complete, model saved.")