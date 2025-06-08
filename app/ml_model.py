import os
import threading

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLModel:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.model = None
            self.scaler = None
            self.model_path = "model.pkl"
            self.scaler_path = "scaler.pkl"
            self.feature_names = None
            self.initialized = True
            self.load_or_create_model()

    def load_or_create_model(self):
        """Load existing model or create a new one using California housing dataset"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            housing = fetch_california_housing()
            self.feature_names = housing.feature_names
            print("Model loaded successfully")
        else:
            print("Creating new model...")
            housing = fetch_california_housing()
            X, y = housing.data, housing.target
            self.feature_names = housing.feature_names

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            self.model = RandomForestRegressor(
                n_estimators=50,  # Reduced for faster predictions
                max_depth=8,  # Reduced for faster predictions
                random_state=42,
                n_jobs=1,  # Single thread for consistency
            )
            self.model.fit(X_train_scaled, y_train)

            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)

            X_test_scaled = self.scaler.transform(X_test)
            score = self.model.score(X_test_scaled, y_test)
            print(f"Model R² score: {score:.4f}")

    def predict(self, features):
        """Make prediction for house price"""
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        prediction = self.model.predict(features_scaled)[0]
        return prediction * 100000

    def get_feature_info(self):
        """Get information about the features"""
        return {
            "feature_names": list(self.feature_names),
            "num_features": len(self.feature_names),
            "description": "California housing dataset features",
        }


# Initialize model as singleton
ml_model = MLModel()
