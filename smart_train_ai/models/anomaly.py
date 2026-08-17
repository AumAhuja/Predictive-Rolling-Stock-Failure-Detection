import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Isolation Forest Anomaly Detector trained on normal baseline operating features."""

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self.feature_names: list = []

    def fit(self, X: pd.DataFrame):
        """Fits Isolation Forest model on NORMAL feature DataFrame."""
        self.feature_names = list(X.columns)
        self.model.fit(X.values)
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts anomaly status and normalized anomaly scores.

        Returns:
        - is_anomaly_arr: boolean array (True if anomaly, False if normal)
        - anomaly_score_arr: normalized score between 0.0 (very normal) and 1.0 (extreme anomaly)
        """
        X_vals = X[self.feature_names].values if self.feature_names else X.values

        # Isolation forest score_samples returns raw scores (higher = normal, lower = anomalous)
        raw_scores = self.model.score_samples(X_vals)
        # Normal inliers score ~ -0.35 to -0.42; Outliers score ~ -0.55 to -0.85
        # Map: -0.35 -> 0.0, -0.70 -> 1.0
        anomaly_scores = np.clip((-raw_scores - 0.38) / 0.32, 0.0, 1.0)

        preds = self.model.predict(X_vals)
        is_anomaly = preds == -1

        return is_anomaly, anomaly_scores

    def save(self, filepath: str):
        """Saves trained model artifact."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "AnomalyDetector":
        """Loads trained model artifact."""
        data = joblib.load(filepath)
        obj = cls()
        obj.model = data["model"]
        obj.feature_names = data["feature_names"]
        return obj
