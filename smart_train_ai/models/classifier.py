import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from xgboost import XGBClassifier
from smart_train_ai.schema import FaultType


class FaultClassifier:
    """XGBoost multiclass classifier for identifying train fault types."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        min_confidence: float = 0.40,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_confidence = min_confidence
        self.random_state = random_state

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric="mlogloss",
        )
        self.feature_names: List[str] = []
        self.classes_: List[str] = []
        self.label_to_int: Dict[str, int] = {}
        self.int_to_label: Dict[int, str] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits XGBoost classifier on feature DataFrame X and label Series y using contiguous class mapping."""
        self.feature_names = list(X.columns)
        str_y = y.map(lambda v: v if isinstance(v, str) else v.value).values
        unique_classes = sorted(list(set(str_y)))
        self.classes_ = unique_classes

        self.label_to_int = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        self.int_to_label = {i: cls_name for i, cls_name in enumerate(unique_classes)}

        y_int = np.array([self.label_to_int[v] for v in str_y])
        self.model.fit(X.values, y_int)
        return self

    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[List[str], List[float], List[Dict[str, float]]]:
        """
        Predicts fault class, confidence score, and top feature importances for each sample.

        Returns:
        - predicted_faults: List of FaultType string values (e.g. BEARING_FAULT, UNKNOWN)
        - confidences: List of float confidence scores (0.0 to 1.0)
        - top_features: List of dictionaries of feature importances
        """
        X_vals = X[self.feature_names].values if self.feature_names else X.values

        probs = self.model.predict_proba(X_vals)
        max_indices = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)

        predicted_faults = []
        confidences = []
        top_features = []

        feature_importances = self.model.feature_importances_
        sorted_feat_idx = np.argsort(feature_importances)[::-1]

        for i in range(len(X_vals)):
            prob = float(max_probs[i])
            pred_idx = int(max_indices[i])

            # Fallback to UNKNOWN if max confidence is lower than min threshold
            if prob < self.min_confidence:
                fault_str = FaultType.UNKNOWN.value
            else:
                fault_str = self.int_to_label.get(pred_idx, FaultType.UNKNOWN.value)

            predicted_faults.append(fault_str)
            confidences.append(prob)

            # Top 3 feature importances for explainability
            top_dict = {}
            for idx in sorted_feat_idx[:3]:
                feat_name = self.feature_names[idx]
                feat_imp = float(feature_importances[idx])
                top_dict[feat_name] = feat_imp

            top_features.append(top_dict)

        return predicted_faults, confidences, top_features

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns sorted feature importances for all model features."""
        if not self.feature_names:
            return {}
        imps = self.model.feature_importances_
        return dict(sorted(zip(self.feature_names, map(float, imps)), key=lambda x: x[1], reverse=True))

    def save(self, filepath: str):
        """Saves trained model artifact."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "classes_": self.classes_,
            "label_to_int": self.label_to_int,
            "int_to_label": self.int_to_label,
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "FaultClassifier":
        """Loads trained model artifact."""
        data = joblib.load(filepath)
        obj = cls()
        obj.model = data["model"]
        obj.feature_names = data["feature_names"]
        obj.classes_ = data.get("classes_", [])
        obj.label_to_int = data.get("label_to_int", {})
        obj.int_to_label = data.get("int_to_label", {})
        return obj
