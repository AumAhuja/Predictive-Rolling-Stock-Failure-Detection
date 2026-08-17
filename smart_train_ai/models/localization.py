import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from xgboost import XGBClassifier
from smart_train_ai.schema import FaultLocation


class FaultLocalizer:
    """Classifies physical fault locations across rolling stock components."""

    LOCATIONS = [
        FaultLocation.NONE.value,
        FaultLocation.MOTOR.value,
        FaultLocation.FRONT_BOGIE.value,
        FaultLocation.FRONT_BOGIE_AXLE_1.value,
        FaultLocation.FRONT_BOGIE_AXLE_2.value,
        FaultLocation.REAR_BOGIE.value,
        FaultLocation.REAR_BOGIE_AXLE_1.value,
        FaultLocation.REAR_BOGIE_AXLE_2.value,
        FaultLocation.BODY.value,
        FaultLocation.BRAKE.value,
    ]

    def __init__(self, n_estimators: int = 100, max_depth: int = 5, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            eval_metric="mlogloss",
        )
        self.feature_names: List[str] = []
        self.classes_: List[str] = []
        self.label_to_int: Dict[str, int] = {}
        self.int_to_label: Dict[int, str] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits localization model on feature DataFrame X and location Series y using contiguous class mapping."""
        self.feature_names = list(X.columns)
        str_y = y.map(lambda v: v if isinstance(v, str) else v.value).values
        unique_classes = sorted(list(set(str_y)))
        self.classes_ = unique_classes

        # Map to contiguous integers 0..N-1 for XGBoost
        self.label_to_int = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        self.int_to_label = {i: cls_name for i, cls_name in enumerate(unique_classes)}

        y_int = np.array([self.label_to_int[v] for v in str_y])
        self.model.fit(X.values, y_int)
        return self

    def predict(self, X: pd.DataFrame) -> List[str]:
        """Predicts FaultLocation strings for each sample."""
        X_vals = X[self.feature_names].values if self.feature_names else X.values
        preds_int = self.model.predict(X_vals)
        return [self.int_to_label.get(int(idx), FaultLocation.NONE.value) for idx in preds_int]

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
    def load(cls, filepath: str) -> "FaultLocalizer":
        """Loads trained model artifact."""
        data = joblib.load(filepath)
        obj = cls()
        obj.model = data["model"]
        obj.feature_names = data["feature_names"]
        obj.classes_ = data.get("classes_", [])
        obj.label_to_int = data.get("label_to_int", {})
        obj.int_to_label = data.get("int_to_label", {})
        return obj
