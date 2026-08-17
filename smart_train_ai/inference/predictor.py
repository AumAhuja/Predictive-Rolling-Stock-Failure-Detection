import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from smart_train_ai.schema import (
    SensorRecord,
    PredictionResult,
    FaultType,
    FaultLocation,
)
from smart_train_ai.preprocessing.validation import validate_dataframe
from smart_train_ai.preprocessing.synchronization import synchronize_and_resample
from smart_train_ai.preprocessing.windowing import extract_sliding_windows, SIGNAL_CHANNELS
from smart_train_ai.features.extractor import FeatureExtractor
from smart_train_ai.models.anomaly import AnomalyDetector
from smart_train_ai.models.classifier import FaultClassifier
from smart_train_ai.models.localization import FaultLocalizer
from smart_train_ai.models.health import HealthEngine


class Predictor:
    """Integrated Inference Predictor performing end-to-end diagnostic pipeline execution."""

    def __init__(
        self,
        models_dir: str = "models_artifacts",
        sampling_hz: float = 100.0,
    ):
        self.sampling_hz = sampling_hz
        self.extractor = FeatureExtractor(sampling_hz=self.sampling_hz)
        self.health_engine = HealthEngine()

        anomaly_path = os.path.join(models_dir, "anomaly_model.joblib")
        classifier_path = os.path.join(models_dir, "classifier_model.joblib")
        localizer_path = os.path.join(models_dir, "localization_model.joblib")

        self.anomaly_detector = AnomalyDetector.load(anomaly_path) if os.path.exists(anomaly_path) else None
        self.fault_classifier = FaultClassifier.load(classifier_path) if os.path.exists(classifier_path) else None
        self.fault_localizer = FaultLocalizer.load(localizer_path) if os.path.exists(localizer_path) else None

    def predict_window(
        self,
        records: List[SensorRecord],
        retain_raw_window: bool = True,
    ) -> PredictionResult:
        """Runs full diagnostic pipeline on a 5s window of SensorRecord items."""
        if not records:
            raise ValueError("Empty records list provided for prediction.")

        data = [r.model_dump() for r in records]
        df = pd.DataFrame(data)

        clean_df, _ = validate_dataframe(df)
        if clean_df.empty:
            clean_df = df  # fallback if clean drops everything in small sample

        sync_df = synchronize_and_resample(clean_df, target_hz=self.sampling_hz)
        windows, raw_matrices, metas = extract_sliding_windows(
            sync_df, window_size_seconds=5.0, overlap=0.5, sampling_hz=self.sampling_hz
        )

        if not windows:
            # Fallback if window is shorter than 5 seconds
            window_df = sync_df
            raw_matrix = sync_df[[c for c in SIGNAL_CHANNELS if c in sync_df.columns]].values
            train_id = sync_df["train_id"].iloc[0] if "train_id" in sync_df.columns else "TRAIN001"
        else:
            window_df = windows[-1]
            raw_matrix = raw_matrices[-1]
            train_id = metas[-1].get("train_id", "TRAIN001")

        # 1. Feature Extraction
        features = self.extractor.extract_features(window_df)
        feat_df = pd.DataFrame([features])

        # 2. Stage 1: Anomaly Detection
        if self.anomaly_detector is not None:
            is_anomaly_arr, anomaly_score_arr = self.anomaly_detector.predict(feat_df)
            is_anomaly = bool(is_anomaly_arr[0])
            anomaly_score = float(anomaly_score_arr[0])
        else:
            is_anomaly = False
            anomaly_score = 0.05

        # 3. Stage 2: Fault Classification
        if self.fault_classifier is not None:
            fault_preds, confs, top_imps = self.fault_classifier.predict_with_confidence(feat_df)
            fault_type_str = fault_preds[0]
            confidence = float(confs[0])
        else:
            fault_type_str = FaultType.NORMAL.value
            confidence = 0.95

        # 4. Stage 3: Fault Localization
        if self.fault_localizer is not None and fault_type_str != FaultType.NORMAL.value:
            loc_preds = self.fault_localizer.predict(feat_df)
            location_str = loc_preds[0]
        else:
            location_str = FaultLocation.NONE.value if fault_type_str == FaultType.NORMAL.value else FaultLocation.MOTOR.value

        # Severity estimation (0 for normal, 1-3 based on anomaly score/features)
        severity = 0 if fault_type_str == FaultType.NORMAL.value else int(np.clip(anomaly_score * 3.5, 1, 3))

        # 5. Stage 4: Health Score Engine
        health_score, status, priority, evidence = self.health_engine.compute_health(
            anomaly_score=anomaly_score,
            fault_type=fault_type_str,
            fault_severity=severity,
            confidence=confidence,
            features=features,
        )

        raw_window_dict = None
        if retain_raw_window and raw_matrix is not None:
            raw_window_dict = {
                "channels": [c for c in SIGNAL_CHANNELS if c in sync_df.columns],
                "shape": list(raw_matrix.shape),
                "data_sample": raw_matrix[:5].tolist(),  # sample preview
            }

        return PredictionResult(
            train_id=train_id,
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 2),
            fault_type=FaultType(fault_type_str),
            fault_location=FaultLocation(location_str),
            confidence=round(confidence, 2),
            health_score=health_score,
            status=status,
            maintenance_priority=priority,
            evidence=evidence,
            raw_window=raw_window_dict,
        )
