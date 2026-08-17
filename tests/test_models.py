import pytest
import numpy as np
import pandas as pd
from smart_train_ai.schema import FaultType, FaultLocation
from smart_train_ai.models.anomaly import AnomalyDetector
from smart_train_ai.models.classifier import FaultClassifier
from smart_train_ai.models.localization import FaultLocalizer
from smart_train_ai.models.health import HealthEngine


def test_health_engine_normal():
    engine = HealthEngine()
    features = {
        "imu_mag_rms": 1.01,
        "imu_z_kurtosis": 0.0,
        "bearing_temp_slope": 0.02,
        "bearing_temp_rise": 1.0,
        "current_per_rpm": 0.001,
    }
    score, status, priority, evidence = engine.compute_health(
        anomaly_score=0.05,
        fault_type="NORMAL",
        fault_severity=0,
        confidence=0.95,
        features=features,
    )
    assert score >= 75
    assert status == "HEALTHY"
    assert priority == "LOW"


def test_health_engine_fault():
    engine = HealthEngine()
    features = {
        "imu_mag_rms": 2.5,
        "imu_z_kurtosis": 6.0,
        "bearing_temp_slope": 0.45,
        "bearing_temp_rise": 12.0,
        "current_per_rpm": 0.005,
    }
    score, status, priority, evidence = engine.compute_health(
        anomaly_score=0.85,
        fault_type="BEARING_FAULT",
        fault_severity=2,
        confidence=0.90,
        features=features,
    )
    assert score < 60
    assert status in ["WARNING", "CRITICAL"]
    assert priority in ["HIGH", "CRITICAL"]
    assert len(evidence) >= 2


def test_isolation_forest_and_xgb_fit():
    X = pd.DataFrame({
        "feat1": np.random.normal(0, 1, 50),
        "feat2": np.random.normal(5, 1, 50),
    })
    y = pd.Series(["NORMAL"] * 25 + ["BEARING_FAULT"] * 25)

    detector = AnomalyDetector(contamination=0.05)
    detector.fit(X.iloc[:25])
    is_anom, scores = detector.predict(X)
    assert len(is_anom) == 50

    clf = FaultClassifier(n_estimators=10)
    clf.fit(X, y)
    preds, confs, imps = clf.predict_with_confidence(X)
    assert len(preds) == 50
    assert len(confs) == 50

    loc = FaultLocalizer(n_estimators=10)
    y_loc = pd.Series(["NONE"] * 25 + ["REAR_BOGIE_AXLE_2"] * 25)
    loc.fit(X, y_loc)
    loc_preds = loc.predict(X)
    assert len(loc_preds) == 50
