import pytest
import numpy as np
from smart_train_ai.schema import FaultType
from smart_train_ai.simulator.generator import generate_run_dataframe
from smart_train_ai.preprocessing.synchronization import synchronize_and_resample
from smart_train_ai.features.time_domain import calculate_time_domain_features
from smart_train_ai.features.frequency_domain import calculate_frequency_domain_features
from smart_train_ai.features.extractor import FeatureExtractor


def test_time_domain_features():
    sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    feats = calculate_time_domain_features(sig, prefix="test")
    assert feats["test_mean"] == 3.0
    assert feats["test_min"] == 1.0
    assert feats["test_max"] == 5.0
    assert feats["test_ptp"] == 4.0


def test_frequency_domain_features():
    t = np.linspace(0, 1.0, 100, endpoint=False)
    sig = np.sin(2 * np.pi * 10.0 * t)  # 10Hz sine wave
    feats = calculate_frequency_domain_features(sig, prefix="test", sampling_hz=100.0)
    assert abs(feats["test_dominant_freq"] - 10.0) <= 1.0
    assert feats["test_spectral_energy"] > 0.0


def test_feature_extractor():
    raw_df = generate_run_dataframe(FaultType.BEARING_FAULT, run_id="TEST_RUN", duration_seconds=5.0)
    sync_df = synchronize_and_resample(raw_df, target_hz=100.0)

    extractor = FeatureExtractor(sampling_hz=100.0)
    feats = extractor.extract_features(sync_df)

    assert "imu_z_rms" in feats
    assert "imu_z_dominant_freq" in feats
    assert "bearing_temp_slope" in feats
    assert "current_per_rpm" in feats
