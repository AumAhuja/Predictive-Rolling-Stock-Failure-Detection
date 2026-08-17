import pytest
import pandas as pd
import numpy as np
from smart_train_ai.schema import FaultType
from smart_train_ai.simulator.generator import generate_run_dataframe
from smart_train_ai.preprocessing.validation import validate_dataframe
from smart_train_ai.preprocessing.synchronization import synchronize_and_resample
from smart_train_ai.preprocessing.windowing import extract_sliding_windows


def test_validation():
    raw_df = generate_run_dataframe(FaultType.NORMAL, run_id="TEST_RUN", duration_seconds=5.0)
    clean_df, dropped = validate_dataframe(raw_df)
    assert dropped == 0
    assert len(clean_df) == len(raw_df)


def test_synchronization():
    raw_df = generate_run_dataframe(FaultType.NORMAL, run_id="TEST_RUN", duration_seconds=5.0)
    sync_df = synchronize_and_resample(raw_df, target_hz=100.0)
    assert not sync_df.empty
    assert len(sync_df) >= 500


def test_windowing():
    raw_df = generate_run_dataframe(FaultType.NORMAL, run_id="TEST_RUN", duration_seconds=10.0)
    sync_df = synchronize_and_resample(raw_df, target_hz=100.0)
    windows, raw_matrices, metas = extract_sliding_windows(sync_df, window_size_seconds=5.0, overlap=0.5)

    assert len(windows) >= 2
    assert len(raw_matrices) == len(windows)
    assert raw_matrices[0].shape[0] == 500  # 5s @ 100Hz
    assert metas[0]["run_id"] == "TEST_RUN"
