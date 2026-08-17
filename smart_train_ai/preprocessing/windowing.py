import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any

SIGNAL_CHANNELS = [
    "imu_x", "imu_y", "imu_z",
    "motor_temperature", "bearing_temperature", "ambient_temperature",
    "rpm", "voltage", "current", "power", "load"
]


def extract_sliding_windows(
    df: pd.DataFrame,
    window_size_seconds: float = 5.0,
    overlap: float = 0.5,
    sampling_hz: float = 100.0,
    time_col: str = "timestamp",
) -> Tuple[List[pd.DataFrame], List[np.ndarray], List[Dict[str, Any]]]:
    """
    Extracts sliding windows from synchronized sensor data.

    Returns:
    - windows: List of DataFrame windows
    - raw_matrices: List of raw 3D temporal arrays shape (steps, num_channels) for DL
    - window_metas: List of metadata dictionaries (run_id, fault_type, severity, location, etc.)
    """
    if df.empty:
        return [], [], []

    step_samples = int(np.round(window_size_seconds * sampling_hz))
    stride_samples = max(int(np.round(step_samples * (1.0 - overlap))), 1)

    available_channels = [c for c in SIGNAL_CHANNELS if c in df.columns]

    windows: List[pd.DataFrame] = []
    raw_matrices: List[np.ndarray] = []
    window_metas: List[Dict[str, Any]] = []

    num_rows = len(df)
    for start_idx in range(0, num_rows - step_samples + 1, stride_samples):
        end_idx = start_idx + step_samples
        window_df = df.iloc[start_idx:end_idx].copy()

        raw_matrix = window_df[available_channels].values

        meta = {
            "run_id": window_df["run_id"].iloc[0] if "run_id" in window_df.columns else "UNKNOWN",
            "train_id": window_df["train_id"].iloc[0] if "train_id" in window_df.columns else "TRAIN001",
            "start_timestamp": float(window_df[time_col].iloc[0]),
            "end_timestamp": float(window_df[time_col].iloc[-1]),
            "fault_type": window_df["fault_type"].iloc[-1] if "fault_type" in window_df.columns else "NORMAL",
            "fault_location": window_df["fault_location"].iloc[-1] if "fault_location" in window_df.columns else "NONE",
            "fault_severity": int(window_df["fault_severity"].iloc[-1]) if "fault_severity" in window_df.columns else 0,
            "channels": available_channels,
        }

        windows.append(window_df)
        raw_matrices.append(raw_matrix)
        window_metas.append(meta)

    return windows, raw_matrices, window_metas
