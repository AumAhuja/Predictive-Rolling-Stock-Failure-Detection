import pandas as pd
import numpy as np


def synchronize_and_resample(
    df: pd.DataFrame,
    target_hz: float = 100.0,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Synchronizes multi-rate sensors (IMU @ 100Hz, INA219 @ 10Hz, DS18B20 @ 1Hz, HX711 @ 80Hz)
    to a single uniform target frequency grid using forward-fill / linear interpolation.
    """
    if df.empty:
        return df

    df_sorted = df.sort_values(by=time_col).copy()

    min_time = df_sorted[time_col].min()
    max_time = df_sorted[time_col].max()
    duration = max_time - min_time

    if duration <= 0:
        return df_sorted

    dt = 1.0 / target_hz
    num_samples = int(np.round(duration * target_hz)) + 1
    target_grid = np.linspace(min_time, max_time, num_samples)

    # Columns to interpolate numerically
    numeric_cols = [
        "imu_x", "imu_y", "imu_z",
        "motor_temperature", "bearing_temperature", "ambient_temperature",
        "rpm", "voltage", "current", "power", "load"
    ]
    numeric_cols = [c for c in numeric_cols if c in df_sorted.columns]

    # Meta columns to forward-fill
    meta_cols = ["run_id", "train_id", "zone", "fault_type", "fault_location", "fault_severity"]
    meta_cols = [c for c in meta_cols if c in df_sorted.columns]

    resampled_data = {time_col: target_grid}

    for col in numeric_cols:
        # Interpolate numeric values onto uniform grid
        resampled_data[col] = np.interp(
            target_grid,
            df_sorted[time_col].values,
            df_sorted[col].values
        )

    resampled_df = pd.DataFrame(resampled_data)

    # Forward-fill metadata
    for col in meta_cols:
        # Match nearest time index for categorical metadata
        idx = np.searchsorted(df_sorted[time_col].values, target_grid, side="right") - 1
        idx = np.clip(idx, 0, len(df_sorted) - 1)
        resampled_df[col] = df_sorted[col].values[idx]

    return resampled_df
