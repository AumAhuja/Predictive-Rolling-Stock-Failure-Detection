import pandas as pd
from typing import Tuple


def validate_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Validates physical bounds of sensor signals and removes invalid or corrupted records.
    Returns clean DataFrame and count of dropped invalid records.
    """
    initial_count = len(df)
    clean_df = df.copy()

    # Drop null timestamps or non-positive timestamps
    clean_df = clean_df.dropna(subset=["timestamp", "run_id"])
    clean_df = clean_df[clean_df["timestamp"] >= 0]

    # Validate physical sensor ranges
    # Temperature bound: -20C to +120C
    temp_cols = ["motor_temperature", "bearing_temperature", "ambient_temperature"]
    for col in temp_cols:
        if col in clean_df.columns:
            clean_df = clean_df[(clean_df[col] >= -20.0) & (clean_df[col] <= 120.0)]

    # RPM bound: >= 0 and <= 5000 RPM
    if "rpm" in clean_df.columns:
        clean_df = clean_df[(clean_df["rpm"] >= 0.0) & (clean_df["rpm"] <= 5000.0)]

    # Electrical bound: Voltage 0 to 30V, Current 0 to 10A
    if "voltage" in clean_df.columns:
        clean_df = clean_df[(clean_df["voltage"] >= 0.0) & (clean_df["voltage"] <= 30.0)]
    if "current" in clean_df.columns:
        clean_df = clean_df[(clean_df["current"] >= 0.0) & (clean_df["current"] <= 10.0)]

    # IMU bound: -16g to +16g
    imu_cols = ["imu_x", "imu_y", "imu_z"]
    for col in imu_cols:
        if col in clean_df.columns:
            clean_df = clean_df[(clean_df[col] >= -16.0) & (clean_df[col] <= 16.0)]

    dropped_count = initial_count - len(clean_df)
    return clean_df, dropped_count
