import pandas as pd
import numpy as np
from typing import Dict, Any, List
from smart_train_ai.features.time_domain import calculate_time_domain_features
from smart_train_ai.features.frequency_domain import calculate_frequency_domain_features


class FeatureExtractor:
    """Master Feature Extractor converting a 5-second sensor DataFrame window into engineered features."""

    def __init__(self, sampling_hz: float = 100.0):
        self.sampling_hz = sampling_hz

    def extract_features(self, window_df: pd.DataFrame) -> Dict[str, float]:
        """Extracts complete feature dictionary from a single window DataFrame."""
        features: Dict[str, float] = {}

        if window_df.empty:
            return features

        dt = 1.0 / self.sampling_hz

        # 1. IMU Features (X, Y, Z)
        for channel in ["imu_x", "imu_y", "imu_z"]:
            if channel in window_df.columns:
                sig = window_df[channel].values
                # Time domain
                features.update(calculate_time_domain_features(sig, prefix=channel))
                # Frequency domain
                features.update(calculate_frequency_domain_features(sig, prefix=channel, sampling_hz=self.sampling_hz))

        # Composite IMU Vector Magnitude: sqrt(x^2 + y^2 + z^2)
        if all(c in window_df.columns for c in ["imu_x", "imu_y", "imu_z"]):
            imu_mag = np.sqrt(window_df["imu_x"].values**2 + window_df["imu_y"].values**2 + window_df["imu_z"].values**2)
            features.update(calculate_time_domain_features(imu_mag, prefix="imu_mag"))
            features.update(calculate_frequency_domain_features(imu_mag, prefix="imu_mag", sampling_hz=self.sampling_hz))

        # 2. RPM Features
        if "rpm" in window_df.columns:
            rpm_sig = window_df["rpm"].values
            features["rpm_mean"] = float(np.mean(rpm_sig))
            features["rpm_std"] = float(np.std(rpm_sig))
            features["rpm_min"] = float(np.min(rpm_sig))
            features["rpm_max"] = float(np.max(rpm_sig))
            features["rpm_ptp"] = float(features["rpm_max"] - features["rpm_min"])
            features["rpm_variation"] = float(features["rpm_ptp"] / (features["rpm_std"] + 1e-6))

            # RPM acceleration: d(RPM)/dt
            rpm_acc = np.gradient(rpm_sig, dt)
            features["rpm_accel_mean"] = float(np.mean(rpm_acc))
            features["rpm_accel_max"] = float(np.max(np.abs(rpm_acc)))

        # 3. Temperature Features
        amb_temp = float(np.mean(window_df["ambient_temperature"].values)) if "ambient_temperature" in window_df.columns else 25.0
        features["ambient_temp_mean"] = amb_temp

        for temp_col in ["motor_temperature", "bearing_temperature"]:
            if temp_col in window_df.columns:
                t_sig = window_df[temp_col].values
                prefix = temp_col.replace("_temperature", "_temp")
                mean_t = float(np.mean(t_sig))
                features[f"{prefix}_mean"] = mean_t
                features[f"{prefix}_min"] = float(np.min(t_sig))
                features[f"{prefix}_max"] = float(np.max(t_sig))
                features[f"{prefix}_std"] = float(np.std(t_sig))
                # Temperature rise delta
                features[f"{prefix}_rise"] = float(features[f"{prefix}_max"] - features[f"{prefix}_min"])
                # Slope dT/dt
                slope = float(np.polyfit(np.arange(len(t_sig)) * dt, t_sig, 1)[0]) if len(t_sig) > 1 else 0.0
                features[f"{prefix}_slope"] = slope
                # Difference from ambient
                features[f"{prefix}_delta_ambient"] = float(mean_t - amb_temp)

        # 4. Electrical Features (INA219)
        if "voltage" in window_df.columns and "current" in window_df.columns:
            v_sig = window_df["voltage"].values
            i_sig = window_df["current"].values
            p_sig = window_df["power"].values if "power" in window_df.columns else v_sig * i_sig

            features["voltage_mean"] = float(np.mean(v_sig))
            features["current_mean"] = float(np.mean(i_sig))
            features["current_max"] = float(np.max(i_sig))
            features["current_std"] = float(np.std(i_sig))
            features["power_mean"] = float(np.mean(p_sig))
            features["power_max"] = float(np.max(p_sig))
            features["power_std"] = float(np.std(p_sig))

            # Derived relationships: Current / RPM and dCurrent / dRPM
            mean_rpm = features.get("rpm_mean", 0.0)
            if mean_rpm > 1.0:
                features["current_per_rpm"] = float(features["current_mean"] / mean_rpm)
            else:
                features["current_per_rpm"] = 0.0

            # Delta current / Delta RPM ratio
            d_current = float(np.max(i_sig) - np.min(i_sig))
            d_rpm = features.get("rpm_ptp", 0.0)
            features["dcurrent_drpm_ratio"] = float(d_current / (d_rpm + 1e-4))

        # 5. Load Features (HX711)
        if "load" in window_df.columns:
            load_sig = window_df["load"].values
            features["load_mean"] = float(np.mean(load_sig))
            features["load_min"] = float(np.min(load_sig))
            features["load_max"] = float(np.max(load_sig))
            features["load_std"] = float(np.std(load_sig))
            features["load_ptp"] = float(features["load_max"] - features["load_min"])

        return features

    def extract_features_batch(self, windows: List[pd.DataFrame]) -> pd.DataFrame:
        """Extracts features for a list of windows and returns a single DataFrame."""
        rows = [self.extract_features(w) for w in windows]
        return pd.DataFrame(rows)
