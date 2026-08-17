import os
import glob
import pandas as pd
from typing import Tuple
from smart_train_ai.schema import FaultType
from smart_train_ai.preprocessing.validation import validate_dataframe
from smart_train_ai.preprocessing.synchronization import synchronize_and_resample
from smart_train_ai.preprocessing.windowing import extract_sliding_windows
from smart_train_ai.features.extractor import FeatureExtractor
from smart_train_ai.models.anomaly import AnomalyDetector


def train_anomaly_model(
    data_dir: str = "data/synthetic",
    model_output_path: str = "models_artifacts/anomaly_model.joblib",
) -> AnomalyDetector:
    """Loads normal operating dataset runs, extracts features, and trains Isolation Forest anomaly model."""
    print("--- Training Stage 1: Isolation Forest Anomaly Detector ---")

    files = glob.glob(os.path.join(data_dir, "RUN_*.csv")) + glob.glob(os.path.join(data_dir, "RUN_*.parquet"))
    if not files:
        csv_master = os.path.join(data_dir, "master_synthetic_dataset.csv")
        if os.path.exists(csv_master):
            full_df = pd.read_csv(csv_master)
            run_dfs = [group for _, group in full_df.groupby("run_id")]
        else:
            raise FileNotFoundError(f"No dataset found in {data_dir}. Run generate_dataset.py first.")
    else:
        run_dfs = [pd.read_csv(f) if f.endswith(".csv") else pd.read_parquet(f) for f in files]

    extractor = FeatureExtractor(sampling_hz=100.0)
    normal_feature_rows = []

    for run_df in run_dfs:
        clean_df, _ = validate_dataframe(run_df)
        if clean_df.empty:
            continue
        # Check if run is NORMAL
        fault = clean_df["fault_type"].iloc[0]
        if fault != FaultType.NORMAL.value:
            continue

        sync_df = synchronize_and_resample(clean_df, target_hz=100.0)
        windows, _, _ = extract_sliding_windows(sync_df, window_size_seconds=5.0, overlap=0.5)

        for w in windows:
            feat = extractor.extract_features(w)
            normal_feature_rows.append(feat)

    X_normal = pd.DataFrame(normal_feature_rows)
    print(f"Extracted {len(X_normal)} NORMAL window feature vectors (features={X_normal.shape[1]}).")

    detector = AnomalyDetector(contamination=0.05, n_estimators=100)
    detector.fit(X_normal)
    detector.save(model_output_path)
    print(f"Isolation Forest Anomaly Model successfully saved to {model_output_path}")

    return detector


if __name__ == "__main__":
    train_anomaly_model()
