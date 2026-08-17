import os
import glob
import numpy as np
import pandas as pd
from typing import Tuple, List
from smart_train_ai.preprocessing.validation import validate_dataframe
from smart_train_ai.preprocessing.synchronization import synchronize_and_resample
from smart_train_ai.preprocessing.windowing import extract_sliding_windows
from smart_train_ai.features.extractor import FeatureExtractor
from smart_train_ai.models.classifier import FaultClassifier


def load_dataset_features(data_dir: str = "data/synthetic") -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads raw dataset runs, extracts window features, and tags every window with its run_id group.

    Returns:
    - X: Feature DataFrame
    - y: Fault label Series
    - groups: run_id group Series for zero-leakage GroupKFold splitting
    """
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

    feature_list: List[dict] = []
    labels: List[str] = []
    groups: List[str] = []

    for run_df in run_dfs:
        clean_df, _ = validate_dataframe(run_df)
        if clean_df.empty:
            continue

        run_id = clean_df["run_id"].iloc[0]
        sync_df = synchronize_and_resample(clean_df, target_hz=100.0)
        windows, _, metas = extract_sliding_windows(sync_df, window_size_seconds=5.0, overlap=0.5)

        for w, meta in zip(windows, metas):
            feat = extractor.extract_features(w)
            feature_list.append(feat)
            labels.append(meta["fault_type"])
            groups.append(run_id)

    X = pd.DataFrame(feature_list)
    y = pd.Series(labels, name="fault_type")
    g = pd.Series(groups, name="run_id")

    return X, y, g


def train_classifier_model(
    data_dir: str = "data/synthetic",
    model_output_path: str = "models_artifacts/classifier_model.joblib",
) -> FaultClassifier:
    """Trains XGBoost fault classifier with strict run_id group splitting to prevent data leakage."""
    print("--- Training Stage 2: XGBoost Fault Classifier ---")
    X, y, groups = load_dataset_features(data_dir)
    print(f"Loaded total {len(X)} windows across {groups.nunique()} unique physical run_ids.")

    # Unique run_ids split (80% train runs, 20% test runs)
    unique_runs = groups.unique()
    rng = np.random.default_rng(42)
    train_runs = set(rng.choice(unique_runs, size=int(len(unique_runs) * 0.80), replace=False))

    train_mask = groups.isin(train_runs)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    print(f"Train set: {len(X_train)} windows ({len(train_runs)} runs). Test set: {len(X_test)} windows ({len(unique_runs) - len(train_runs)} runs).")

    classifier = FaultClassifier(n_estimators=150, max_depth=6, learning_rate=0.05)
    classifier.fit(X_train, y_train)

    preds, confs, _ = classifier.predict_with_confidence(X_test)
    acc = float(np.mean(np.array(preds) == y_test.values))
    print(f"Test Accuracy on unseen run_ids: {acc * 100:.2f}%")

    classifier.save(model_output_path)
    print(f"XGBoost Fault Classifier saved to {model_output_path}")

    return classifier


if __name__ == "__main__":
    train_classifier_model()
