import os
import glob
import numpy as np
import pandas as pd
from typing import Tuple, List
from smart_train_ai.preprocessing.validation import validate_dataframe
from smart_train_ai.preprocessing.synchronization import synchronize_and_resample
from smart_train_ai.preprocessing.windowing import extract_sliding_windows
from smart_train_ai.features.extractor import FeatureExtractor
from smart_train_ai.models.localization import FaultLocalizer


def load_localization_features(data_dir: str = "data/synthetic") -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Loads dataset windows tagged with fault location labels and run_id groups."""
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
    locations: List[str] = []
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
            locations.append(meta["fault_location"])
            groups.append(run_id)

    X = pd.DataFrame(feature_list)
    y = pd.Series(locations, name="fault_location")
    g = pd.Series(groups, name="run_id")

    return X, y, g


def train_localization_model(
    data_dir: str = "data/synthetic",
    model_output_path: str = "models_artifacts/localization_model.joblib",
) -> FaultLocalizer:
    """Trains Fault Localizer model enforcing run_id group splitting."""
    print("--- Training Stage 3: Fault Localizer ---")
    X, y, groups = load_localization_features(data_dir)
    print(f"Loaded {len(X)} windows across {groups.nunique()} unique run_ids for localization.")

    unique_runs = groups.unique()
    rng = np.random.default_rng(42)
    train_runs = set(rng.choice(unique_runs, size=int(len(unique_runs) * 0.80), replace=False))

    train_mask = groups.isin(train_runs)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    localizer = FaultLocalizer(n_estimators=100, max_depth=5)
    localizer.fit(X_train, y_train)

    preds = localizer.predict(X_test)
    acc = float(np.mean(np.array(preds) == y_test.values))
    print(f"Localization Accuracy on unseen run_ids: {acc * 100:.2f}%")

    localizer.save(model_output_path)
    print(f"Fault Localizer saved to {model_output_path}")

    return localizer


if __name__ == "__main__":
    train_localization_model()
