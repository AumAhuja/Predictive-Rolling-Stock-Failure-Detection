import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from smart_train_ai.models.anomaly import AnomalyDetector
from smart_train_ai.models.classifier import FaultClassifier
from smart_train_ai.models.localization import FaultLocalizer
from smart_train_ai.training.train_classifier import load_dataset_features


def evaluate_models(
    data_dir: str = "data/synthetic",
    models_dir: str = "models_artifacts",
    reports_dir: str = "reports",
):
    """Evaluates all trained models on an independent run_id test split and saves metrics/plots."""
    os.makedirs(reports_dir, exist_ok=True)
    print("--- Evaluating Smart Train AI Models ---")

    X, y, groups = load_dataset_features(data_dir)

    # Enforce run_id split (same 20% test set)
    unique_runs = groups.unique()
    rng = np.random.default_rng(42)
    train_runs = set(rng.choice(unique_runs, size=int(len(unique_runs) * 0.80), replace=False))

    test_mask = ~groups.isin(train_runs)
    X_test, y_test = X[test_mask], y[test_mask]

    metrics_report = {}

    # 1. Anomaly Detector Evaluation
    anomaly_path = os.path.join(models_dir, "anomaly_model.joblib")
    if os.path.exists(anomaly_path):
        detector = AnomalyDetector.load(anomaly_path)
        is_anom, scores = detector.predict(X_test)
        y_true_anom = (y_test != "NORMAL").values

        fpr = float(np.mean(is_anom[~y_true_anom])) if np.sum(~y_true_anom) > 0 else 0.0
        fnr = float(np.mean(~is_anom[y_true_anom])) if np.sum(y_true_anom) > 0 else 0.0

        metrics_report["anomaly_detection"] = {
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "mean_normal_anomaly_score": round(float(np.mean(scores[~y_true_anom])), 4) if np.sum(~y_true_anom) > 0 else 0.0,
            "mean_fault_anomaly_score": round(float(np.mean(scores[y_true_anom])), 4) if np.sum(y_true_anom) > 0 else 0.0,
        }
        print(f"Anomaly Detector -> FPR: {fpr*100:.2f}%, FNR: {fnr*100:.2f}%")

    # 2. Fault Classifier Evaluation
    classifier_path = os.path.join(models_dir, "classifier_model.joblib")
    if os.path.exists(classifier_path):
        classifier = FaultClassifier.load(classifier_path)
        preds, confs, _ = classifier.predict_with_confidence(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        metrics_report["fault_classification"] = {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
        }
        print(f"Fault Classifier -> Accuracy: {acc*100:.2f}%, Macro F1: {macro_f1:.4f}")

        # Confusion Matrix Plot
        labels = sorted(list(set(y_test).union(set(preds))))
        cm = confusion_matrix(y_test, preds, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("XGBoost Fault Classification Confusion Matrix (Unseen run_ids)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        cm_plot_path = os.path.join(reports_dir, "classifier_confusion_matrix.png")
        plt.savefig(cm_plot_path)
        plt.close()

    # Save metrics JSON
    report_json_path = os.path.join(reports_dir, "evaluation_metrics.json")
    with open(report_json_path, "w") as f:
        json.dump(metrics_report, f, indent=2)

    print(f"Evaluation report saved to {report_json_path}")
    return metrics_report


if __name__ == "__main__":
    evaluate_models()
