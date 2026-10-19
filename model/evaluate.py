"""
Machine Learning Model Evaluation Module
Author: Jayprakash Maurya (T.Y. B.Sc. IT - Vidyalankar School of Information Technology)
Project: Detecting Water Leaks Using Acoustic Signals

This script evaluates trained models on the held-out TEST SET (15% unseen files):
1. Evaluates best model on test metrics: Accuracy, Precision, Recall, F1-Score.
2. Computes and saves the Confusion Matrix as model/confusion_matrix.png.
3. Evaluates all candidate models on the test set and exports model/model_comparison.csv.
4. Strictly prints only computed, empirical test values.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Ensure project root is in sys.path
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_PROJECT_DIR not in sys.path:
    sys.path.insert(0, BASE_PROJECT_DIR)

from feature_extraction.features import FEATURE_NAMES

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_PROJECT_DIR, "dataset")
FEATURES_CSV = os.path.join(DATASET_DIR, "extracted_features.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "leak_model.joblib")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
COMPARISON_CSV_PATH = os.path.join(MODEL_DIR, "model_comparison.csv")
SPLIT_INFO_PATH = os.path.join(MODEL_DIR, "split_info.json")


def load_model_and_test_data():
    """Load model artifact, split manifest, and held-out test data."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run 'python model/train.py' first!"
        )

    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Features file not found at {FEATURES_CSV}.")

    model_artifact = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURES_CSV)

    if os.path.exists(SPLIT_INFO_PATH):
        with open(SPLIT_INFO_PATH, "r") as f:
            split_info = json.load(f)
        test_files = set(split_info["test_files"])
    else:
        test_files = set(model_artifact["split_info"]["test_files"])

    test_df = df[df["file_id"].isin(test_files)].copy()
    return model_artifact, test_df


def plot_and_save_confusion_matrix(y_true, y_pred, model_name: str):
    """Save high-resolution Confusion Matrix visualization for project report."""
    cm = confusion_matrix(y_true, y_pred)
    class_labels = ["NORMAL", "POSSIBLE LEAK"]

    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format="d")

    ax.set_title(
        f"Test Set Confusion Matrix\nModel: {model_name}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Predicted Condition", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Condition", fontsize=11, fontweight="bold")

    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close(fig)
    print(f"Confusion Matrix saved to: {CONFUSION_MATRIX_PATH}")


def evaluate_all_models(all_models: dict, X_test, y_test) -> pd.DataFrame:
    """Compare all candidate models on the exact same unseen test set."""
    rows = []
    for name, pipeline in all_models.items():
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
        })

    comparison_df = pd.DataFrame(rows)
    # Sort descending by F1-Score
    comparison_df = comparison_df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    return comparison_df


def main():
    print("=" * 70)
    print("STEP 4: MODEL EVALUATION ON UNSEEN TEST SET")
    print("=" * 70)

    model_artifact, test_df = load_model_and_test_data()
    best_name = model_artifact["best_model_name"]
    best_model = model_artifact["best_model"]
    all_models = model_artifact.get("all_models", {best_name: best_model})

    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df["label"].values

    n_files = test_df["file_id"].nunique()
    n_segments = len(test_df)
    print(f"Evaluating on {n_files} held-out test audio files ({n_segments} 1-second segments).")
    print(f"Test Set Ground Truth: {dict(test_df['label'].value_counts())} (0=Normal, 1=Leak)\n")

    # Evaluate best model
    y_pred_best = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_best)
    prec = precision_score(y_test, y_pred_best, zero_division=0)
    rec = recall_score(y_test, y_pred_best, zero_division=0)
    f1 = f1_score(y_test, y_pred_best, zero_division=0)

    print("=" * 70)
    print(f"BEST MODEL TEST METRICS ({best_name})")
    print("=" * 70)
    print(f"Test Accuracy:  {acc * 100:.2f}% ({acc:.4f})")
    print(f"Test Precision: {prec * 100:.2f}% ({prec:.4f})")
    print(f"Test Recall:    {rec * 100:.2f}% ({rec:.4f})")
    print(f"Test F1-Score:  {f1 * 100:.2f}% ({f1:.4f})")
    print("=" * 70)

    # Save Confusion Matrix
    plot_and_save_confusion_matrix(y_test, y_pred_best, best_name)

    # Compare all candidate models on test set
    comparison_df = evaluate_all_models(all_models, X_test, y_test)
    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)
    print(f"Model comparison saved to: {COMPARISON_CSV_PATH}\n")

    print("ALL CANDIDATE MODELS - TEST SET BENCHMARK:")
    print("-" * 70)
    print(comparison_df.to_string(index=False))
    print("-" * 70)
    print("Model evaluation successfully completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
