"""
Machine Learning Training Module
Author: Jayprakash Maurya (T.Y. B.Sc. IT - Vidyalankar School of Information Technology)
Project: Detecting Water Leaks Using Acoustic Signals

This script trains and compares 4 machine learning models:
1. Support Vector Machine (SVM with RBF kernel)
2. Random Forest Classifier
3. K-Nearest Neighbors (KNN)
4. Decision Tree Classifier

Key ML Engineering Principles:
- Stratified Split by FILE (70% train, 15% validation, 15% test) to prevent data leakage.
- StandardScaler integrated into scikit-learn Pipeline.
- Best model selection based strictly on validation set F1-Score.
- Model saved to model/leak_model.joblib.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

# Suppress minor scikit-learn probability deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is in sys.path
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_PROJECT_DIR not in sys.path:
    sys.path.insert(0, BASE_PROJECT_DIR)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from feature_extraction.features import FEATURE_NAMES, extract_dataset_features

RANDOM_STATE = 42
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_PROJECT_DIR, "dataset")
FEATURES_CSV = os.path.join(DATASET_DIR, "extracted_features.csv")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "leak_model.joblib")
SPLIT_INFO_PATH = os.path.join(MODEL_DIR, "split_info.json")


def load_or_extract_data() -> pd.DataFrame:
    """Load existing extracted features CSV or generate it if missing."""
    if os.path.exists(FEATURES_CSV):
        print(f"Loading existing features from: {FEATURES_CSV}")
        df = pd.read_csv(FEATURES_CSV)
    else:
        print("Features CSV not found. Extracting features from raw dataset...")
        df = extract_dataset_features(DATASET_DIR, output_csv=FEATURES_CSV)
    return df


def split_data_by_file(df: pd.DataFrame):
    """
    Split dataset strictly by FILE into Train (70%), Validation (15%), Test (15%).

    Viva Explanation:
    - If we split randomly by segment, segments from the SAME audio file could
      appear in both training and test sets. This causes severe data leakage,
      producing falsely inflated test metrics.
    - Splitting by distinct FILE ensures the test set contains completely unseen recordings.
    """
    # Get unique files and their corresponding class label
    file_labels = df.groupby("file_id")["label"].first().reset_index()

    # Split into 70% Train, 30% Temp (Val + Test)
    train_files, temp_files = train_test_split(
        file_labels,
        test_size=0.30,
        stratify=file_labels["label"],
        random_state=RANDOM_STATE,
    )

    # Split remaining 30% equally into 15% Validation and 15% Test
    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.50,
        stratify=temp_files["label"],
        random_state=RANDOM_STATE,
    )

    train_set = df[df["file_id"].isin(train_files["file_id"])].copy()
    val_set = df[df["file_id"].isin(val_files["file_id"])].copy()
    test_set = df[df["file_id"].isin(test_files["file_id"])].copy()

    split_info = {
        "train_files": train_files["file_id"].tolist(),
        "val_files": val_files["file_id"].tolist(),
        "test_files": test_files["file_id"].tolist(),
    }

    with open(SPLIT_INFO_PATH, "w") as f:
        json.dump(split_info, f, indent=2)

    return train_set, val_set, test_set, split_info


def get_candidate_models() -> dict:
    """
    Define 4 standard candidate classification models inside Pipelines.
    Each pipeline standardizes features (mean=0, std=1) before classification.
    """
    return {
        "Support Vector Machine (SVM)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, C=1.0, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_STATE)),
        ]),
        "K-Nearest Neighbors (KNN)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        ]),
    }


def train_and_compare(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Train candidate models and compare their validation set performance."""
    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df["label"].values

    X_val = val_df[FEATURE_NAMES].values
    y_val = val_df["label"].values

    models = get_candidate_models()
    results = []

    print("\nTraining and validating candidate models:")
    print("-" * 75)
    print(f"{'Model Name':<30} | {'Val Acc':<8} | {'Val Prec':<8} | {'Val Rec':<8} | {'Val F1':<8}")
    print("-" * 75)

    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None

    for name, pipeline in models.items():
        # Train on training split
        pipeline.fit(X_train, y_train)

        # Predict on validation split
        y_pred = pipeline.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        results.append({
            "model_name": name,
            "pipeline": pipeline,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

        print(f"{name:<30} | {acc:.4f}   | {prec:.4f}   | {rec:.4f}   | {f1:.4f}")

        # If score is strictly better, or equal score with higher ensemble priority
        priority_order = ["Random Forest", "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)", "Decision Tree"]
        is_better = (f1 > best_f1) or (
            np.isclose(f1, best_f1, atol=1e-5)
            and priority_order.index(name) < priority_order.index(best_model_name if best_model_name else name)
        )
        if is_better:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline

    print("-" * 75)
    print(f"Best Model Selected: {best_model_name} (Validation F1 = {best_f1:.4f})")
    return results, best_model_name, best_pipeline, best_f1


def main():
    print("=" * 70)
    print("STEP 4: MODEL TRAINING & ALGORITHM COMPARISON")
    print("=" * 70)

    # 1. Load extracted features
    df = load_or_extract_data()
    print(f"Total dataset records: {len(df)} segments across {df['file_id'].nunique()} files.")

    # 2. Stratified file split
    train_df, val_df, test_df, split_info = split_data_by_file(df)
    print(f"Train set:      {len(split_info['train_files'])} files ({len(train_df)} segments)")
    print(f"Validation set: {len(split_info['val_files'])} files ({len(val_df)} segments)")
    print(f"Test set:       {len(split_info['test_files'])} files ({len(test_df)} segments)")

    # 3. Train and compare models
    results, best_name, best_pipeline, best_f1 = train_and_compare(train_df, val_df)

    # 4. Save trained models and best model artifact
    # Also save dictionary of all trained candidate models so evaluate.py can compare them on test set
    all_trained_models = {r["model_name"]: r["pipeline"] for r in results}

    model_artifact = {
        "best_model": best_pipeline,
        "best_model_name": best_name,
        "best_val_f1": best_f1,
        "feature_names": FEATURE_NAMES,
        "all_models": all_trained_models,
        "split_info": split_info,
    }

    joblib.dump(model_artifact, MODEL_SAVE_PATH)
    print(f"\nModel artifact successfully saved to: {MODEL_SAVE_PATH}")
    print("Saved files split manifest to:        " + SPLIT_INFO_PATH)
    print("Ready for evaluation on held-out test set (run model/evaluate.py).")
    print("=" * 70)


if __name__ == "__main__":
    main()
