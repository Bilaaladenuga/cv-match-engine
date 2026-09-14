"""
Phase 12 — Baseline model training.

Trains three scikit-learn classifiers on the Phase 12 feature tables and
saves the artifacts under ml/models/:

    Logistic Regression   linear reference point, interpretable coefficients
    Random Forest         nonlinear, robust to unscaled features
    Gradient Boosting     usually the strongest tabular baseline

Class imbalance (~50/25/25) is handled with class weights / sample weights.
Each model is evaluated on the held-out test feature table with per-class
metrics so no single number can hide a failing class (Phase 13 will expand
this into a full evaluation report).

Usage (from backend/ with the venv active):
    python ../ml/training/train_baseline.py
Artifacts:
    ml/models/baseline_<name>.joblib      fitted pipeline
    ml/models/training_report.json        hyperparams + test metrics per model
                                          (includes the label<->int mapping)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "ml" / "models"

MODEL_VERSION = "match-model-v0.3.1-baseline"  # v0.3.1: artifacts carry prior-calibration metadata

LABEL_ORDER = ["No Fit", "Potential Fit", "Good Fit"]


def load_split(split: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(PROCESSED_DIR / f"{split}_features.csv")
    if "extraction_error" in df.columns:
        n_bad = int(df["extraction_error"].notna().sum())
        if n_bad:
            print(f"[{split}] dropping {n_bad} rows with extraction errors")
        df = df[df["extraction_error"].isna()]
    feature_names = _feature_names(df)
    X = df[feature_names]
    y_raw = df["label"]
    y = pd.Series(np.where(y_raw == "Good Fit", 2, np.where(y_raw == "Potential Fit", 1, 0)), index=df.index)
    return X, y, y_raw


def _feature_names(df: pd.DataFrame) -> list[str]:
    meta = {"label", "label_int", "split_row", "extraction_error"}
    return [c for c in df.columns if c not in meta]


def compute_natural_prior() -> dict[str, float]:
    """Class prior of the unstratified source split (data/raw/train.csv).

    Embedded into artifact calibration_ metadata and the training report.
    Falls back to None values (no calibration) when the raw CSV is absent,
    so training on a machine without data/raw still succeeds.
    """
    raw = REPO_ROOT / "data" / "raw" / "train.csv"
    if not raw.exists():
        print("[warn] data/raw/train.csv not found; calibration metadata omitted")
        return {}
    counts = pd.read_csv(raw, usecols=["label"])["label"].value_counts()
    return {label: float(counts.get(label, 0) / counts.sum()) for label in LABEL_ORDER}


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=1.0,
                random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                random_state=42,
            )),
        ]),
    }


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    report = classification_report(
        y_test, pred, target_names=LABEL_ORDER, output_dict=True, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "per_class_f1": {
            name: round(float(f1_score(y_test, pred, labels=[i], average="macro")), 4)
            for i, name in enumerate(LABEL_ORDER)
        },
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": report,
        "mean_max_probability": round(float(np.max(proba, axis=1).mean()), 4),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, _ = load_split("train")
    X_test, y_test, _ = load_split("test")
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"train label counts: {np.bincount(y_train).tolist()} (No/Potential/Good)")

    results: dict[str, dict] = {}
    for name, model in build_models().items():
        t0 = time.time()
        # Gradient boosting has no class_weight; use sample weights instead.
        if name == "gradient_boosting":
            counts = np.bincount(y_train)
            weights = np.array([len(y_train) / (counts[label] * len(counts)) for label in y_train])
            model.fit(X_train, y_train, clf__sample_weight=weights)
        else:
            model.fit(X_train, y_train)
        fit_s = round(time.time() - t0, 1)

        metrics = evaluate(model, X_test, y_test)
        metrics["fit_seconds"] = fit_s
        metrics["model_version"] = MODEL_VERSION
        results[name] = metrics
        print(f"{name}: acc={metrics['accuracy']} macro_f1={metrics['macro_f1']} "
              f"per-class F1={metrics['per_class_f1']} ({fit_s}s)")

        # Calibration metadata (Phase 13): the stratified table teaches a
        # uniform prior; serving corrects to the natural source prior via
        # app/ml/calibration.py. Storing it ON the artifact keeps model and
        # calibration inseparable.
        model.calibration_ = {
            "method": "saerens_prior_correction",
            "natural_prior": compute_natural_prior(),
            "training_prior": {name: 1.0 / 3.0 for name in LABEL_ORDER},
        }
        joblib.dump(model, MODELS_DIR / f"baseline_{name}.joblib")

    results["label_mapping"] = {name: i for i, name in enumerate(LABEL_ORDER)}
    results["natural_prior"] = compute_natural_prior()
    results["feature_names"] = list(X_train.columns)
    results["dataset"] = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "source": "cnamuangtoun/resume-job-description-fit (train subsample 700/class, test 100/class; "
                  "int8-quantized MiniLM embeddings; resume bodies may repeat across the "
                  "upstream train/test boundary - see docs/ml-methodology.md)",
    }
    (MODELS_DIR / "training_report.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\nartifacts -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
