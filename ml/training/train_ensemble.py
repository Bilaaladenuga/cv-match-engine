"""
Model ensemble — combine GradientBoosting and RandomForest for better generalization.

Ensembling reduces overfitting and improves robustness by combining
multiple models' predictions. This script trains both models and
creates an ensemble that averages their probabilities.

Usage:
    python ml/training/train_ensemble.py

Artifacts:
    ml/models/v0.5_ensemble.joblib   Fitted ensemble pipeline
    ml/models/v0.5_ensemble_report.json   Evaluation report
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "ml" / "models"

MODEL_VERSION = "match-model-v0.5.0-ensemble"
LABEL_ORDER = ["No Fit", "Potential Fit", "Good Fit"]
NATURAL_PRIOR = {"No Fit": 0.5036, "Potential Fit": 0.2493, "Good Fit": 0.2471}


def load_split(split: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED_DIR / f"{split}_features.csv")
    if "extraction_error" in df.columns:
        df = df[df["extraction_error"].isna()]
    meta = {"label", "label_int", "split_row", "extraction_error"}
    feature_names = [c for c in df.columns if c not in meta]
    X = df[feature_names]
    y_raw = df["label"]
    y = pd.Series(
        np.where(y_raw == "Good Fit", 2, np.where(y_raw == "Potential Fit", 1, 0)),
        index=df.index,
    )
    return X, y


def augment_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "required_skill_coverage" in X.columns and "preferred_skill_coverage" in X.columns:
        X["total_skill_coverage"] = X["required_skill_coverage"] * 0.7 + X["preferred_skill_coverage"] * 0.3
    if "n_candidate_skills" in X.columns and "n_required_skills" in X.columns:
        X["skills_per_required"] = X["n_candidate_skills"] / (X["n_required_skills"] + 1)
        X["skill_surplus_ratio"] = (X["n_candidate_skills"] - X["n_required_skills"]) / (X["n_required_skills"] + 1)
    if "n_candidate_skills" in X.columns and "experience_gap_years" in X.columns:
        X["skills_per_year"] = X["n_candidate_skills"] / (X["experience_gap_years"] + 5)
    coverage_cols = [c for c in X.columns if c.startswith("cov_") and c.endswith("_required")]
    if coverage_cols:
        X["mean_category_coverage"] = X[coverage_cols].mean(axis=1)
        X["min_category_coverage"] = X[coverage_cols].min(axis=1)
        X["max_category_coverage"] = X[coverage_cols].max(axis=1)
        X["coverage_variance"] = X[coverage_cols].var(axis=1)
    if "semantic_similarity" in X.columns and "skill_overlap_ratio" in X.columns:
        X["semantic_skill_gap"] = X["semantic_similarity"] - X["skill_overlap_ratio"]
        X["semantic_skill_product"] = X["semantic_similarity"] * X["skill_overlap_ratio"]
    if "education_level_score" in X.columns and "seniority_match" in X.columns:
        X["edu_seniority_alignment"] = X["education_level_score"] * X["seniority_match"]
    if "skills_per_100_words" in X.columns:
        X["skill_density_squared"] = X["skills_per_100_words"] ** 2
    if "job_title_similarity" in X.columns and "skill_overlap_ratio" in X.columns:
        X["title_skill_alignment"] = X["job_title_similarity"] * X["skill_overlap_ratio"]
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return X


class EnsembleClassifier:
    """Simple ensemble that averages probabilities from multiple models."""

    def __init__(self, models: list[tuple[str, Pipeline]], weights: list[float] | None = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.classes_ = None
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        for name, model in self.models:
            print(f"  Training {name}...")
            if "gradient_boosting" in name:
                # Compute sample weights for GB
                counts = np.bincount(y.values)
                n = len(y)
                sample_weights = np.array([n / (counts[label] * 3) for label in y.values])
                model.fit(X, y, clf__sample_weight=sample_weights)
            else:
                model.fit(X, y)
        self.classes_ = model.classes_
        self.feature_names_in_ = X.columns.tolist()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = []
        for name, model in self.models:
            probas.append(model.predict_proba(X))
        # Weighted average
        avg_proba = np.zeros_like(probas[0])
        for proba, weight in zip(probas, self.weights):
            avg_proba += proba * weight
        return avg_proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    per_class_f1 = {
        name: round(float(f1_score(y_test, pred, labels=[i], average="macro")), 4)
        for i, name in enumerate(LABEL_ORDER)
    }

    y_good_binary = (y_test == 2).astype(int)
    good_auc = roc_auc_score(y_good_binary, proba[:, 2])

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "good_vs_rest_auc": round(float(good_auc), 4),
        "log_loss": round(float(log_loss(y_test, proba)), 4),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train_raw, y_train = load_split("train")
    X_test_raw, y_test = load_split("test")

    X_train = augment_features(X_train_raw)
    X_test = augment_features(X_test_raw)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train labels: {dict(zip(LABEL_ORDER, np.bincount(y_train)))}")

    # Define individual models
    gb_model = Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=10,
            subsample=0.9,
            random_state=42,
        )),
    ])

    rf_model = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            random_state=42,
        )),
    ])

    # Create ensemble using sklearn's VotingClassifier
    ensemble = VotingClassifier(
        estimators=[
            ("gradient_boosting", gb_model),
            ("random_forest", rf_model),
            ("logistic_regression", lr_model),
        ],
        voting="soft",  # Average probabilities
        weights=[0.5, 0.3, 0.2],  # GB gets most weight
    )

    # Train
    print("\nTraining ensemble...")
    t0 = time.time()
    ensemble.fit(X_train, y_train)
    fit_time = round(time.time() - t0, 1)
    print(f"Fit time: {fit_time}s")

    # Evaluate
    print("\nEvaluating ensemble...")
    results = evaluate(ensemble, X_test, y_test)
    print(f"Accuracy: {results['accuracy']}")
    print(f"Macro F1: {results['macro_f1']}")
    print(f"Per-class F1: {results['per_class_f1']}")
    print(f"Good-vs-Rest AUC: {results['good_vs_rest_auc']}")

    # Compare with individual models
    print("\n--- Individual model comparison ---")
    for name, model in ensemble.named_estimators_.items():
        model_results = evaluate(model, X_test, y_test)
        print(f"{name}: acc={model_results['accuracy']}, macro_f1={model_results['macro_f1']}")

    # Save ensemble
    joblib.dump(ensemble, MODELS_DIR / "v0.5_ensemble.joblib")

    # Save report
    report = {
        "model_version": MODEL_VERSION,
        "results": results,
        "fit_time": fit_time,
        "weights": dict(zip(["gradient_boosting", "random_forest", "logistic_regression"], ensemble.weights)),
        "feature_names": list(X_train.columns),
    }
    report_path = MODELS_DIR / "v0.5_ensemble_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nEnsemble saved to {MODELS_DIR / 'v0.5_ensemble.joblib'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
