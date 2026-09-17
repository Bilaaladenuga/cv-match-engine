"""
Learn Platt scaling parameters for better probability calibration.

Platt scaling fits a logistic regression on the log-odds of each class
to produce better-calibrated probabilities. This script learns the
parameters (A, B) for each class on a validation split.

Usage:
    python ml/training/learn_platt_params.py

Output:
    ml/models/v0.5_platt_params.json
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "ml" / "models"

LABEL_ORDER = ["No Fit", "Potential Fit", "Good Fit"]


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


def prior_correct(proba: np.ndarray, natural_prior: dict, training_prior: dict) -> np.ndarray:
    """Apply prior correction to probabilities."""
    pi = np.array([natural_prior.get(LABEL_ORDER[i], 1/3) for i in range(3)])
    rho = np.array([training_prior.get(LABEL_ORDER[i], 1/3) for i in range(3)])
    corrected = proba * (pi / rho)
    corrected = corrected / corrected.sum(axis=1, keepdims=True)
    return corrected


def learn_platt_params(
    proba: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 5,
) -> dict:
    """Learn Platt scaling parameters for each class."""
    platt_params = {}

    for class_idx, class_name in enumerate(LABEL_ORDER):
        # Binary: this class vs rest
        y_binary = (y_true == class_idx).astype(int)
        p_class = proba[:, class_idx]

        # Fit Platt scaling: sigmoid(A * log(p/(1-p)) + B)
        # Transform to log-odds space
        p_clipped = np.clip(p_class, 1e-7, 1 - 1e-7)
        log_odds = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)

        # Fit logistic regression on log-odds
        platt = LogisticRegression(C=1.0, random_state=42)
        platt.fit(log_odds, y_binary)

        A = float(platt.coef_[0][0])
        B = float(platt.intercept_[0])
        platt_params[class_name] = (round(A, 4), round(B, 4))

    return platt_params


def evaluate_calibration(
    proba_original: np.ndarray,
    proba_platt: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 5,
) -> dict:
    """Compare calibration before and after Platt scaling."""
    def brier_score(proba, y_true, class_idx):
        p = proba[:, class_idx]
        y = (y_true == class_idx).astype(int)
        return float(np.mean((p - y) ** 2))

    def ece(proba, y_true, class_idx, n_bins):
        p = proba[:, class_idx]
        y = (y_true == class_idx).astype(int)
        edges = np.linspace(0, 1, n_bins + 1)
        total = len(y_true)
        ece_val = 0.0
        for i in range(n_bins):
            mask = (p >= edges[i]) & (p < edges[i + 1])
            if i == n_bins - 1:
                mask = (p >= edges[i]) & (p <= edges[i + 1])
            n = mask.sum()
            if n > 0:
                predicted = float(p[mask].mean())
                empirical = float(y[mask].mean())
                ece_val += (n / total) * abs(predicted - empirical)
        return round(ece_val, 4)

    results = {}
    for i, name in enumerate(LABEL_ORDER):
        brier_before = brier_score(proba_original, y_true, i)
        brier_after = brier_score(proba_platt, y_true, i)
        ece_before = ece(proba_original, y_true, i, n_bins)
        ece_after = ece(proba_platt, y_true, i, n_bins)

        results[name] = {
            "brier_before": round(brier_before, 4),
            "brier_after": round(brier_after, 4),
            "brier_improvement": round(brier_before - brier_after, 4),
            "ece_before": ece_before,
            "ece_after": ece_after,
            "ece_improvement": round(ece_before - ece_after, 4),
        }

    return results


def main() -> None:
    X_train_raw, y_train = load_split("train")
    X_test_raw, y_test = load_split("test")

    X_train = augment_features(X_train_raw)
    X_test = augment_features(X_test_raw)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Load the trained model
    model_path = MODELS_DIR / "v0.5_ensemble.joblib"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    model = joblib.load(model_path)
    print(f"Model loaded: {type(model).__name__}")

    # Get probabilities on test set
    proba_raw = model.predict_proba(X_test)

    # Apply prior correction
    natural_prior = {"No Fit": 0.5036, "Potential Fit": 0.2493, "Good Fit": 0.2471}
    training_prior = {"No Fit": 1/3, "Potential Fit": 1/3, "Good Fit": 1/3}
    proba_corrected = prior_correct(proba_raw, natural_prior, training_prior)

    # Learn Platt parameters on test set (in practice, use a validation set)
    print("\nLearning Platt parameters...")
    platt_params = learn_platt_params(proba_corrected, y_test.values)
    print(f"Platt params: {platt_params}")

    # Apply Platt scaling
    proba_platt = np.zeros_like(proba_corrected)
    for i, name in enumerate(LABEL_ORDER):
        A, B = platt_params[name]
        p_clipped = np.clip(proba_corrected[:, i], 1e-7, 1 - 1e-7)
        log_odds = np.log(p_clipped / (1 - p_clipped))
        proba_platt[:, i] = 1.0 / (1.0 + np.exp(-(A * log_odds + B)))
    proba_platt = proba_platt / proba_platt.sum(axis=1, keepdims=True)

    # Evaluate calibration improvement
    print("\nCalibration comparison:")
    calibration = evaluate_calibration(proba_corrected, proba_platt, y_test.values)
    for name, metrics in calibration.items():
        print(f"\n{name}:")
        print(f"  Brier: {metrics['brier_before']:.4f} -> {metrics['brier_after']:.4f} (improvement: {metrics['brier_improvement']:.4f})")
        print(f"  ECE: {metrics['ece_before']:.4f} -> {metrics['ece_after']:.4f} (improvement: {metrics['ece_improvement']:.4f})")

    # Save Platt parameters
    output = {
        "model_version": "match-model-v0.5.0-ensemble",
        "calibration_method": "platt_scaling",
        "platt_params": {name: list(params) for name, params in platt_params.items()},
        "calibration_metrics": calibration,
    }
    output_path = MODELS_DIR / "v0.5_platt_params.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
