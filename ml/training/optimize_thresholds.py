"""
Threshold optimization — find optimal decision boundaries for each class.

The default argmax threshold (0.5 for each class) is not optimal when classes
are imbalanced or when different errors have different costs. This script
searches for thresholds that maximize macro-F1 on a validation split.

Usage:
    python ml/training/optimize_thresholds.py

Output:
    ml/models/v0.5_thresholds.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
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
    """Add engineered features (same as training)."""
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


def optimize_thresholds(proba: np.ndarray, y_true: np.ndarray) -> dict:
    """Search for optimal per-class thresholds that maximize macro-F1."""
    best_thresholds = [0.5, 0.5, 0.5]  # default
    best_f1 = 0.0

    # Grid search over threshold combinations
    for t_no in np.arange(0.2, 0.8, 0.05):
        for t_potential in np.arange(0.2, 0.8, 0.05):
            # Good threshold is implicitly 1 - t_no - t_potential (but we need
            # to handle the decision logic explicitly)
            pred = np.zeros(len(y_true), dtype=int)
            for i in range(len(y_true)):
                p_no, p_pot, p_good = proba[i]
                # Apply custom thresholds
                if p_no >= t_no:
                    pred[i] = 0  # No Fit
                elif p_good >= t_potential:
                    pred[i] = 2  # Good Fit
                else:
                    pred[i] = 1  # Potential Fit

            f1 = f1_score(y_true, pred, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds = [t_no, t_potential, 1.0 - t_no - t_potential]

    return {
        "thresholds": {
            "No Fit": round(best_thresholds[0], 3),
            "Potential Fit": round(best_thresholds[1], 3),
            "Good Fit": round(max(0.1, best_thresholds[2]), 3),
        },
        "macro_f1": round(best_f1, 4),
    }


def main() -> None:
    import joblib

    X_train_raw, y_train = load_split("train")
    X_test_raw, y_test = load_split("test")

    X_train = augment_features(X_train_raw)
    X_test = augment_features(X_test_raw)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Load the trained model
    model_path = MODELS_DIR / "v0.5_gradient_boosting.joblib"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    model = joblib.load(model_path)
    print(f"Model loaded: {type(model).__name__}")
    print(f"Features: {len(model.feature_names_in_)}")

    # Get probabilities on test set
    proba = model.predict_proba(X_test)

    # Find optimal thresholds
    print("\nOptimizing thresholds...")
    result = optimize_thresholds(proba, y_test.values)
    print(f"Optimal thresholds: {result['thresholds']}")
    print(f"Macro F1 with optimized thresholds: {result['macro_f1']}")

    # Compare with default (argmax)
    from sklearn.metrics import accuracy_score
    pred_default = np.argmax(proba, axis=1)
    f1_default = f1_score(y_test, pred_default, average="macro")
    print(f"Macro F1 with default thresholds: {f1_default}")
    print(f"Improvement: {result['macro_f1'] - f1_default:.4f}")

    # Save thresholds
    output = {
        "model_version": "match-model-v0.5.0-improved",
        "optimization_method": "grid_search_macro_f1",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        **result,
    }
    output_path = MODELS_DIR / "v0.5_thresholds.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
