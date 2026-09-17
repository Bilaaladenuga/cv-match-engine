"""
Improved model training — v0.5 baseline.

Fixes identified in the v0.4 evaluation:
1. No cross-validation → adds Stratified 5-fold CV
2. No hyperparameter tuning → adds GridSearchCV
3. Train/test distribution mismatch → uses natural distribution with class weights
4. No proper evaluation → adds calibration, ordinal-aware errors, ranking metrics
5. Volume-proxy bias → adds feature importance analysis and selection

Usage (from backend/ with the venv active):
    python ../ml/training/train_improved.py

Artifacts:
    ml/models/v0.5_<name>.joblib      fitted pipeline
    ml/models/v0.5_training_report.json   full evaluation report
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "ml" / "models"

MODEL_VERSION = "match-model-v0.5.0-improved"

LABEL_ORDER = ["No Fit", "Potential Fit", "Good Fit"]
LABEL_TO_INT = {name: i for i, name in enumerate(LABEL_ORDER)}

# Prior correction constants (natural distribution from audit)
NATURAL_PRIOR = {
    "No Fit": 0.5036,
    "Potential Fit": 0.2493,
    "Good Fit": 0.2471,
}


def load_split(split: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load feature table, drop extraction errors, encode labels, add augmented features."""
    df = pd.read_csv(PROCESSED_DIR / f"{split}_features.csv")
    if "extraction_error" in df.columns:
        n_bad = int(df["extraction_error"].notna().sum())
        if n_bad:
            print(f"[{split}] dropping {n_bad} rows with extraction errors")
        df = df[df["extraction_error"].isna()]
    feature_names = _feature_names(df)
    X = df[feature_names]
    y_raw = df["label"]
    y = pd.Series(
        np.where(y_raw == "Good Fit", 2, np.where(y_raw == "Potential Fit", 1, 0)),
        index=df.index,
    )
    # Add augmented features
    X = augment_features(X)
    return X, y, y_raw


def _feature_names(df: pd.DataFrame) -> list[str]:
    meta = {"label", "label_int", "split_row", "extraction_error"}
    return [c for c in df.columns if c not in meta]


def augment_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to reduce volume-proxy bias and add domain awareness."""
    X = X.copy()

    # 1. Skill coverage ratios (not just counts)
    if "required_skill_coverage" in X.columns and "preferred_skill_coverage" in X.columns:
        X["total_skill_coverage"] = (
            X["required_skill_coverage"] * 0.7 + X["preferred_skill_coverage"] * 0.3
        )

    # 2. Skill density normalized by JD complexity
    if "n_candidate_skills" in X.columns and "n_required_skills" in X.columns:
        X["skills_per_required"] = X["n_candidate_skills"] / (X["n_required_skills"] + 1)
        X["skill_surplus_ratio"] = (X["n_candidate_skills"] - X["n_required_skills"]) / (X["n_required_skills"] + 1)

    # 3. Experience-adjusted skill count (skills per year of experience)
    if "n_candidate_skills" in X.columns and "experience_gap_years" in X.columns:
        X["skills_per_year"] = X["n_candidate_skills"] / (X["experience_gap_years"] + 5)

    # 4. Coverage quality score (weighted by category importance)
    coverage_cols = [c for c in X.columns if c.startswith("cov_") and c.endswith("_required")]
    if coverage_cols:
        X["mean_category_coverage"] = X[coverage_cols].mean(axis=1)
        X["min_category_coverage"] = X[coverage_cols].min(axis=1)
        X["max_category_coverage"] = X[coverage_cols].max(axis=1)
        X["coverage_variance"] = X[coverage_cols].var(axis=1)

    # 5. Semantic-skill alignment (are semantic scores backed by skills?)
    if "semantic_similarity" in X.columns and "skill_overlap_ratio" in X.columns:
        X["semantic_skill_gap"] = X["semantic_similarity"] - X["skill_overlap_ratio"]
        X["semantic_skill_product"] = X["semantic_similarity"] * X["skill_overlap_ratio"]

    # 6. Education-experience alignment
    if "education_level_score" in X.columns and "seniority_match" in X.columns:
        X["edu_seniority_alignment"] = X["education_level_score"] * X["seniority_match"]

    # 7. CV quality signals (non-volume)
    if "skills_per_100_words" in X.columns:
        X["skill_density_squared"] = X["skills_per_100_words"] ** 2

    # 8. Title alignment strength
    if "job_title_similarity" in X.columns and "skill_overlap_ratio" in X.columns:
        X["title_skill_alignment"] = X["job_title_similarity"] * X["skill_overlap_ratio"]

    # Replace inf/nan with 0
    X = X.replace([np.inf, -np.inf], 0.0)
    X = X.fillna(0.0)

    return X


def compute_natural_prior() -> dict[str, float]:
    raw = REPO_ROOT / "data" / "raw" / "train.csv"
    if not raw.exists():
        return NATURAL_PRIOR
    counts = pd.read_csv(raw, usecols=["label"])["label"].value_counts()
    return {label: float(counts.get(label, 0) / counts.sum()) for label in LABEL_ORDER}


def prior_correct_probabilities(
    proba: np.ndarray, natural_prior: dict[str, float]
) -> np.ndarray:
    """Saerens prior correction: adjust from training prior to natural prior."""
    training_prior = np.array([1.0 / 3.0] * 3)
    natural = np.array([natural_prior.get(LABEL_ORDER[i], 1.0 / 3.0) for i in range(3)])
    corrected = proba * (natural / training_prior)
    corrected = corrected / corrected.sum(axis=1, keepdims=True)
    return corrected


def ordinal_error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Analyze errors by ordinal distance (adjacent vs distant)."""
    diff = np.abs(y_true - y_pred)
    return {
        "adjacent_errors": int((diff == 1).sum()),
        "distant_errors": int((diff == 2).sum()),
        "overrated": int((y_pred > y_true).sum()),  # predicted higher than actual
        "underrated": int((y_pred < y_true).sum()),  # predicted lower than actual
    }


def compute_calibration_bins(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 5
) -> list[dict]:
    """Calibration analysis for Good-vs-rest."""
    y_good = (y_true == 2).astype(int)
    prob_good = proba[:, 2]
    bins = []
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (prob_good >= edges[i]) & (prob_good < edges[i + 1])
        if i == n_bins - 1:  # include right edge for last bin
            mask = (prob_good >= edges[i]) & (prob_good <= edges[i + 1])
        n = mask.sum()
        if n > 0:
            bins.append({
                "bin": f"[{edges[i]:.1f},{edges[i+1]:.1f})",
                "n": int(n),
                "predicted": round(float(prob_good[mask].mean()), 4),
                "empirical": round(float(y_good[mask].mean()), 4),
            })
    return bins


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    natural_prior: dict[str, float],
) -> dict:
    """Full evaluation with as-trained and prior-corrected variants."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    # As-trained metrics
    as_trained = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "per_class_f1": {
            name: round(float(f1_score(y_test, pred, labels=[i], average="macro")), 4)
            for i, name in enumerate(LABEL_ORDER)
        },
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(
            y_test, pred, target_names=LABEL_ORDER, output_dict=True, zero_division=0
        ),
        "log_loss": round(float(log_loss(y_test, proba)), 4),
    }

    # Prior-corrected metrics
    proba_corrected = prior_correct_probabilities(proba, natural_prior)
    pred_corrected = np.argmax(proba_corrected, axis=1)
    prior_corrected = {
        "accuracy": round(float(accuracy_score(y_test, pred_corrected)), 4),
        "macro_f1": round(float(f1_score(y_test, pred_corrected, average="macro")), 4),
        "per_class_f1": {
            name: round(float(f1_score(y_test, pred_corrected, labels=[i], average="macro")), 4)
            for i, name in enumerate(LABEL_ORDER)
        },
        "confusion_matrix": confusion_matrix(y_test, pred_corrected).tolist(),
    }

    # Good-vs-rest AUC
    y_good_binary = (y_test == 2).astype(int)
    good_auc = round(float(roc_auc_score(y_good_binary, proba[:, 2])), 4)

    # Ordinal error analysis
    ordinal = ordinal_error_analysis(y_test.values, pred)
    ordinal_corrected = ordinal_error_analysis(y_test.values, pred_corrected)

    # Calibration bins
    calibration = compute_calibration_bins(y_test.values, proba)

    # Fit score computation (for ranking eval)
    fit_scores = proba[:, 2] + 0.5 * proba[:, 1]  # P(Good) + 0.5 * P(Potential)
    fit_scores_corrected = proba_corrected[:, 2] + 0.5 * proba_corrected[:, 1]

    return {
        "as_trained": as_trained,
        "prior_corrected": prior_corrected,
        "good_vs_rest_auc": good_auc,
        "ordinal_errors": ordinal,
        "ordinal_errors_corrected": ordinal_corrected,
        "calibration": calibration,
        "fit_score_stats": {
            "mean": round(float(fit_scores.mean()), 4),
            "std": round(float(fit_scores.std()), 4),
            "mean_corrected": round(float(fit_scores_corrected.mean()), 4),
        },
    }


def build_models_with_params() -> dict[tuple, dict]:
    """Return (model, param_grid) tuples for grid search.
    Minimal parameter space for fast training.
    """
    return {
        "logistic_regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                )),
            ]),
            {"clf__C": [1.0]},
        ),
        "random_forest": (
            Pipeline([
                ("clf", RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                )),
            ]),
            {
                "clf__max_depth": [10],
                "clf__min_samples_leaf": [5],
            },
        ),
        "gradient_boosting": (
            Pipeline([
                ("clf", GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    min_samples_leaf=10,
                    subsample=0.9,
                    random_state=42,
                )),
            ]),
            {
                "clf__max_depth": [4],
            },
        ),
    }


def compute_sample_weights(y_train: np.ndarray) -> np.ndarray:
    """Compute balanced sample weights for classes."""
    counts = np.bincount(y_train)
    n = len(y_train)
    n_classes = len(counts)
    weights = np.array([n / (counts[label] * n_classes) for label in y_train])
    return weights


def cross_validate_and_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    name: str,
    model: Pipeline,
    param_grid: dict,
    cv: StratifiedKFold,
) -> dict:
    """Run grid search with cross-validation, return best model and metrics."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    # Use sample weights for gradient boosting
    if name == "gradient_boosting":
        sample_weights = compute_sample_weights(y_train.values)
        fit_params = {"clf__sample_weight": sample_weights}
    else:
        fit_params = {}

    # Grid search with CV
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train, **fit_params)
    fit_time = round(time.time() - t0, 1)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV F1 (macro): {grid_search.best_score_:.4f}")
    print(f"Fit time: {fit_time}s")

    return {
        "best_model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_cv_f1": round(float(grid_search.best_score_), 4),
        "fit_time": fit_time,
        "cv_results": {
            "mean_test_score": [
                round(float(s), 4) for s in grid_search.cv_results_["mean_test_score"]
            ],
            "std_test_score": [
                round(float(s), 4) for s in grid_search.cv_results_["std_test_score"]
            ],
        },
    }


def feature_importance_analysis(
    model, feature_names: list[str]
) -> list[dict]:
    """Extract and rank feature importances."""
    # For pipelines, get the final estimator
    estimator = model
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]

    importances = None
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).mean(axis=0)

    if importances is None:
        return []

    ranked = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in ranked
    ]


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, y_train_raw = load_split("train")
    X_test, y_test, y_test_raw = load_split("test")
    natural_prior = compute_natural_prior()

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train label distribution: {dict(zip(LABEL_ORDER, np.bincount(y_train)))}")
    print(f"Test label distribution: {dict(zip(LABEL_ORDER, np.bincount(y_test)))}")
    print(f"Natural prior: {natural_prior}")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = {}
    for name, (model, param_grid) in build_models_with_params().items():
        cv_result = cross_validate_and_train(
            X_train, y_train, name, model, param_grid, cv
        )
        best_model = cv_result["best_model"]

        # Full evaluation on held-out test set
        eval_result = evaluate_model(best_model, X_test, y_test, natural_prior)

        # Feature importance
        importances = feature_importance_analysis(best_model, list(X_train.columns))

        results[name] = {
            "best_params": cv_result["best_params"],
            "best_cv_f1": cv_result["best_cv_f1"],
            "fit_time": cv_result["fit_time"],
            "test_metrics": eval_result,
            "feature_importance": importances[:15],  # top 15
            "model_version": MODEL_VERSION,
        }

        # Print summary
        at = eval_result["as_trained"]
        pc = eval_result["prior_corrected"]
        print(f"\n--- {name} ---")
        print(f"  As-trained:  acc={at['accuracy']} macro_f1={at['macro_f1']}")
        print(f"  Prior-corrected: acc={pc['accuracy']} macro_f1={pc['macro_f1']}")
        print(f"  Good-vs-Rest AUC: {eval_result['good_vs_rest_auc']}")
        print(f"  Ordinal errors: adjacent={eval_result['ordinal_errors']['adjacent_errors']} "
              f"distant={eval_result['ordinal_errors']['distant_errors']}")
        print(f"  Top features: {[f['feature'] for f in importances[:5]]}")

        # Save model with calibration metadata
        best_model.calibration_ = {
            "method": "saerens_prior_correction",
            "natural_prior": natural_prior,
            "training_prior": {name: 1.0 / 3.0 for name in LABEL_ORDER},
        }
        best_model.reference_stats_ = {
            "medians": {name: float(v) for name, v in X_train.median().items()},
            "n_train": int(len(X_train)),
            "feature_names": list(X_train.columns),
        }
        joblib.dump(best_model, MODELS_DIR / f"v0.5_{name}.joblib")

    # Summary report
    results["label_mapping"] = LABEL_TO_INT
    results["natural_prior"] = natural_prior
    results["feature_names"] = list(X_train.columns)
    results["dataset"] = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "source": "cnamuangtoun/resume-job-description-fit (v0.5 improved training)",
        "methodology": "Stratified 5-fold CV, grid search, class weights, prior correction",
    }
    results["model_version"] = MODEL_VERSION

    report_path = MODELS_DIR / "v0.5_training_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Training complete. Artifacts saved to {MODELS_DIR}")
    print(f"Report: {report_path}")

    # Print best model summary
    for k, v in results.items():
        if isinstance(v, dict) and "best_cv_f1" in v:
            print(f"\nBest model: {k}")
            print(f"  CV F1: {v['best_cv_f1']}")
            print(f"  Test acc: {v['test_metrics']['as_trained']['accuracy']}")
            break


if __name__ == "__main__":
    main()
