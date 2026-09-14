"""
Phase 14 — global feature importance via permutation on the fit score.

Complements the per-match local explainer (backend/app/ml/explainer.py):
local contributions say WHY this candidate got this score; permutation
importance says WHICH features the model relies on across the whole test
set. The two views can legitimately disagree locally (interactions), but
large local contributions on globally-unimportant features are a flag.

Importance = mean drop in the batch fit score (P(Good) + 0.5*P(Potential))
when one feature's column is randomly permuted, averaged over n repeats
with a fixed seed. Reported alongside training medians so the global view
stays interpretable.

Writes ml/evaluation/perm_importance.json (committable evidence).

Usage (from backend/ with the venv active):
    python ../ml/evaluation/perm_importance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "ml" / "models" / "baseline_gradient_boosting.joblib"
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "test_features.csv"
OUT_PATH = REPO_ROOT / "ml" / "evaluation" / "perm_importance.json"

N_REPEATS = 10
SEED = 13


def fit_scores(model, features: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(features)
    classes = list(model.classes_)
    p = {int(c): proba[:, i] for i, c in enumerate(classes)}
    return p.get(2, 0.0) + 0.5 * p.get(1, 0.0)


def main() -> None:
    if not MODEL_PATH.exists():
        sys.exit("Model artifact missing; run ml/training/train_baseline.py first")
    model = joblib.load(MODEL_PATH)
    feature_order = list(model.feature_names_in_)
    feat = pd.read_csv(FEATURES_PATH)
    features = feat[feature_order].astype(float)

    baseline = fit_scores(model, features)
    rng = np.random.default_rng(SEED)

    records = []
    for name in feature_order:
        drops = []
        for _ in range(N_REPEATS):
            perturbed = features.copy()
            perturbed[name] = rng.permutation(perturbed[name].to_numpy())
            drops.append(float((baseline - fit_scores(model, perturbed)).mean()))
        records.append(
            {
                "feature": name,
                "importance": round(float(np.mean(drops)), 5),
                "std": round(float(np.std(drops)), 5),
                "training_median": round(float(features[name].median()), 4),
            }
        )

    records.sort(key=lambda r: -r["importance"])
    payload = {
        "meta": {
            "model_version": getattr(model, "model_version", "unknown"),
            "n_rows": int(len(features)),
            "n_repeats": N_REPEATS,
            "seed": SEED,
            "metric": "mean drop in fit score P(Good)+0.5*P(Potential) under column permutation",
        },
        "importances": records,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{'feature':<32} {'importance':>10} {'std':>8}  median")
    for r in records[:15]:
        print(f"{r['feature']:<32} {r['importance']:>10.5f} {r['std']:>8.5f}  {r['training_median']}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
