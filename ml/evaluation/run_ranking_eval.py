"""
Phase 13 — ranked-retrieval evaluation over the held-out test sample.

For each baseline model, rank candidates within job slates (JD groups,
the production ranking scenario) and within CV groups (one candidate, many
jobs - exploratory, see docs/ml-methodology.md 6.1), then report
Precision@K / Recall@K / NDCG@K against a seeded random-ordering baseline.

Scores use the same fit convention as the serving scorer
(backend/app/ml/model_scorer.py):

    fit_score = P(Good Fit) + 0.5 * P(Potential Fit)   -> 0..1

Artifacts written to ml/evaluation/:
    ranking_eval.json      metrics for every model x grouping
    ranking_eval.md        human-readable report (committable evidence)

Usage (from backend/ with the venv active):
    python ../ml/evaluation/run_ranking_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.ranking_metrics import (  # noqa: E402
    GAIN_MAP,
    aggregates_to_dict,
    evaluate_grouped_ranking,
)

MODELS = {
    "logistic_regression": REPO_ROOT / "ml" / "models" / "baseline_logistic_regression.joblib",
    "random_forest": REPO_ROOT / "ml" / "models" / "baseline_random_forest.joblib",
    "gradient_boosting": REPO_ROOT / "ml" / "models" / "baseline_gradient_boosting.joblib",
}

EVAL_DIR = REPO_ROOT / "ml" / "evaluation"
KS = (3, 5, 10)
MIN_GROUP = 4

LABEL_ORDER = ["No Fit", "Potential Fit", "Good Fit"]


def fit_score_from_proba(proba: np.ndarray, classes: np.ndarray) -> float:
    """P(Good) + 0.5*P(Potential), robust to any class ordering."""
    p = {int(c): float(v) for c, v in zip(classes, proba, strict=True)}
    return p.get(2, 0.0) + 0.5 * p.get(1, 0.0)


def load_test_frame() -> pd.DataFrame:
    """Feature table + CV/JD group keys aligned by original test.csv index."""
    feat = pd.read_csv(REPO_ROOT / "data" / "processed" / "test_features.csv")
    src = pd.read_csv(
        REPO_ROOT / "data" / "raw" / "test.csv",
        usecols=["resume_text", "job_description_text"],
    )
    import hashlib

    src["cv_hash"] = src["resume_text"].apply(lambda t: hashlib.md5(t.encode()).hexdigest()[:12])
    src["jd_hash"] = src["job_description_text"].apply(lambda t: hashlib.md5(t.encode()).hexdigest()[:12])

    rows = src.iloc[feat["split_row"].to_numpy()].reset_index(drop=True)
    feat["cv_group"] = rows["cv_hash"].to_numpy()
    feat["jd_group"] = rows["jd_hash"].to_numpy()
    return feat


def score_all_models(feat: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return {model_name: array of fit scores} for every feature row."""
    feature_cols = [c for c in feat.columns if c not in PROTECTED_COLS]
    feature_matrix = feat[feature_cols].to_numpy(dtype=float)
    out = {}
    for name, path in MODELS.items():
        if not path.exists():
            print(f"  ! missing artifact, skipping: {name}")
            continue
        model = joblib.load(path)
        proba = model.predict_proba(feature_matrix)
        out[name] = np.array(
            [fit_score_from_proba(p, model.classes_) for p in proba]
        )
    return out


PROTECTED_COLS = {"label", "label_int", "split_row", "extraction_error", "gain", "cv_group", "jd_group"}


def main() -> None:
    feat = load_test_frame()
    feat["gain"] = feat["label"].map(GAIN_MAP)
    if feat["gain"].isna().any():
        sys.exit(f"Unmapped labels: {feat.loc[feat['gain'].isna(), 'label'].unique()}")

    scores = score_all_models(feat)
    if not scores:
        sys.exit("No model artifacts found; run ml/training/train_baseline.py first")

    results: dict = {
        "meta": {
            "n_rows": int(len(feat)),
            "label_counts": feat["label"].value_counts().to_dict(),
            "min_group_size": MIN_GROUP,
            "k_values": list(KS),
            "score_convention": "P(Good) + 0.5*P(Potential)",
        },
        "jd_groups": {},
        "cv_groups": {},
    }

    for name, s in scores.items():
        feat[f"score_{name}"] = s
        print(f"== {name} ==")
        results["jd_groups"][name] = aggregates_to_dict(
            evaluate_grouped_ranking(feat, f"score_{name}", "jd_group", k_values=KS, min_group_size=MIN_GROUP)
        )
        results["cv_groups"][name] = aggregates_to_dict(
            evaluate_grouped_ranking(feat, f"score_{name}", "cv_group", k_values=KS, min_group_size=MIN_GROUP)
        )
        jd = results["jd_groups"][name]
        print(
            f"  JD groups: NDCG@5={jd['ndcg_at_k']['5']} P@5={jd['precision_at_k']['5']} "
            f"(random NDCG@5={jd['random_baseline_ndcg']['5']}, groups={jd['n_groups_used']['5']})"
        )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "ranking_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {EVAL_DIR / 'ranking_eval.json'}")
    _write_markdown(results)


def _write_markdown(results: dict) -> None:
    meta = results["meta"]
    lines = [
        "# Ranking Evaluation (Phase 13)",
        "",
        f"Held-out test sample: {meta['n_rows']} rows "
        f"({meta['label_counts'].get('No Fit', 0)} No / "
        f"{meta['label_counts'].get('Potential Fit', 0)} Potential / "
        f"{meta['label_counts'].get('Good Fit', 0)} Good). "
        f"Groups with <{meta['min_group_size']} rows are excluded.",
        "",
        "Score convention (identical to serving): `P(Good Fit) + 0.5 * P(Potential Fit)`.",
        "Relevance gains: No Fit = 0, Potential Fit = 1, Good Fit = 2 (graded).",
        "Random baseline = seeded within-group score shuffle (50 permutations).",
        "",
        "## JD groups - recruiter slates (production ranking scenario)",
        "",
    ]

    def table(group_key: str, all_results: dict) -> list[str]:
        rows = [
            "| Model | P@3 | rand P@3 | R@3 | N@3 | rand N@3 | P@5 | rand P@5 | R@5 | N@5 | rand N@5 | groups |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, agg in all_results[group_key].items():
            p, r, n = agg["precision_at_k"], agg["recall_at_k"], agg["ndcg_at_k"]
            rbp, rbn = agg["random_baseline_precision_at_k"], agg["random_baseline_ndcg"]

            def cell(k: str, d: dict[str, float], metric_agg: dict) -> str:
                # A metric cell is only meaningful when at least one group of
                # sufficient size contributed at that K.
                if metric_agg["n_groups_used"].get(k, 0) == 0:
                    return "n/a"
                return f"{d[k]}"

            def groups_at(k: str, metric_agg: dict) -> int:
                return metric_agg["n_groups_used"].get(k, 0)

            rows.append(
                f"| {name} | {cell('3', p, agg)} | {cell('3', rbp, agg)} | {cell('3', r, agg)} | {cell('3', n, agg)} | {cell('3', rbn, agg)} "
                f"| {cell('5', p, agg)} | {cell('5', rbp, agg)} | {cell('5', r, agg)} | {cell('5', n, agg)} | {cell('5', rbn, agg)} "
                f"| {groups_at('5', agg)} |"
            )
        return rows

    lines += table("jd_groups", results)
    lines += [
        "",
        "### Interpretation (JD slates, full test set)",
        "",
        "- The full 1,759-row set is imbalanced (857 No / 444 Potential / 458",
        "  Good), so relevant candidates are the MINORITY in a slate. This is",
        "  the honest production distribution - the earlier 300-row balanced",
        "  sample was an easier, non-representative regime (NDCG@5 ~0.94 there",
        "  vs ~0.71-0.73 here).",
        "- NDCG@5 lift over random: +0.05 to +0.10 depending on model - the",
        "  models DO put the strongest candidates (Good Fit, gain 2) near the",
        "  top of a slate.",
        "- P@5 shows NO lift over random (~0.50 both): the models surface",
        "  Good fits but cannot separate Potential Fit from No Fit when the",
        "  relevant class is the minority. Binary shortlist precision is the",
        "  weak spot - consistent with Potential Fit being the weakest class",
        "  in classification too.",
        "- Practical read: the ranking is useful for surfacing excellent",
        "  candidates, not for binary go/no-go shortlisting.",
        "",
        "## CV groups - one candidate, many jobs (exploratory)",
        "",
        "The public dataset reuses CV bodies across the upstream train/test",
        "boundary, so per-CV aggregates may be optimistic. See",
        "docs/ml-methodology.md section 6.1.",
        "",
    ]
    lines += table("cv_groups", results)
    lines += [
        "",
        "### Interpretation (CV groups)",
        "",
        "- 177 CV groups with >=4 applications (1,224 rows) - statistically",
        "  meaningful, unlike the earlier 300-row sample.",
        "- There is NO lift over random on any metric for any model: ordering",
        "  a candidate's job fits is currently at chance level. The features",
        "  aggregate the whole CV, so they capture what the candidate IS but",
        "  barely vary across jobs - a listwise comparison problem our",
        "  pointwise model cannot see. Likely fixes: per-pair interaction",
        "  features or a pairwise/listwise objective.",
        "",
    ]

    (EVAL_DIR / "ranking_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {EVAL_DIR / 'ranking_eval.md'}")


if __name__ == "__main__":
    main()
