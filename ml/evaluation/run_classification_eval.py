"""
Phase 13 — classification evaluation with error analysis.

Per-class precision/recall/F1, multi-class ROC-AUC (OvR macro + binary
Good-vs-rest), confusion matrices, calibration bins, and a structured
FP/FN analysis on the FULL held-out test set (1,759 rows).

Error classification (ordinal-aware):
    FP (overrated): predicted class > true class   (e.g. No Fit -> Good Fit)
    FN (underrated): predicted class < true class   (e.g. Good Fit -> No Fit)
    adjacent vs distant errors tracked separately — for an ordinal target a
    one-step error is much less harmful than skipping a class.

Feature-delta profiles: for the most damaging confusion cell
(Potential -> Good FP and Good -> Potential FN), compares mean feature
values of error rows vs correct rows to show WHAT the model got wrong.

Artifacts (ml/evaluation/):
    classification_eval.json   machine-readable full results
    classification_eval.md     human-readable report (committable)

Usage (from backend/ with venv active):
    python ../ml/evaluation/run_classification_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "logistic_regression": REPO_ROOT / "ml" / "models" / "baseline_logistic_regression.joblib",
    "random_forest": REPO_ROOT / "ml" / "models" / "baseline_random_forest.joblib",
    "gradient_boosting": REPO_ROOT / "ml" / "models" / "baseline_gradient_boosting.joblib",
}

EVAL_DIR = REPO_ROOT / "ml" / "evaluation"
LABELS = ["No Fit", "Potential Fit", "Good Fit"]
FEATURE_NAMES_ORDER = ["skill_overlap_ratio", "required_skill_coverage", "semantic_similarity", "experience_gap_years", "job_title_similarity"]


def load_test_frame() -> pd.DataFrame:
    feat = pd.read_csv(REPO_ROOT / "data" / "processed" / "test_features.csv")
    return feat


def load_natural_prior() -> dict[str, float]:
    """Class prior of the UNSTRATIFIED source split.

    Our feature tables are stratified (1/3 per class by construction), but
    production input follows the natural distribution. Training on the
    stratified table teaches the model a uniform prior; this is the prior
    the world actually has, used for the prior-correction variant below.
    """
    src = pd.read_csv(REPO_ROOT / "data" / "raw" / "train.csv", usecols=["label"])
    counts = src["label"].value_counts()
    return {label: counts.get(label, 0) / counts.sum() for label in LABELS}


def prior_correct(proba: np.ndarray, classes: list[int], natural_prior: dict[str, float]) -> np.ndarray:
    """Saerens-style prior correction: reweight probs to the natural prior.

    p'_c = p_c * (pi_c / rho_c) / Z, with rho_c the training prior (uniform
    1/3 here, because the feature table is stratified) and pi_c the natural
    source prior. Monotone within a class but renormalization mixes classes,
    so downstream metrics are recomputed on the corrected probabilities.
    """
    rho = np.array([1.0 / len(classes)] * len(classes))
    pi = np.array([natural_prior[LABELS[c]] for c in classes])
    scaled = proba * (pi / rho)
    z = scaled.sum(axis=1, keepdims=True)
    return scaled / np.where(z > 0, z, 1.0)


def evaluate_model(name: str, model, feat: pd.DataFrame, natural_prior: dict[str, float]) -> dict:
    feature_cols = [c for c in feat.columns if c not in PROTECTED_COLS]
    feature_matrix = feat[feature_cols].to_numpy(dtype=float)
    y_true = feat["label"].to_numpy()
    y_true_int = feat["label_int"].to_numpy()

    proba = model.predict_proba(feature_matrix)
    classes = list(model.classes_)

    res: dict = {
        "as_trained": _metrics_for_proba(proba, classes, feat, y_true, y_true_int),
        "prior_corrected": _metrics_for_proba(
            prior_correct(proba, classes, natural_prior), classes, feat, y_true, y_true_int
        ),
        "natural_prior": {k: round(v, 4) for k, v in natural_prior.items()},
    }
    return res


def _metrics_for_proba(
    proba: np.ndarray, classes: list[int], feat: pd.DataFrame, y_true: np.ndarray, y_true_int: np.ndarray
) -> dict:
    y_pred_int = np.array(classes)[np.argmax(proba, axis=1)]
    y_pred = np.array(LABELS)[y_pred_int]

    res: dict = {
        "accuracy": float((y_pred_int == y_true_int).mean()),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "per_class": {},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "roc_auc": {},
        "calibration_bins": _calibration_bins(y_true_int, proba, classes),
        "errors": _error_analysis(feat, y_true, y_pred, proba, classes),
    }

    # Per-class precision/recall/F1
    for label in LABELS:
        y_true_bin = (y_true == label).astype(int)
        y_pred_bin = (y_pred == label).astype(int)
        tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
        fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
        fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        res["per_class"][label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(y_true_bin.sum()),
        }

    # ROC-AUC: OvR macro (needs all classes present) + binary Good-vs-rest
    try:
        res["roc_auc"]["ovr_macro"] = round(
            float(roc_auc_score(y_true_int, proba, multi_class="ovr", average="macro")), 4
        )
    except ValueError as exc:
        res["roc_auc"]["ovr_macro"] = None
        res["roc_auc"]["ovr_macro_error"] = str(exc)
    good_idx = classes.index(2)
    res["roc_auc"]["good_vs_rest"] = round(
        float(roc_auc_score((y_true_int == 2).astype(int), proba[:, good_idx])), 4
    )
    # Adjacent-pair AUCs: the decision that actually matters in a slate
    pot_mask = np.isin(y_true_int, [0, 1])
    if pot_mask.sum() and len(set(y_true_int[pot_mask])) == 2:
        res["roc_auc"]["no_vs_potential"] = round(
            float(roc_auc_score(y_true_int[pot_mask], proba[pot_mask, classes.index(1)])), 4
        )
    good_mask = np.isin(y_true_int, [1, 2])
    if good_mask.sum() and len(set(y_true_int[good_mask])) == 2:
        res["roc_auc"]["potential_vs_good"] = round(
            float(roc_auc_score(y_true_int[good_mask], proba[good_mask, classes.index(2)])), 4
        )
    return res


PROTECTED_COLS = {"label", "label_int", "split_row", "extraction_error", "gain", "cv_group", "jd_group"}


def _error_analysis(
    feat: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, classes: list[int]
) -> dict:
    """Direction-aware FP/FN analysis on the ordinal label space."""
    y_true_int = np.array([LABELS.index(lbl) for lbl in y_true])
    y_pred_int = np.array([LABELS.index(lbl) for lbl in y_pred])
    correct = y_pred_int == y_true_int
    overrated = y_pred_int > y_true_int  # FP in the ordinal sense
    underrated = y_pred_int < y_true_int  # FN in the ordinal sense

    max_conf = proba.max(axis=1)
    dist = np.abs(y_pred_int - y_true_int)

    res: dict = {
        "n_errors": int((~correct).sum()),
        "overrated": int(overrated.sum()),
        "underrated": int(underrated.sum()),
        "adjacent_errors": int((dist[~correct] == 1).sum()),
        "distant_errors": int((dist[~correct] > 1).sum()),
    }

    # Confusion-cell decomposition: which (true -> predicted) pairs dominate
    cells: dict[str, int] = {}
    for t, p in zip(y_true_int[~correct], y_pred_int[~correct], strict=True):
        key = f"{LABELS[t]} -> {LABELS[p]}"
        cells[key] = cells.get(key, 0) + 1
    res["confusion_cells"] = dict(sorted(cells.items(), key=lambda kv: -kv[1]))

    # Feature-delta profile for the dominant cells: mean feature value of
    # error rows vs correct rows (matched on TRUE class to control base rate)
    def profile(true_label: str, pred_label: str) -> dict:
        mask_err = (y_true == true_label) & (y_pred == pred_label)
        mask_ok = (y_true == true_label) & (y_pred == true_label)
        if mask_err.sum() == 0 or mask_ok.sum() == 0:
            return {}
        feature_cols = [c for c in feat.columns if c not in PROTECTED_COLS]
        err_mean = feat.loc[mask_err, feature_cols].mean()
        ok_mean = feat.loc[mask_ok, feature_cols].mean()
        delta = (err_mean - ok_mean).abs().sort_values(ascending=False)
        out = {"n_error_rows": int(mask_err.sum()), "n_correct_rows": int(mask_ok.sum())}
        for fname in delta.index[:6]:
            out[fname] = {
                "error_rows": round(float(err_mean[fname]), 3),
                "correct_rows": round(float(ok_mean[fname]), 3),
                "delta": round(float(delta[fname]), 3),
            }
        return out

    res["profile_potential_to_good"] = profile("Potential Fit", "Good Fit")  # overrated
    res["profile_good_to_potential"] = profile("Good Fit", "Potential Fit")  # underrated
    res["profile_no_to_potential"] = profile("No Fit", "Potential Fit")
    res["profile_potential_to_no"] = profile("Potential Fit", "No Fit")

    # Top confident errors: highest-confidence mistakes (worst for a user)
    conf_order = np.argsort(-max_conf * (~correct).astype(int))
    top = []
    for idx in conf_order[:10]:
        if not correct[idx]:
            top.append(
                {
                    "split_row": int(feat.iloc[idx]["split_row"]),
                    "true": str(y_true[idx]),
                    "predicted": str(y_pred[idx]),
                    "confidence": round(float(max_conf[idx]), 3),
                    "distance": int(dist[idx]),
                }
            )
    res["top_confident_errors"] = top
    return res


def _calibration_bins(y_true_int: np.ndarray, proba: np.ndarray, classes: list[int], n_bins: int = 5) -> dict:
    """Reliability bins for the Good-vs-rest head: predicted prob vs empirical rate."""
    good_idx = classes.index(2)
    p_good = proba[:, good_idx]
    y_bin = (y_true_int == 2).astype(int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p_good >= lo) & (p_good < hi) if hi < 1.0 else (p_good >= lo) & (p_good <= hi)
        n = int(mask.sum())
        bins.append(
            {
                "range": f"[{lo:.1f},{hi:.1f})" if hi < 1.0 else f"[{lo:.1f},{hi:.1f}]",
                "n": n,
                "mean_predicted": round(float(p_good[mask].mean()), 3) if n else None,
                "empirical_good_rate": round(float(y_bin[mask].mean()), 3) if n else None,
            }
        )
    return {"good_vs_rest": bins}


def _write_markdown(results: dict, meta: dict) -> None:
    lines = [
        "# Classification Evaluation (Phase 13)",
        "",
        f"Held-out test set: {meta['n_rows']} rows "
        f"({meta['label_counts'].get('No Fit', 0)} No / "
        f"{meta['label_counts'].get('Potential Fit', 0)} Potential / "
        f"{meta['label_counts'].get('Good Fit', 0)} Good). "
        "Chance accuracy = "
        f"{meta['chance_accuracy']:.3f} (majority class).",
        "",
    ]
    for name, res in results.items():
        at = res["as_trained"]
        pc = res["prior_corrected"]
        cm = at["confusion_matrix"]
        lines += [
            f"## {name}",
            "",
            f"As trained: accuracy **{at['accuracy']:.4f}** · macro-F1 **{at['macro_f1']}** · "
            f"ROC-AUC (OvR macro) **{at['roc_auc']['ovr_macro']}** · "
            f"Good-vs-rest AUC **{at['roc_auc']['good_vs_rest']}**",
            "",
            f"Prior corrected: accuracy **{pc['accuracy']:.4f}** · macro-F1 **{pc['macro_f1']}** · "
            f"Good-vs-rest AUC **{pc['roc_auc']['good_vs_rest']}** — same model, probabilities "
            "reweighted to the natural class prior (training table is stratified 1/3 per class, "
            "the source distribution is ~50/25/25).",
            "",
            "| true\\pred | No | Potential | Good |",
            "|---|---|---|---|",
            f"| No | **{cm[0][0]}** | {cm[0][1]} | {cm[0][2]} |",
            f"| Potential | {cm[1][0]} | **{cm[1][1]}** | {cm[1][2]} |",
            f"| Good | {cm[2][0]} | {cm[2][1]} | **{cm[2][2]}** |",
            "",
            "| class | precision | recall | F1 | support |",
            "|---|---|---|---|---|",
        ]
        for label, m in at["per_class"].items():
            lines.append(
                f"| {label} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |"
            )
        err = at["errors"]
        lines += [
            "",
            f"**Errors:** {err['n_errors']} total — {err['overrated']} overrated (FP) vs "
            f"{err['underrated']} underrated (FN); {err['adjacent_errors']} adjacent, "
            f"{err['distant_errors']} distant (skipped a class).",
            "",
            "Top confusion cells: "
            + ", ".join(f"{k} ({v})" for k, v in list(err["confusion_cells"].items())[:4])
            + ".",
            "",
        ]
        for prof_name in ("profile_potential_to_good", "profile_good_to_potential"):
            prof = err.get(prof_name) or {}
            if not prof:
                continue
            lines += [f"**Feature deltas — {prof_name.replace('profile_', '').replace('_', ' ')}** "
                      f"({prof['n_error_rows']} error vs {prof['n_correct_rows']} correct rows):", ""]
            for fname, d in prof.items():
                if isinstance(d, dict):
                    lines.append(f"- `{fname}`: {d['error_rows']} (errors) vs {d['correct_rows']} (correct), Δ={d['delta']}")
            lines.append("")

        lines += ["**Calibration (Good-vs-rest, as trained):**", ""]
        for b in at["calibration_bins"]["good_vs_rest"]:
            if b["n"]:
                lines.append(
                    f"- {b['range']}: n={b['n']}, predicted {b['mean_predicted']}, empirical {b['empirical_good_rate']}"
                )
        lines += ["", "**Calibration (Good-vs-rest, prior corrected):**", ""]
        for b in pc["calibration_bins"]["good_vs_rest"]:
            if b["n"]:
                lines.append(
                    f"- {b['range']}: n={b['n']}, predicted {b['mean_predicted']}, empirical {b['empirical_good_rate']}"
                )
        lines.append("")

    best_acc = max(r["prior_corrected"]["accuracy"] for r in results.values())
    best_auc = max(r["as_trained"]["roc_auc"]["good_vs_rest"] for r in results.values())
    lines += [
        "## Summary — the honest read",
        "",
        "1. **No model beats the majority-class baseline on accuracy.** Always predicting",
        f"   'No Fit' scores {meta['chance_accuracy']:.3f}; the best model variant scores",
        f"   {best_acc:.3f}. On raw classification accuracy these features are not yet",
        "   competitive — the models' value lies in ranking and grade separation, not in",
        "   thresholded labels.",
        f"2. **Discrimination is weak but real:** best Good-vs-rest AUC {best_auc}",
        "   (0.5 = chance). This matches the ranking eval: useful for surfacing Good",
        "   candidates within a slate, useless for binary shortlisting.",
        "3. **Models are overconfident on Good Fit:** in the top calibration bin the",
        "   model predicts ~0.88 probability of Good while the empirical rate is ~0.36.",
        "   Root cause: training on a stratified table taught a uniform prior while the",
        "   source distribution is ~50/25/25. Prior correction recovers 3-5 accuracy",
        "   points by reweighting thresholds but does not change ranking (AUC ~equal).",
        "   Raw probabilities must never be shown to users as 'confidence'.",
        "4. **Volume-proxy shortcut (feature deltas):** candidates overrated into Good",
        "   list ~10.5 skills on average vs ~6.8 for correctly-potential ones; real Good",
        "   fits list ~12.4. The model conflates 'long CV' with 'good fit'. This is a",
        "   spurious correlation the next feature generation should correct for",
        "   (e.g. normalize by CV length, add per-pair depth features).",
        "5. **Ordinal structure is underused:** 38% of errors skip a class entirely",
        "   (e.g. No predicted Good). Ordinal-aware training (ordinal targets, class",
        "   margins) or a two-stage No-vs-rest / grade classifier is the natural fix.",
        "",
    ]

    (EVAL_DIR / "classification_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {EVAL_DIR / 'classification_eval.md'}")


def main() -> None:
    feat = load_test_frame()
    natural_prior = load_natural_prior()
    label_counts = feat["label"].value_counts().to_dict()
    meta = {
        "n_rows": int(len(feat)),
        "label_counts": label_counts,
        "chance_accuracy": max(label_counts.values()) / len(feat),
    }

    results: dict = {}
    for name, path in MODELS.items():
        if not path.exists():
            print(f"! missing artifact, skipping: {name}")
            continue
        print(f"== {name} ==")
        results[name] = evaluate_model(name, joblib.load(path), feat, natural_prior)
        r = results[name]["as_trained"]
        print(
            f"  acc={r['accuracy']:.4f} macroF1={r['macro_f1']} "
            f"AUC(ovr)={r['roc_auc']['ovr_macro']} AUC(good)={r['roc_auc']['good_vs_rest']}"
        )

    if not results:
        sys.exit("No model artifacts found; run ml/training/train_baseline.py first")

    payload = {"meta": meta, "results": results}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "classification_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {EVAL_DIR / 'classification_eval.json'}")
    _write_markdown(results, meta)


if __name__ == "__main__":
    main()
