"""
v0.4.0 — add CV-length normalization features to existing feature tables.

The v0.4.0 schema adds cv_word_count, skills_per_100_words, and
cv_length_bucket (see backend/app/ml/feature_extraction.py). All three
derive from cv_text + n_candidate_skills, which are already available, so
instead of re-running the expensive embedding-based extraction we join the
feature tables back to the raw source CSVs (by split_row) and compute only
the new columns with the SAME shared helper the serving path uses
(compute_cv_length_features). The 33 expensive columns are passed through
untouched, guaranteeing identical values.

This keeps train/serve consistency: the augmentation uses the same code
path as build_feature_vector, just factored out.

Usage (from backend/ with the venv active):
    python ../ml/preprocessing/augment_cv_length_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ml.feature_extraction import (  # noqa: E402
    FEATURE_NAMES,
    compute_cv_length_features,
)

PROCESSED = REPO_ROOT / "data" / "processed"
LENGTH_COLS = ["cv_word_count", "skills_per_100_words", "cv_length_bucket"]
META = {"label", "label_int", "split_row", "extraction_error"}


def augment_split(split: str) -> None:
    table_path = PROCESSED / f"{split}_features.csv"
    raw_path = REPO_ROOT / "data" / "raw" / f"{split}.csv"
    if not table_path.exists() or not raw_path.exists():
        sys.exit(f"missing {table_path} or {raw_path}")

    table = pd.read_csv(table_path)
    raw = pd.read_csv(raw_path, usecols=["resume_text"])

    if "cv_word_count" in table.columns:
        print(f"[{split}] already augmented ({len(table)} rows); skipping")
        return

    if not set(["split_row", "n_candidate_skills"]).issubset(table.columns):
        sys.exit(f"[{split}] feature table lacks split_row/n_candidate_skills")

    # sanity: the expensive columns must already match FEATURE_NAMES
    existing = [c for c in table.columns if c not in META and c not in LENGTH_COLS]
    if existing != [f for f in FEATURE_NAMES if f not in LENGTH_COLS]:
        sys.exit(f"[{split}] schema mismatch: {len(existing)} cols vs FEATURE_NAMES")

    # split_row indexes the original source CSV row (provenance guarantee
    # fixed in v0.2.1). Join on it to recover the raw CV text.
    texts = raw["resume_text"].iloc[table["split_row"].to_numpy()].to_numpy()
    n_skills = table["n_candidate_skills"].to_numpy()

    new_cols = [
        compute_cv_length_features(str(text), int(n))
        for text, n in zip(texts, n_skills, strict=True)
    ]
    new_frame = pd.DataFrame(new_cols, index=table.index)
    out = pd.concat([table, new_frame], axis=1)

    # verify column order matches FEATURE_NAMES exactly
    expected = FEATURE_NAMES
    got = [c for c in out.columns if c not in META]
    if got != expected:
        sys.exit(f"[{split}] post-augment schema mismatch:\n got: {got}\n exp: {expected}")

    out.to_csv(table_path, index=False)
    print(f"[{split}] wrote {table_path} ({len(out)} rows, {len(out.columns)} cols)")


def main() -> None:
    for split in ("train", "test"):
        augment_split(split)


if __name__ == "__main__":
    main()
