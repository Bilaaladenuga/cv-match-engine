"""
Phase 12 — Dataset preprocessing.

Reads the raw Hugging Face `resume-job-description-fit` CSVs
(data/raw/train.csv, data/raw/test.csv), cleans them, validates the label
set, and writes the processed training table to data/processed/.

Run from backend/ with the venv active:
    python ../ml/preprocessing/build_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

EXPECTED_LABELS = {"Good Fit", "Potential Fit", "No Fit"}
LABEL_TO_INT = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def load_raw(split: str) -> pd.DataFrame:
    path = RAW_DIR / f"{split}.csv"
    if not path.exists():
        sys.exit(
            f"Missing {path}. Download the dataset first:\n"
            "  curl -L -o data/raw/train.csv "
            "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/main/train.csv\n"
            "  curl -L -o data/raw/test.csv "
            "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/main/test.csv"
        )
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and normalize text columns."""
    df = df.copy()

    required = {"resume_text", "job_description_text", "label"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Unexpected columns; missing {missing}. Got: {list(df.columns)}")

    before = len(df)
    df = df.drop_duplicates(subset=["resume_text", "job_description_text"])
    deduped = before - len(df)

    df = df.dropna(subset=["resume_text", "job_description_text", "label"])

    # Whitespace normalization (no other text mutation: the engines expect
    # realistic raw text, same as production input).
    for col in ("resume_text", "job_description_text"):
        df[col] = df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # Length guard: pathological rows would break parsing/embedding budgets.
    df = df[df["resume_text"].str.len().between(100, 30_000)]
    df = df[df["job_description_text"].str.len().between(100, 20_000)]

    unexpected = set(df["label"].unique()) - EXPECTED_LABELS
    if unexpected:
        sys.exit(f"Unexpected labels {unexpected}; expected subset of {EXPECTED_LABELS}")

    df = df[df["label"].isin(EXPECTED_LABELS)]
    df["label_int"] = df["label"].map(LABEL_TO_INT).astype(int)
    df = df.reset_index(drop=True)

    stats = {
        "rows_raw": before,
        "rows_deduped": int(deduped),
        "rows_final": len(df),
        "label_counts": df["label"].value_counts().to_dict(),
    }
    return df, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    profile: dict = {"dataset": "cnamuangtoun/resume-job-description-fit"}

    for split in args.splits:
        raw = load_raw(split)
        cleaned, stats = clean(raw)
        out_path = PROCESSED_DIR / f"{split}.parquet"
        try:
            cleaned.to_parquet(out_path, index=False)
        except Exception:
            out_path = PROCESSED_DIR / f"{split}.csv"
            cleaned.to_csv(out_path, index=False)
        profile[split] = {**stats, "path": str(out_path)}
        print(f"[{split}] raw={stats['rows_raw']} final={stats['rows_final']} -> {out_path}")
        print(f"        labels: {stats['label_counts']}")

    with open(PROCESSED_DIR / "dataset_profile.json", "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)
    print(f"Wrote {PROCESSED_DIR / 'dataset_profile.json'}")


if __name__ == "__main__":
    main()
