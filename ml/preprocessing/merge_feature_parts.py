"""
Merge incremental feature-extraction part files into the canonical tables.

Part files are produced by extending extraction with --skip-per-class
(see ml/features/extract_features.py). This script concatenates them,
deduplicates on (split, split_row), sorts by split_row, and writes the
canonical `<split>_features.csv` used by ml/training/train_baseline.py.

Usage (from backend/ with the venv active):
    python ../ml/preprocessing/merge_feature_parts.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def merge_split(split: str) -> None:
    canonical = PROCESSED_DIR / f"{split}_features.csv"
    parts: list[Path] = []
    if canonical.exists():
        parts.append(canonical)
    part_files = sorted(PROCESSED_DIR.glob(f"{split}_features_part*.csv"))
    if not part_files:
        print(f"[{split}] no part files found; nothing to do")
        return
    parts.extend(part_files)

    frames = [pd.read_csv(p) for p in parts]
    merged = pd.concat(frames, ignore_index=True)

    before = len(merged)
    merged = merged.drop_duplicates(subset=["split_row"], keep="first")
    merged = merged.sort_values("split_row").reset_index(drop=True)
    dropped = before - len(merged)

    merged.to_csv(canonical, index=False)
    for p in part_files:
        p.unlink(missing_ok=True)

    counts = merged["label"].value_counts().to_dict()
    errors = int(merged["extraction_error"].notna().sum()) if "extraction_error" in merged else 0
    print(f"[{split}] {len(parts)} files -> {canonical.name}: {len(merged)} rows "
          f"({dropped} duplicates dropped), {counts}, {errors} extraction errors")


def main() -> None:
    for split in ("train", "test"):
        merge_split(split)


if __name__ == "__main__":
    main()
