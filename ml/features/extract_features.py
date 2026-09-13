"""
Phase 12 — Batch feature extraction.

Runs backend/app/ml/feature_extraction.extract_features over the processed
dataset rows and writes a feature table (one row per CV-JD pair).

Long-running job: embeddings make this ~1-3s per pair on CPU. The script
checkpoints progress every N rows, so it can be interrupted and resumed.

Usage (from backend/ with the venv active):
    python ../ml/features/extract_features.py --split train --per-class 100
    python ../ml/features/extract_features.py --split test  --per-class 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

sys.path.insert(0, str(BACKEND_DIR))

from app.ml.feature_extraction import FEATURE_NAMES, extract_features  # noqa: E402


def select_rows(
    df: pd.DataFrame,
    per_class: int | None = None,
    limit: int | None = None,
    skip_per_class: int | None = None,
) -> pd.DataFrame:
    """Optionally subsample: stratified per-class slice, then a global cap."""
    if skip_per_class is not None:
        # cumcount = per-label ordinal position, so this keeps rows
        # [skip_per_class : skip_per_class + per_class] per label
        # regardless of how the source CSV is ordered.
        rank = df.groupby("label").cumcount()
        df = df[rank >= skip_per_class]
    if per_class is not None:
        df = df.groupby("label", group_keys=False).head(per_class)
    if limit is not None:
        df = df.head(limit)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--per-class", type=int, default=None,
                        help="Stratified subsample: take the first N rows per label")
    parser.add_argument("--skip-per-class", type=int, default=None,
                        help="Skip the first N rows per label before applying --per-class. "
                             "Used to extend an existing extraction without re-processing "
                             "rows that are already in another part file.")
    parser.add_argument("--limit", type=int, default=None, help="Global row cap")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--out", default=None, help="Output CSV path override")
    args = parser.parse_args()

    src = PROCESSED_DIR / f"{args.split}.csv"
    if not src.exists():
        sys.exit(f"Missing {src}; run ml/preprocessing/build_dataset.py first")
    df = pd.read_csv(src)
    df = select_rows(df, args.per_class, args.limit, args.skip_per_class)
    # Keep the source-CSV index: split_row must identify the original row so
    # part files can be merged safely and re-extraction stays deterministic.
    # (Earlier reset_index calls here silently destroyed provenance and
    # corrupted a merge - ordinals are only valid within a single selection.)

    out_path = Path(args.out) if args.out else PROCESSED_DIR / f"{args.split}_features.csv"
    progress_path = out_path.with_suffix(".progress.json")

    # Resume support
    done_rows: list[dict] = []
    start_idx = 0
    if progress_path.exists() and out_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        start_idx = state.get("next_index", 0)
        if start_idx > 0:
            done_rows = pd.read_csv(out_path).to_dict("records")
            print(f"Resuming from row {start_idx} ({len(done_rows)} rows already done)")

    t0 = time.time()
    failures = 0
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        # row.name = original source-CSV index, preserved through selection.
        # split_row is therefore globally unique across train/test parts
        # (label-blocked CSV => disjoint index ranges per split).
        record = {"label": row["label"], "label_int": row["label_int"], "split_row": int(row.name)}
        try:
            record.update(extract_features(row["resume_text"], row["job_description_text"]))
        except Exception as exc:  # noqa: BLE001 - keep batch going, mark row failed
            failures += 1
            record["extraction_error"] = str(exc)[:200]
            for name in FEATURE_NAMES:
                record.setdefault(name, float("nan"))
        done_rows.append(record)

        if (i + 1) % args.checkpoint_every == 0 or i + 1 == len(df):
            pd.DataFrame(done_rows).to_csv(out_path, index=False)
            progress_path.write_text(
                json.dumps({"next_index": i + 1, "total": len(df)}), encoding="utf-8"
            )
            rate = (i + 1 - start_idx) / max(time.time() - t0, 1e-9)
            print(f"[{i + 1}/{len(df)}] {rate:.2f} rows/s  failures={failures}", flush=True)

    result = pd.DataFrame(done_rows)
    result.to_csv(out_path, index=False)
    progress_path.unlink(missing_ok=True)
    print(f"Wrote {out_path} ({len(result)} rows, {failures} failures)")


if __name__ == "__main__":
    main()
