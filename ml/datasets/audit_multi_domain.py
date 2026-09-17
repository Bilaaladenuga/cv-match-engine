"""v0.5 Task 1 — audit multi-domain CV-JD datasets for the retrain.

For each dataset in DATASETS (HF dataset ids), pull a dispersed sample via the
HuggingFace datasets-server API (no `datasets` library required), classify
each row into the engine's career fields, and write a JSON report with
per-field counts, label distribution, provenance (synthetic-vs-real signals),
and a GO/NO-GO verdict per dataset against docs/model-v05-plan.md §3.

Sampling
--------
Rows are taken at EVENLY SPACED offsets across the whole split (stride =
total // sample_size), not one contiguous window: many CV-JD datasets store
rows grouped by label or domain, and a contiguous sample can be badly
biased (the v0.4 dataset's first 200 rows are all "Good Fit").

Field classification signal (deliberately cheap and auditable):
1. Primary: dataset-declared domain columns (resume_domain / category etc.)
   bridged to our taxonomy field ids.
2. Fallback: taxonomy alias hits on the resume text — a row is attributed
   to its dominant field when that field has >= MIN_SKILL_HITS canonical
   skills present and >= 40% of all alias hits; anything weaker stays
   'unknown' rather than being guessed.

Breadth gate
------------
The plan's bar is ~300 usable training rows per field at FULL scale, so the
gate is applied as a projection: field_fraction_in_sample ×
dataset_rows_total >= PROJECTED_MIN_ROWS (plus >= 3 sample rows as evidence
the field actually exists, not a rounding artifact). A raw
"rows-in-sample" gate would wrongly fail rare-but-viable fields: law at
1.5% of an 80k dataset is ~1,200 projected rows.

Usage:  python ml/datasets/audit_multi_domain.py [--per-dataset N] [--timeout S]
Output: ml/datasets/multi_domain_audit.json + a printed summary table.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.nlp.taxonomy import load_taxonomy  # noqa: E402  (repo-root import)

# --- Audit manifest ---------------------------------------------------------
# Field: (dataset_id, text/label/domain column mappings, per-dataset notes).
DATASETS = [
    {
        "id": "cnamuangtoun/resume-job-description-fit",
        "text_fields": ["resume_text", "job_description_text"],
        "label_field": "label",
        "domain_fields": [],
        "note": "v0.4.0 training data; audited to quantify its field skew",
    },
    {
        "id": "med2425/resume-job-fit-merged-v1",
        "text_fields": ["resume", "jd"],
        "label_field": "label",
        "domain_fields": ["resume_domain", "jd_domain"],
        "source_field": "source",
        "note": "80k rows, 3-class labels + domain annotations; check provenance",
    },
    {
        "id": "batuhanmtl/job_resume_fit",
        "text_fields": ["resume_text", "job_text"],
        "label_field": None,  # continuous ai_match_score, no 3-class label
        "domain_fields": ["category"],
        "note": "category-tagged; continuous score would need banding",
    },
]

# Plan §3 acceptance criteria, operationalized for an audit SAMPLE.
MIN_CLASSIFIED_FRACTION = 0.60
MIN_FIELD_GROUPS = 5       # distinct fields that clear the projection gate
PROJECTED_MIN_ROWS = 300   # plan §3: ~300 usable training rows per field
MIN_SAMPLE_EVIDENCE = 3    # field must appear >= 3x in the sample to project
MIN_ROWS_TOTAL = 5000      # dataset must plausibly yield enough rows
SAMPLE_SIZE = 300
MAX_ROWS_CAP = 100         # datasets-server /rows page limit
LABELS_EXPECTED = {"No Fit", "Potential Fit", "Good Fit"}

# Bridge from dataset-declared domain strings to our taxonomy field ids.
# Domains without a clean counterpart map to None (declared but unmapped).
DOMAIN_BRIDGE = {
    "software": "programming",
    "information-technology": "programming",
    "information_technology": "programming",
    "engineering": "engineering",
    "finance": "finance",
    "data": "data_science",
    "healthcare": "healthcare",
    "medical": "healthcare",
    "legal": "law",
    "law": "law",
    "education": "education",
    "teaching": "education",
    "marketing": "marketing",
    "sales": "sales",
    "hr": "hr",
    "design": "design",
    "designer": "design",
    "media": "media",
    "trades": "trades",
    "logistics": "logistics",
    "science": "science",
    "hospitality": "hospitality",
    "public_sector": "public_sector",
}

FIELDS_DIR = REPO_ROOT / "ml" / "datasets"
OUTPUT_PATH = FIELDS_DIR / "multi_domain_audit.json"


# --- HF datasets-server client ----------------------------------------------
def fetch_json(url: str, timeout: float, retries: int = 3, backoff: float = 15.0) -> dict:
    """GET with retry/backoff for HF rate limits (HTTP 429)."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": "cv-match-engine-audit/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            last_exc = exc
            wait = backoff * (attempt + 1)
            print(f"    429 rate-limited; retrying in {wait:.0f}s …", flush=True)
            time.sleep(wait)
    raise last_exc  # pragma: no cover


def fetch_split_rows(
    dataset_id: str, split: str, num_rows: int, timeout: float
) -> tuple[list[dict], int, list[str]]:
    """Dispersed sample: evenly spaced windows across the whole split.

    stride = total // num_rows guarantees the offsets span the dataset.
    Window length is capped both by the API limit and by the stride itself
    (pages must never overlap, which would duplicate rows).
    """
    info = fetch_json(
        f"https://datasets-server.huggingface.co/info?dataset={urllib.parse.quote(dataset_id, safe='')}",
        timeout,
    )
    di = info["dataset_info"]["default"]
    total = next(iter(di.get("splits", {}).values())).get("num_examples", 0)
    feats = [f for f, v in di.get("features", {}).items() if isinstance(v, dict) and v.get("dtype") == "string"]
    if total == 0 or num_rows <= 0:
        return [], total, feats

    stride = max(1, total // num_rows)
    offset = random.randint(0, max(0, total - 1))
    rows: list[dict] = []
    while len(rows) < num_rows and offset < total:
        length = min(MAX_ROWS_CAP, stride, num_rows - len(rows))
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={urllib.parse.quote(dataset_id, safe='')}"
            f"&config=default&split={split}&offset={offset}&length={length}"
        )
        page = fetch_json(url, timeout)
        batch = [r["row"] for r in page.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += stride
    return rows, total, feats


# --- Field classification ----------------------------------------------------
def build_alias_index(taxonomy) -> dict[str, str]:
    index: dict[str, str] = {}
    for skill in taxonomy.skills:
        for alias in [skill.name, *skill.aliases]:
            index.setdefault(alias.lower(), skill.name)
    return index


def classify_field(
    lowered_text: str, alias_index: dict[str, str], taxonomy, min_hits: int
) -> tuple[str | None, dict[str, int]]:
    """Return (field_id | None, per-field hit counts) for a resume text.

    A row is attributed to its dominant field when that field has >= min_hits
    canonical skills present and at least 40% of all alias hits. Generalist
    CVs legitimately spread hits across categories (a dev with Python, AWS,
    SQL and Git hits 4 fields) — so the bar is dominance, not purity; rows
    below it stay 'unknown' rather than being guessed.
    """
    lowered = lowered_text
    hits: Counter[str] = Counter()
    seen_alias: set[str] = set()
    for alias, canonical in alias_index.items():
        if alias in seen_alias or len(alias) < 3:
            continue
        if alias in lowered:
            seen_alias.add(alias)
            skill = taxonomy.get_skill(canonical)
            if skill:
                hits[skill.category] += 1
    if not hits:
        return None, {}
    best_field, best_n = hits.most_common(1)[0]
    total_hits = sum(hits.values())
    if best_n < min_hits or best_n < 0.4 * total_hits:
        return None, dict(hits)  # ambiguous or too weak → unknown
    return best_field, dict(hits)


def project_field_support(
    field_counts: dict[str, int], sampled_rows: int, dataset_rows_total: int
) -> dict[str, tuple[int, int]]:
    """Project sample field counts to full-dataset scale.

    Returns {field: (projected_rows, rows_in_sample)}. The plan's bar is
    ~300 usable training rows per field at FULL scale: a raw rows-in-sample
    gate would fail rare-but-viable fields (law at 1.5% of 80k ≈ 1,200
    projected rows), so the gate multiplies the sample share by the total.
    """
    n = max(1, sampled_rows)
    return {
        f: (round(c / n * dataset_rows_total), c)
        for f, c in field_counts.items()
        if f != "unknown" and c > 0
    }


def audit_dataset(spec: dict, per_dataset: int, timeout: float, min_hits: int) -> dict:
    rows, total, feats = fetch_split_rows(spec["id"], "train", per_dataset, timeout)
    taxonomy = load_taxonomy()
    alias_index = build_alias_index(taxonomy)

    field_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    field_label: dict[str, Counter] = {}
    resume_domain_counts: Counter[str] = Counter()  # resume-side only (no JD double count)
    source_counts: Counter[str] = Counter()
    cross_domain = 0
    cross_domain_pairs = 0
    mean_chars = 0
    declared_field = 0  # rows where a declared domain mapped to one of our fields
    for row in rows:
        resume_text = " ".join(str(row.get(f, "")) for f in spec["text_fields"][:1])
        jd_text = " ".join(str(row.get(f, "")) for f in spec["text_fields"][1:])
        mean_chars += len(resume_text)

        # Primary signal: first declared domain field that bridges cleanly.
        field: str | None = None
        for df in spec["domain_fields"]:
            declared = str(row.get(df, "")).strip().lower()
            mapped = DOMAIN_BRIDGE.get(declared)
            if mapped:
                field = mapped
                declared_field += 1
                break
        # Fallback: taxonomy alias inference on the resume text.
        if field is None:
            field, _hits = classify_field(resume_text.lower(), alias_index, taxonomy, min_hits)
        field_counts[field or "unknown"] += 1

        label = str(row.get(spec["label_field"], "?")) if spec["label_field"] else None
        if label:
            label_counts[label] += 1
            if field:
                field_label.setdefault(field, Counter())[label] += 1

        for df in spec["domain_fields"]:
            resume_domain_counts[str(row.get(df, "?"))] += 1
        if "resume_domain" in spec["domain_fields"] and "jd_domain" in spec["domain_fields"]:
            cross_domain_pairs += 1
            if str(row.get("resume_domain")) != str(row.get("jd_domain")):
                cross_domain += 1
        if spec.get("source_field"):
            source_counts[str(row.get(spec["source_field"], "?"))] += 1

    # Projection gate: field share in sample × full dataset size.
    n = len(rows) or 1
    projected = project_field_support(field_counts, n, total)
    supported_fields = {
        f: proj[0]
        for f, proj in projected.items()
        if proj[0] >= PROJECTED_MIN_ROWS and proj[1] >= MIN_SAMPLE_EVIDENCE
    }
    classified = sum(c for f, c in field_counts.items() if f != "unknown")
    gates = {
        "rows_available": total >= MIN_ROWS_TOTAL,
        "label_scheme": (not spec["label_field"]) or set(label_counts) & LABELS_EXPECTED != set(),
        "classified_fraction": classified >= MIN_CLASSIFIED_FRACTION * len(rows),
        "field_breadth": len(supported_fields) >= MIN_FIELD_GROUPS,
    }
    return {
        "dataset_id": spec["id"],
        "note": spec["note"],
        "sampled_rows": len(rows),
        "dataset_rows_total": total,
        "string_features": feats,
        "label_distribution": dict(label_counts),
        "field_counts": dict(field_counts),
        "field_label_distribution": {f: dict(c) for f, c in sorted(field_label.items())},
        "projected_rows_per_field": projected,
        "supported_fields": supported_fields,
        "declared_domain_mapped_fraction": round(declared_field / n, 3) if rows else 0.0,
        "resume_domain_columns": dict(resume_domain_counts) if spec["domain_fields"] else None,
        "source_distribution": dict(source_counts) or None,
        "cross_domain_pair_fraction": round(cross_domain / cross_domain_pairs, 3) if cross_domain_pairs else 0.0,
        "mean_resume_chars": round(mean_chars / n) if rows else 0,
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
        "missing_gates": [k for k, v in gates.items() if not v],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-dataset", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--min-skill-hits", type=int, default=3)
    args = parser.parse_args()

    random.seed(42)  # reproducible sampling offsets
    reports = []
    for spec in DATASETS:
        print(f"Auditing {spec['id']} …", flush=True)
        try:
            reports.append(audit_dataset(spec, args.per_dataset, args.timeout, args.min_skill_hits))
        except Exception as exc:  # network/HTTP errors → record, keep auditing
            reports.append({"dataset_id": spec["id"], "error": f"{type(exc).__name__}: {exc}", "verdict": "ERROR"})
        time.sleep(1.0)

    OUTPUT_PATH.write_text(json.dumps({"sample_size": args.per_dataset, "results": reports}, indent=2), encoding="utf-8")

    print(f"\n{'dataset':45s} {'rows':>8s} {'supported':>9s} {'classified':>10s}  verdict")
    print("-" * 92)
    for r in reports:
        if "error" in r:
            print(f"{r['dataset_id']:45s} {'-':>8s} {'-':>9s} {'-':>10s}  ERROR: {r['error'][:40]}")
            continue
        frac = sum(v for k, v in r["field_counts"].items() if k != "unknown") / max(1, r["sampled_rows"])
        print(
            f"{r['dataset_id']:45s} {r['dataset_rows_total']:>8d} {len(r['supported_fields']):>9d} {frac:>9.0%}  {r['verdict']}"
            + (f"  (missing: {', '.join(r['missing_gates'])})" if r["missing_gates"] else "")
        )
    print(f"\nFull report: {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
