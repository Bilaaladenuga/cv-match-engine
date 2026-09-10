# ML Methodology — Career Match Engine

> This document explains how the matching engine computes scores, why each
> design choice was made, and how the ML components will be evaluated and
> improved. It is written to be read top-to-bottom by a new engineer, or
> consulted section-by-section.

---

## 1. System philosophy

The engine separates **deterministic processing** (parsers, taxonomy,
rules) from **ML/NLP processing** (embeddings, learned models) and an
**application layer** (API, database, frontend). No LLM API sits behind
the matching pipeline; every component is independently testable and every
score it produces can be traced back to specific evidence.

Design rules that follow from this:

1. Every score carries an explanation grounded in extracted data.
2. Every result is stamped with a `model_version` so historical reports
   stay reproducible.
3. Weights and thresholds are configuration, not hard-coded magic.
4. The system reports *model-estimated compatibility*, never hiring
   probability (see §9, Ethics).

---

## 2. Matching pipeline overview

```
CV text ──► CV Parser (Ph 3–4) ──► CandidateProfile
                                        │
JD text ──► JD Parser (Ph 6) ──► JobProfile
                                        │
        ┌───────────────┼───────────────┬──────────────────┐
        ▼               ▼               ▼                  ▼
   Skill Matcher   Experience      Semantic          Education &
   (Ph 8)          Matcher (Ph 9)  Matcher (Ph 10)   Cert Matchers (Ph 11)
        └───────────────┴───────┬───────┴──────────────────┘
                                ▼
                     Hybrid Matching Model (Ph 11)
                     weighted sum, explanations,
                     model_version
                                │
                                ▼
                      Match Report (API + DB)
```

Deterministic components: parsers, taxonomy resolution, education matcher,
certification aliases. ML components: sentence-transformer embeddings
(Phases 7, 10), semantic fallbacks, and — from Phase 12 — the learned
scoring model that will eventually replace the hand-set weights.

---

## 3. Component scoring

### 3.1 Skills score (Phase 8)

Each job skill is matched against candidate skills through a waterfall:

| Level | Mechanism | Score | Class |
|---|---|---|---|
| Exact | canonical == canonical | 1.0 | MATCHED |
| Alias | taxonomy alias resolution (`JS` → JavaScript) | 1.0 | MATCHED |
| Semantic | embedding cosine ≥ 0.70 | 0.70–0.95 | PARTIAL |
| Related | same category / related-skills graph, cosine ≥ 0.40 | 0.40–0.70 | PARTIAL |
| Missing | nothing above threshold | 0.0 | MISSING |
| Unknown | skill absent from taxonomy | 0.0 | UNKNOWN (manual review) |

Aggregates: `required_coverage` (fraction of required skills MATCHED),
`required_plus_partial`, and `overall_score` — a weighted average where
required skills count 1.0 and preferred skills 0.5.

**Why a waterfall instead of pure embeddings:** deterministic matches are
free, instant, explainable, and never wrong. Embeddings only run where the
taxonomy is silent, and a related skill (AWS vs Azure) is never presented
as a full match.

### 3.2 Experience score (Phase 9)

Three dimensions, each scored 0–1 with a linear gap function:
- **Total years** — candidate total vs JD minimum.
- **Per-skill years** — role-title/context scans first (confidence 0.70–0.95);
  when a skill is only listed in the skills section, total career duration is
  attributed at low confidence (0.50).
- **Seniority level** — junior < 2y ≤ mid < 5y ≤ senior < 8y ≤ lead < 10y ≤ principal.

Classification bands: `strong_match` (≥ 1y surplus), `meets`, `near_match`
(within 1y below), `below`, `unknown`.

**Known limitation:** attribution from the skills section assumes the skill
was exercised across the career, which over-estimates for recently adopted
skills. Role-description scanning (not just titles) is the planned
refinement; the low confidence mark keeps it from dominating the report.

### 3.3 Semantic score (Phase 10)

Weighted combination of three embedding strategies (all-MiniLM-L6-v2,
384-d, cosine):

| Strategy | Weight | Rationale |
|---|---|---|
| Full-document | 0.35 | Global topical relevance |
| Section-level | 0.30 | summary↔description, skills↔requirements, experience↔responsibilities |
| Skill-level | 0.35 | Terminology overlap of the actual skill sets |

Weights auto-renormalize when a strategy has no data. Output is normalized
to 0–100 with labels Strong ≥ 85, Good ≥ 70, Moderate ≥ 55, Fair ≥ 40.

**Why not one big cosine:** full-document similarity is dominated by shared
boilerplate ("experience", "team", "responsible for"); section- and
skill-level signals isolate the informative parts. The mix was validated
against hand-labeled sample pairs and is configurable, not sacred.

### 3.4 Education score (Phase 11)

Deterministic, two components blended 60/40:
- **Level:** strict hierarchy (HS < Associate < Bachelor < Master < PhD).
  Meeting or exceeding = 1.0; one level below = 0.6; deeper gaps decay.
- **Field relevance:** exact/alias 1.0 → related group (CS/SWE/CPE) 0.8 →
  broad tech family 0.6 → unrelated 0.3.

No requirement ⇒ neutral 0.85 (absence of a requirement is not evidence
*for* education). Missing CV degree with a stated requirement ⇒ 0.0 with
evidence noting the parser may have failed rather than the candidate lacking
it. University prestige is deliberately not scored (§9).

### 3.5 Certification score (Phase 11)

Credentials get **no partial credit**: exact or alias match = 1.0, semantic
fallback only above a deliberately high 0.80 cosine (distinct certs like AWS
SAA vs AWS Developer must not merge), otherwise 0.0. Score = fraction of
required certifications held; no requirement ⇒ neutral 0.85.

---

## 4. The hybrid model and weight rationale (Phase 11)

```
overall = 0.40·skills + 0.25·semantic + 0.20·experience
        + 0.10·education + 0.05·certifications
```

The initial weights encode one judgment: **demonstrable, specific skill
overlap is the best available proxy for day-one productivity**, so skills
lead and soft/noisy signals trail:

| Weight | Rationale |
|---|---|
| skills 0.40 | Hardest evidence; least gameable by wording; directly checkable. |
| semantic 0.25 | Catches relevance skill lists miss, but noisy at high values — complements, never dominates. |
| experience 0.20 | Real but weak alone (title inflation, career variability). |
| education 0.10 | Usually a filter, not a predictor of performance; kept from ever dominating. |
| certs 0.05 | Required by a minority of roles; neutral (0.85) when absent. |

Weights live in `app/scoring/weights.py`, are validated (unknown/missing
keys, range, sum-to-1), and can be persisted/loaded as JSON — so they are
configuration, not hard-coded behavior. Phase 12/13 will learn weights from
labeled data and compare against this baseline; if learned weights win on
held-out data, `model_version` bumps and this document is updated.

Explanations rank components by **weighted impact** (raw × weight), because
that is what actually moved the score, and every factor string carries its
point contribution.

---

## 5. Feature engineering (Phase 12)

The learned model will consume these features — all already computed by the
deterministic pipeline, which keeps training faithful to inference:

| Feature | Source | Type |
|---|---|---|
| skill_overlap_ratio | skill matcher | float 0–1 |
| required_skill_coverage | skill matcher | float 0–1 |
| preferred_skill_coverage | skill matcher | float 0–1 |
| semantic_similarity | semantic matcher | float 0–1 |
| experience_gap_years | experience matcher | float |
| experience_score | experience matcher | float 0–1 |
| seniority_match | experience matcher | bool/float |
| education_level_score | education matcher | float 0–1 |
| education_field_score | education matcher | float 0–1 |
| certification_match_ratio | cert matcher | float 0–1 |
| job_title_similarity | embeddings | float 0–1 (planned) |
| skill_category_coverage | taxonomy | float 0–1 (planned) |

**Leakage rule:** features may only use information available at inference
time on a single (CV, JD) pair. Dataset-side artifacts (popularity priors,
dataset id) are excluded.

---

## 6. Dataset plan (Phase 12)

Requirements: (CV text, job description, relevance/compatibility label)
triples with enough volume to train a small supervised model.

Candidate sources, in preference order:
1. **Existing public datasets** — e.g. resume–job description fit datasets
   on Kaggle/HuggingFace with fit labels, or relevance-labeled pairs derived
   from recruitment dumps. Screened for license and label quality.
2. **Distant supervision** — pairs built from real job boards where
   application/interview outcomes act as weak labels.
3. **Synthetic augmentation** — perturbing matched pairs (dropping skills,
   inflating experience) to widen the score range and rebalance classes;
   synthetic data is used only to augment, never as the sole source.

Preprocessing: dedupe identical CVs/JDs, strip contact data (§9), normalize
through the same parsers used at inference, label distribution checked for
imbalance before training.

---

## 7. Model plan & evaluation (Phases 12–13)

Baselines in order: Logistic Regression (interpretability floor), Random
Forest, Gradient Boosting. Metrics:

- **Classification:** Accuracy, Precision, Recall, F1 (macro — not just
  accuracy, because classes are imbalanced), ROC-AUC, confusion matrix.
- **Ranking:** Precision@K, Recall@K, NDCG@K.
- **Calibration:** reliability curves if scores are exposed as probabilities.

An honest evaluation includes error analysis (false positives = candidates
overrated by the model; false negatives = underrated), class imbalance
handling, and explicit dataset-limitation notes. A single high metric is
not treated as success. The current hybrid (hand-weighted) model serves as
the baseline to beat.

---

## 8. Reproducibility & versioning

- `model_version` (currently `match-model-v0.1`) stamps every Match row and
  API response; format `match-model-v<major>.<minor>` — minor bumps for
  tuning, major for redesign.
- Embedding model fixed at `sentence-transformers/all-MiniLM-L6-v2` (384-d);
  changing it is a major bump.
- Taxonomy has `schema_version`; skill changes are additive and seeded
  idempotently.
- See `docs/model-card.md` (Phase 23) for the full model card.

---

## 9. Ethics

- The system reports **model-estimated compatibility**, never "chance of
  getting hired". Every score ships with a decision-support disclaimer.
- Protected characteristics (gender, age, ethnicity, disability), university
  prestige, and employment gaps are **not** features and must never become
  features.
- CVs are personal data: raw text is stored for the user's own analysis,
  logs never carry CV contents, and parsing failures log metadata only.
- Known bias risk: text embeddings can encode occupational gender/age
  associations; we mitigate by (a) keeping semantic weight below skills,
  (b) auditing score distributions across groups once real usage data
  exists, and (c) keeping a human in the loop as a stated product
  requirement.
