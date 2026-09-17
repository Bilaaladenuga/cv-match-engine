# Model v0.5 Retrain Plan

Status: **planned** (not yet implemented). This document defines what v0.5
changes, why, and how we will know it worked. Written after the taxonomy
expansion to 26 career fields; read with `docs/model-card.md` (v0.4.0) and
`docs/ml-methodology.md`.

---

## 1. Motivation

The v0.4.0 classifier was trained on a **tech-sector** dataset
(`cnamuangtoun/resume-job-description-fit`) over a **36-feature** schema
whose 17 `cov_*` coverage features are pinned to the original taxonomy
categories (programming, frontend, backend, database, cloud, devops,
data_science, machine_learning, other). Since then:

1. The taxonomy grew to **26 categories / 280 skills**. New-domain skills
   aggregate into `cov_other` at serving time, so the engine *works* for
   nursing, welding, or paralegal CVs — but the trained model receives no
   field-resolved signal for them.
2. The product is now explicitly field-agnostic. A model whose learned
   thresholds are tech-market artifacts is the main remaining gap.

## 2. The core problem (decided up front)

Schema expansion and data expansion are **not independent**:

| Option | What happens | Verdict |
|---|---|---|
| Add `cov_<newcat>` features only | Training rows are still all tech → new features have ~zero variance → nothing learned, extra noise | rejected |
| Add field-diverse data only | New-domain skills fall into `cov_other` → model sees coarser signal than it could | useful but leaves value on the table |
| **Data first, then schema, then retrain** | Features are promoted only for categories with real support in the new data | **chosen** |
| Neither (stay on v0.4.0) | Deterministic engines carry non-tech fields; ML probabilities stay tech-flavored | fallback if no data is obtainable |

## 3. Dataset strategy (the real work)

Requirement: CV–JD pairs with fit labels, spanning the target fields,
with enough per-field support for coverage features to be learnable
(target ≥ ~300 rows per major field group).

1. **Search first.** Audit HF/academic datasets for multi-domain
   resume–job fit labels. Acceptance criteria: documented label semantics,
   per-field counts, license permitting redistribution of derived
   features.
2. **If none exists: weak-label real data.** Take an unlabeled multi-domain
   CV–JD corpus and label it with the deterministic engines (skill
   coverage, experience gap, education match) as weak supervisors, then
   **human-audit a held-out slice** (≈200 pairs, stratified by field) and
   report human/model agreement. Risks: the model would partly learn the
   engines' own biases — mitigations: agreement reporting, keeping the
   semantic-similarity and ML features as the model's ways to *exceed* the
   engines, and stating the circularity plainly in the model card.
3. **Split discipline** (Phase 13 lessons applied): stratify by field AND
   keep job-slate groups intact for ranking metrics.

## 4. Schema v0.5

- Promote `cov_<cat>_required` + `cov_<cat>_n` for the top new categories
  ranked by JD demand in the new dataset (cap the list to keep dimensionality
  sane; candidates: engineering, healthcare, finance, law, education,
  marketing, sales, hr, logistics, trades).
- Bump the code-level feature-schema version constant; old artifacts stay
  reproducible (same mechanism as the v0.3→v0.4 bump).
- Categories not promoted keep flowing into `cov_other` (unchanged).

## 5. Training protocol

Same baselines as Phase 12 (Logistic Regression / Random Forest /
Gradient Boosting, class weights), prior correction retained (v0.3.1),
evaluation extended:

- Per-class metrics, confusion matrices, error analysis — **overall and
  per field group**.
- Ranking metrics (Precision@K, NDCG@5 within job slates) — overall and
  per field group. A model that ranks tech well and everything else
  randomly is precisely the v0.5 failure mode we are measuring for.
- Calibration quality per field group.

## 6. Acceptance gates (defined before training)

1. **No tech regression**: tech-slice macro-F1 within ~2 pts of v0.4.0.
2. **Generalization**: macro-F1 averaged across field groups ≥ v0.4.0 +
   0.03, or — if the comparison baseline has no non-tech rows — reported
   with a human-audit agreement figure.
3. **Field fairness**: best-vs-worst field-group NDCG@5 gap ≤ 0.15.
4. **Calibration**: reliability curve slope within [0.8, 1.2] per major
   field group after prior correction.

Failing a gate → the plan explicitly allows shipping **v0.4.0 semantics
with better documentation** (fallback row of the decision table) rather
than a worse model.

## 7. Rollout

1. Train offline; artifacts to `ml/models/` stamped `match-model-v0.5.0`
   (dataset hash + feature schema version + seeds recorded in the model
   card).
2. **Shadow mode**: serving scorer computes v0.5 alongside v0.4.0 and logs
   (never serves) the delta; compare per-field score distributions.
3. Flip the served version; old stored reports keep their stamped version
   (reproducibility contract, Phase 23).
4. Update `docs/model-card.md` with the v0.5 schema, per-field results,
   and new limitations.

## 8. Out of scope (explicitly)

- LLM-based scoring or LLM feature generation (violates the project's
  explainability/engine-first philosophy).
- Field-fine-tuned embedding models (possible v0.6 lever).
- Per-field model selection at serving time (one model, per-field
  *evaluation* instead — simpler and auditable).

## 9. Work breakdown

| # | Task | Notes |
|---|---|---|
| 1 | Dataset acquisition/audit script + per-field counts report | go/no-go on "search" path |
| 2 | Weak-label generation + human audit slice (if path 2) | agreement metrics mandatory |
| 3 | Schema v0.5 bump + feature-table regeneration | reuse share-keyed join approach from v0.4.0 |
| 4 | Per-field evaluation harness | extends Phase 13 scripts |
| 5 | Training + gate report | gates from §6 |
| 6 | Model card + shadow rollout + flip | §7 |

## 10. Ethics note

No protected attributes are used as features or evaluation strata (Phase
27). Field-group evaluation is occupational, not demographic; a
per-field fairness gate (§6.3) also limits proxies that would encode
field-typical demographics through the back door.
