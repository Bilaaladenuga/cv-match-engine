"""
Skill Matching Engine — Phase 8

Compares candidate skills against job requirements using a multi-level
matching strategy:

    Level 1 — EXACT / ALIAS
        Candidate has the same skill (or a known alias) as required.
        Taxonomy resolution guarantees unambiguous canonical matching.
        Score: 1.0

    Level 2 — SEMANTIC
        Candidate doesn't have the exact skill, but a different skill has
        high cosine similarity (>= SEMANTIC_THRESHOLD).  e.g. "React" ↔
        "Frontend Development" when they appear as separate taxonomy entries.
        Score: cosine similarity (0.70 – 0.95)

    Level 3 — RELATED
        Same taxonomy category + moderate similarity (< SEMANTIC_THRESHOLD
        but >= RELATED_THRESHOLD), or a closely related cloud/platform
        equivalent (e.g. AWS ↔ Azure ↔ GCP).
        Score: cosine similarity (0.40 – 0.70)

    Level 4 — MISSING
        No match at any level.
        Score: 0.0

Classification labels: MATCHED, PARTIAL, MISSING, UNKNOWN.

UNKNOWN is reserved for skills not in the taxonomy — the engine cannot
evaluate them, so they are surfaced for manual review.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from app.ml.embeddings import cosine_similarity_matrix, embed_texts
from app.nlp.taxonomy import TAXONOMY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning thresholds
# ---------------------------------------------------------------------------

SEMANTIC_THRESHOLD = 0.70  # above this → MATCHED via semantic
RELATED_THRESHOLD = 0.40   # above this but below semantic → RELATED
# Below RELATED_THRESHOLD → MISSING

# Cloud platforms that are "related but not equivalent"
_CLOUD_EQUIVALENTS = frozenset({"AWS", "Google Cloud Platform", "Azure", "GCP"})

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class MatchLevel(StrEnum):
    """How a candidate skill maps to a job requirement."""

    EXACT = "exact"
    ALIAS = "alias"
    SEMANTIC = "semantic"
    RELATED = "related"
    MISSING = "missing"
    UNKNOWN = "unknown"


class MatchClass(StrEnum):
    """High-level classification shown to the user."""

    MATCHED = "matched"       # EXACT or ALIAS
    PARTIAL = "partial"       # SEMANTIC or RELATED
    MISSING = "missing"       # MISSING
    UNKNOWN = "unknown"       # UNKNOWN


@dataclass
class SkillMatch:
    """One requirement's match result."""

    required_skill: str            # As phrased in the job description
    candidate_skill: str | None    # Best-matching candidate skill (or None)
    match_level: MatchLevel
    match_class: MatchClass
    score: float                   # 0.0 – 1.0
    taxonomy_resolved: str | None  # Canonical name from taxonomy, if found
    evidence: str = ""             # Human-readable explanation


@dataclass
class SkillMatchResult:
    """Aggregate result of matching all job requirements against a CV."""

    matches: list[SkillMatch]
    required_skills: list[str]
    preferred_skills: list[str]

    # Pre-computed aggregates
    matched_skills: list[str] = field(default_factory=list)
    partial_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    unknown_skills: list[str] = field(default_factory=list)

    required_coverage: float = 0.0       # fraction of required that are MATCHED
    required_plus_partial: float = 0.0   # fraction of required that are MATCHED or PARTIAL
    overall_score: float = 0.0           # weighted average across all requirements

    def to_dict(self) -> dict:
        """Serialize for API responses / database storage."""
        return {
            "matches": [
                {
                    "required_skill": m.required_skill,
                    "candidate_skill": m.candidate_skill,
                    "match_level": m.match_level.value,
                    "match_class": m.match_class.value,
                    "score": m.score,
                    "taxonomy_resolved": m.taxonomy_resolved,
                    "evidence": m.evidence,
                }
                for m in self.matches
            ],
            "summary": {
                "matched_skills": self.matched_skills,
                "partial_skills": self.partial_skills,
                "missing_skills": self.missing_skills,
                "unknown_skills": self.unknown_skills,
                "required_coverage": round(self.required_coverage, 4),
                "required_plus_partial": round(self.required_plus_partial, 4),
                "overall_score": round(self.overall_score, 4),
            },
        }


# ---------------------------------------------------------------------------
# Related-skill graph (lightweight, in-module)
# ---------------------------------------------------------------------------

# Groups of skills that are considered "related but not equivalent".
# Skills in the same group are RELATED when one is required and the other
# is possessed by the candidate.
RELATED_GROUPS: list[frozenset[str]] = [
    # Cloud providers
    _CLOUD_EQUIVALENTS,
    # Relational databases
    frozenset({"PostgreSQL", "MySQL", "SQLite", "MariaDB", "Microsoft SQL Server"}),
    # NoSQL databases
    frozenset({"MongoDB", "Cassandra", "DynamoDB", "CouchDB", "Redis"}),
    # Container orchestration
    frozenset({"Docker", "Kubernetes"}),
    # Frontend frameworks
    frozenset({"React", "Angular", "Vue.js", "Svelte", "Next.js"}),
    # Python web frameworks
    frozenset({"Django", "FastAPI", "Flask"}),
    # Testing tools
    frozenset({"pytest", "Jest", "Playwright", "Cypress", "Selenium"}),
    # Data serialization
    frozenset({"REST API", "GraphQL", "gRPC"}),
]

# Build lookup: skill_name → set of related skill names
_RELATED_LOOKUP: dict[str, set[str]] = {}
for _group in RELATED_GROUPS:
    for _skill in _group:
        _RELATED_LOOKUP.setdefault(_skill, set()).update(_group - {_skill})


def are_related(skill_a: str, skill_b: str) -> bool:
    """Check if two skills are in the same related-skill group."""
    return skill_b in _RELATED_LOOKUP.get(skill_a, set())


# ---------------------------------------------------------------------------
# Skill Matcher
# ---------------------------------------------------------------------------


class SkillMatcher:
    """
    Multi-level skill matching engine.

    Usage:
        matcher = SkillMatcher()
        result = matcher.match(
            candidate_skills=["Python", "React", "PostgreSQL"],
            required_skills=["Python", "React", "Docker"],
            preferred_skills=["AWS"],
        )
    """

    def match(
        self,
        candidate_skills: list[str],
        required_skills: list[str],
        preferred_skills: list[str] | None = None,
        candidate_embeddings: list[str] | None = None,
        job_embeddings: list[str] | None = None,
    ) -> SkillMatchResult:
        """
        Match candidate skills against job requirements.

        Args:
            candidate_skills: Skills extracted from the CV (canonical names).
            required_skills: Required skills from the job description.
            preferred_skills: Preferred/nice-to-have skills from the JD.
            candidate_embeddings: Optional pre-computed embedding texts for
                candidate skills (defaults to canonical names).
            job_embeddings: Optional pre-computed embedding texts for
                job skill names (defaults to canonical names).

        Returns:
            SkillMatchResult with per-skill matches and aggregates.
        """
        preferred = preferred_skills or []
        all_job_skills = required_skills + preferred

        if not all_job_skills:
            return SkillMatchResult(
                matches=[],
                required_skills=required_skills,
                preferred_skills=preferred,
            )

        # --- Step 1: Exact / Alias matching via taxonomy ---
        candidate_canonicals: dict[str, str] = {}  # canonical → original
        for skill in candidate_skills:
            resolved = TAXONOMY.canonicalize(skill)
            canonical = resolved if resolved else skill
            candidate_canonicals[canonical] = skill

        candidate_set = set(candidate_canonicals.keys())

        # --- Step 2: Precompute embeddings for semantic matching ---
        candidate_emb_names = candidate_embeddings or list(candidate_canonicals.keys())
        job_emb_names = job_embeddings or all_job_skills

        candidate_embs = embed_texts(candidate_emb_names) if candidate_emb_names else None
        job_embs = embed_texts(job_emb_names) if job_emb_names else None

        # Compute similarity matrix if both sides have embeddings
        sim_matrix = None
        if (
            candidate_embs is not None
            and job_embs is not None
            and len(candidate_embs) > 0
            and len(job_embs) > 0
        ):
            sim_matrix = cosine_similarity_matrix(candidate_embs, job_embs)

        # --- Step 3: Match each job skill ---
        matches: list[SkillMatch] = []

        for job_idx, job_skill in enumerate(all_job_skills):
            is_preferred = job_skill in preferred
            job_resolved = TAXONOMY.canonicalize(job_skill)
            job_canonical = job_resolved if job_resolved else job_skill

            match = self._match_one(
                job_skill=job_skill,
                job_canonical=job_canonical,
                job_idx=job_idx,
                candidate_set=candidate_set,
                candidate_canonicals=candidate_canonicals,
                candidate_emb_names=candidate_emb_names,
                sim_matrix=sim_matrix,
                is_preferred=is_preferred,
            )
            matches.append(match)

        # --- Step 4: Compute aggregates ---
        return self._aggregate(
            matches=matches,
            required_skills=required_skills,
            preferred_skills=preferred,
        )

    def _match_one(
        self,
        job_skill: str,
        job_canonical: str,
        job_idx: int,
        candidate_set: set[str],
        candidate_canonicals: dict[str, str],
        candidate_emb_names: list[str],
        sim_matrix: list[list[float]] | None,
        is_preferred: bool,
    ) -> SkillMatch:
        """Match a single job skill against all candidate skills."""

        # Level 1: Exact / Alias match (taxonomy resolution)
        if job_canonical in candidate_set:
            original = candidate_canonicals[job_canonical]
            # Determine if it was an alias or exact
            is_alias = job_canonical != job_skill.lower().strip()
            level = MatchLevel.ALIAS if is_alias else MatchLevel.EXACT
            return SkillMatch(
                required_skill=job_skill,
                candidate_skill=original,
                match_level=level,
                match_class=MatchClass.MATCHED,
                score=1.0,
                taxonomy_resolved=job_canonical,
                evidence=self._exact_evidence(job_skill, original, job_canonical, is_alias),
            )

        # Skill not in taxonomy at all → UNKNOWN
        if not TAXONOMY.canonicalize(job_skill):
            # Not recognized by taxonomy at all
            return SkillMatch(
                required_skill=job_skill,
                candidate_skill=None,
                match_level=MatchLevel.UNKNOWN,
                match_class=MatchClass.UNKNOWN,
                score=0.0,
                taxonomy_resolved=None,
                evidence=f"'{job_skill}' is not in the skill taxonomy; manual review needed.",
            )

        # Level 2 & 3: Semantic / Related matching
        if sim_matrix is not None and len(candidate_emb_names) > 0:
            # Find the best semantic match (column = all candidates vs this job skill)
            col_idx = job_idx if job_idx < sim_matrix.shape[1] else None
            if col_idx is not None:
                col = sim_matrix[:, col_idx]
                best_cand_idx = int(col.argmax())
                best_score = float(col[best_cand_idx])
                best_candidate = candidate_emb_names[best_cand_idx]

                # Semantic match (high similarity)
                if best_score >= SEMANTIC_THRESHOLD:
                    return SkillMatch(
                        required_skill=job_skill,
                        candidate_skill=best_candidate,
                        match_level=MatchLevel.SEMANTIC,
                        match_class=MatchClass.PARTIAL,
                        score=round(best_score, 4),
                        taxonomy_resolved=job_canonical,
                        evidence=(
                            f"'{best_candidate}' is semantically similar to '{job_skill}' "
                            f"(similarity: {best_score:.2f})."
                        ),
                    )

                # Related match (same category or cloud equivalents)
                if best_score >= RELATED_THRESHOLD:
                    # Extra check: related-skill graph or same taxonomy category
                    cat_match = self._same_category(job_canonical, best_candidate)
                    graph_match = are_related(job_canonical, best_candidate)

                    if cat_match or graph_match:
                        return SkillMatch(
                            required_skill=job_skill,
                            candidate_skill=best_candidate,
                            match_level=MatchLevel.RELATED,
                            match_class=MatchClass.PARTIAL,
                            score=round(best_score, 4),
                            taxonomy_resolved=job_canonical,
                            evidence=(
                                f"'{best_candidate}' is related to '{job_skill}' "
                                f"(similarity: {best_score:.2f})."
                            ),
                        )

        # Level 4: Missing
        return SkillMatch(
            required_skill=job_skill,
            candidate_skill=None,
            match_level=MatchLevel.MISSING,
            match_class=MatchClass.MISSING,
            score=0.0,
            taxonomy_resolved=job_canonical,
            evidence=f"Candidate does not have '{job_skill}' or a sufficiently similar skill.",
        )

    @staticmethod
    def _same_category(skill_a: str, skill_b: str) -> bool:
        """Check if two skills share the same taxonomy category."""
        def_a = TAXONOMY.resolve(skill_a)
        def_b = TAXONOMY.resolve(skill_b)
        if def_a and def_b:
            return def_a.category == def_b.category
        return False

    @staticmethod
    def _exact_evidence(
        job_skill: str, original: str, canonical: str, is_alias: bool
    ) -> str:
        """Generate human-readable evidence for an exact/alias match."""
        if is_alias:
            return (
                f"Candidate lists '{original}' which maps to '{canonical}' "
                f"(alias for required skill '{job_skill}')."
            )
        if job_skill.lower().strip() == canonical.lower():
            return f"Candidate directly lists '{canonical}', matching the requirement."
        return (
            f"Candidate skill '{original}' resolves to '{canonical}', "
            f"matching required skill '{job_skill}'."
        )

    @staticmethod
    def _aggregate(
        matches: list[SkillMatch],
        required_skills: list[str],
        preferred_skills: list[str],
    ) -> SkillMatchResult:
        """Compute aggregate metrics from individual matches."""
        matched = [m.required_skill for m in matches if m.match_class == MatchClass.MATCHED]
        partial = [m.required_skill for m in matches if m.match_class == MatchClass.PARTIAL]
        missing = [m.required_skill for m in matches if m.match_class == MatchClass.MISSING]
        unknown = [m.required_skill for m in matches if m.match_class == MatchClass.UNKNOWN]

        # Coverage: fraction of required skills that are MATCHED
        n_required = len(required_skills)
        n_matched_required = len([s for s in matched if s in required_skills])
        n_partial_required = len([s for s in partial if s in required_skills])

        required_coverage = n_matched_required / n_required if n_required > 0 else 0.0
        required_plus_partial = (
            (n_matched_required + n_partial_required) / n_required if n_required > 0 else 0.0
        )

        # Overall score: weighted by importance
        # Required skills weighted 1.0, preferred skills weighted 0.5
        total_weight = 0.0
        weighted_sum = 0.0
        for m in matches:
            weight = 0.5 if m.required_skill in preferred_skills else 1.0
            total_weight += weight
            weighted_sum += m.score * weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return SkillMatchResult(
            matches=matches,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            matched_skills=matched,
            partial_skills=partial,
            missing_skills=missing,
            unknown_skills=unknown,
            required_coverage=required_coverage,
            required_plus_partial=required_plus_partial,
            overall_score=overall_score,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def match_skills(
    candidate_skills: list[str],
    required_skills: list[str],
    preferred_skills: list[str] | None = None,
) -> SkillMatchResult:
    """
    Quick entry point for skill matching.

    Args:
        candidate_skills: Skills from the CV.
        required_skills: Required skills from the JD.
        preferred_skills: Preferred skills from the JD.

    Returns:
        SkillMatchResult with all matches and aggregates.
    """
    matcher = SkillMatcher()
    return matcher.match(
        candidate_skills=candidate_skills,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
    )
