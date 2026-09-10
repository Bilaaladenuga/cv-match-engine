"""
Certification Matching — Phase 11

Compares certifications requested in the job description against the
certifications detected on the candidate's CV.

Design notes
------------
    - Deterministic first: alias/equality on lowercased names (the taxonomy
      does not own certifications, so a small local alias set handles common
      equivalents like "aws saa" → "aws certified solutions architect").
    - Semantic fallback: when no deterministic pair matches, embed both sides
      and allow a high-threshold match (>= 0.80). Certifications are specific
      credentials; a lower threshold would produce false "matches" between
      distinct certs (AWS SAA vs AWS Developer). On mismatch the score is 0,
      never "related" — either you hold the credential or you don't.
    - No requirement  →  neutral 0.85.
    - Vague requirement with no cert name (e.g. just "certifications")  →
      treated as no requirement.

Protected-characteristic-free and title-free by design (ethics, Phase 27).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Threshold above which two differently-worded certs are treated as the same
# credential by the semantic fallback.
_SEMANTIC_CERT_THRESHOLD = 0.80

_NEUTRAL_SCORE = 0.85

# Small alias set for common certifications (lowercased). Extendable.
_CERT_ALIASES: dict[str, str] = {
    "aws saa": "aws certified solutions architect",
    "aws solutions architect": "aws certified solutions architect",
    "aws certified solutions architect associate": "aws certified solutions architect",
    "aws cda": "aws certified developer",
    "aws developer associate": "aws certified developer",
    "gcp ace": "google cloud certified professional cloud architect",
    "azure fundamentals az-900": "azure fundamentals",
    "ckad": "certified kubernetes application developer",
    "cka": "certified kubernetes administrator",
}

# Normalizations applied before comparison
_REPLACEMENTS = {
    "certified": "certified",
    "-": " ",
}


def _normalize(name: str) -> str:
    """Lowercase, collapse punctuation/whitespace for stable comparison."""
    n = name.lower().strip()
    for old, new in _REPLACEMENTS.items():
        n = n.replace(old, new)
    return " ".join(n.split())


def _resolve_alias(name: str) -> str:
    n = _normalize(name)
    return _CERT_ALIASES.get(n, n)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CertificationMatch:
    """One required certification's result."""

    required: str
    candidate: str | None
    match_type: str        # "exact", "alias", "semantic", "missing"
    score: float           # 1.0 or 0.0 (no partial credit for credentials)
    evidence: str

    def to_dict(self) -> dict:
        return {
            "required": self.required,
            "candidate": self.candidate,
            "match_type": self.match_type,
            "score": self.score,
            "evidence": self.evidence,
        }


@dataclass
class CertificationMatchResult:
    """Aggregate result over all required certifications."""

    matches: list[CertificationMatch] = field(default_factory=list)
    score: float = _NEUTRAL_SCORE       # 0.0–1.0 aggregate
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "matches": [m.to_dict() for m in self.matches],
            "score": round(self.score, 4),
            "matched": self.matched,
            "missing": self.missing,
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def match_certifications(
    candidate_certs: list[str],
    required_certs: list[str],
) -> CertificationMatchResult:
    """
    Compare candidate certifications against job-required certifications.

    Args:
        candidate_certs: Certification names detected on the CV.
        required_certs: Certification names required/preferred by the job.
    """
    cand_list = [c for c in (candidate_certs or []) if c and c.strip()]
    req_list = [c for c in (required_certs or []) if c and c.strip()]

    if not req_list:
        return CertificationMatchResult(score=_NEUTRAL_SCORE)

    # Lazy import: embeddings load the sentence-transformer (~100ms first call)
    def _semantic_best(req_norm: str) -> tuple[str | None, float]:
        try:
            from app.ml.embeddings import embed_texts

            vecs = embed_texts([req_norm] + [_resolve_alias(c) for c in cand_list])
            sims = _pairwise(req_norm, vecs)
            best_idx = max(range(len(cand_list)), key=lambda i: sims[i])
            return (cand_list[best_idx], sims[best_idx]) if cand_list else (None, 0.0)
        except Exception as exc:  # model unavailable → deterministic only
            logger.warning("Semantic cert matching unavailable: %s", exc)
            return (None, 0.0)

    def _pairwise(req_norm: str, vecs) -> list[float]:
        req_vec = vecs[0]
        out = []
        for v in vecs[1:]:
            dot = sum(a * b for a, b in zip(req_vec, v, strict=False))
            out.append(dot)
        return out

    matched: list[str] = []
    missing: list[str] = []
    matches: list[CertificationMatch] = []

    for req in req_list:
        req_norm = _resolve_alias(req)
        cand_hit: str | None = None
        mtype = "missing"
        evidence = ""

        # 1) Exact / alias
        for c in cand_list:
            if _resolve_alias(c) == req_norm:
                cand_hit = c
                mtype = "alias" if _normalize(c) != req_norm else "exact"
                break

        # 2) Semantic fallback
        if cand_hit is None and cand_list:
            cand_hit, sim = _semantic_best(req_norm)
            if cand_hit is not None and sim >= _SEMANTIC_CERT_THRESHOLD:
                mtype = "semantic"
                evidence = f"Fuzzy credential match (similarity {sim:.2f})."
            else:
                cand_hit = None

        if cand_hit is not None:
            matched.append(req)
            matches.append(
                CertificationMatch(
                    required=req,
                    candidate=cand_hit,
                    match_type=mtype,
                    score=1.0,
                    evidence=evidence
                    or f"Candidate holds '{cand_hit}'.",
                )
            )
        else:
            missing.append(req)
            matches.append(
                CertificationMatch(
                    required=req,
                    candidate=None,
                    match_type="missing",
                    score=0.0,
                    evidence="Required certification not found on the CV.",
                )
            )

    score = len(matched) / len(req_list)

    return CertificationMatchResult(
        matches=matches,
        score=score,
        matched=matched,
        missing=missing,
    )
