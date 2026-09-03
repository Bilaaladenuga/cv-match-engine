"""
Skill Extractor — identifies skill mentions in CV and job text.

Consumes the canonical skill taxonomy (`skill_taxonomy.json` via
`app.nlp.taxonomy.SkillTaxonomy`) and finds every alias that appears in the
text. Matching is case-insensitive and word-boundary safe so short aliases
like "go", "ml" or "r" never match inside unrelated words.

The extractor exposes the same API it always did (`extract_skills`,
`extract_skills_from_list`, `ExtractedSkill`); extending the vocabulary is
done by extending the taxonomy, not this module.
"""

import re
from dataclasses import dataclass

from app.nlp.taxonomy import TAXONOMY


@dataclass
class ExtractedSkill:
    """A skill extracted from text."""

    name: str  # Canonical skill name from the taxonomy
    raw_text: str  # Original alias text where the skill was found
    category: str  # Category id from the taxonomy
    confidence: float  # 0.0–1.0


def _alias_pattern(alias: str) -> str:
    """
    Build a safe regex for one alias.

    Multi-character aliases use word boundaries so e.g. "go" does not match
    inside "golang". A preceding dot also blocks a match: "js" must not match
    inside "next.js"/"node.js" (`.` is not a word char, so a plain word
    boundary would let it through). Single-character aliases (only "r" in the
    taxonomy today) additionally require whitespace or line edges on both
    sides, so "R&D" is not read as the R language.
    """
    if len(alias) == 1:
        return rf"(?<!\S){re.escape(alias)}(?!\S)"
    return rf"(?<![\w.]){re.escape(alias)}(?!\w)"


# Flat alias -> (canonical name, category), built once at import time
_LOOKUP: dict[str, tuple[str, str]] = TAXONOMY.lookup_items()


def extract_skills(text: str) -> list[ExtractedSkill]:
    """
    Extract skill mentions from text.

    Scans the text for every taxonomy alias and returns deduplicated results
    (one entry per canonical skill) with category and confidence.
    """
    text_lower = text.lower()
    seen: dict[str, ExtractedSkill] = {}

    for alias, (canonical, category) in _LOOKUP.items():
        pattern = _alias_pattern(alias)
        if re.search(pattern, text_lower) and canonical not in seen:
            seen[canonical] = ExtractedSkill(
                name=canonical,
                raw_text=alias,
                category=category,
                confidence=0.9,
            )

    return list(seen.values())


def extract_skills_from_list(text: str) -> list[ExtractedSkill]:
    """
    Extract skills from a comma/semicolon/bullet-separated list.

    This handles the common CV format:
    "Python, JavaScript, React, PostgreSQL, Docker"
    """
    # Split on common delimiters
    items = re.split(r"[,;•\-–—|/]\s*", text)
    results: list[ExtractedSkill] = []

    for item in items:
        item = item.strip().strip('"').strip("'")
        if not item:
            continue
        # Exact alias/canonical match first
        entry = TAXONOMY.resolve(item)
        if entry is not None:
            results.append(ExtractedSkill(
                name=entry.name,
                raw_text=item,
                category=entry.category,
                confidence=0.95,
            ))
        else:
            # The item may combine several skills ("AWS Lambda", "React/Redux")
            results.extend(extract_skills(item))

    # Deduplicate by canonical name
    seen: dict[str, ExtractedSkill] = {}
    for skill in results:
        if skill.name not in seen:
            seen[skill.name] = skill
    return list(seen.values())
