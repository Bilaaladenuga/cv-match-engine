"""
Skill Taxonomy — the canonical, extensible skill vocabulary.

Skills are defined in `skill_taxonomy.json` as a list of entries with a
canonical name, a category id, and a list of aliases (e.g. "JS" → JavaScript,
"postgres" → PostgreSQL). The taxonomy is the single source of truth used by
the skill extractor, the job-description parser, and later by the skill
matching engine.

Extending the taxonomy:

1. Data-driven: add entries to `skill_taxonomy.json` (no code change needed).
2. Programmatic: use `SkillTaxonomy.register_skill(...)` at runtime, e.g. when
   a job advertises a niche skill not yet in the taxonomy.
3. Custom files: load a different/merged taxonomy via `load_taxonomy(path)`.

The taxonomy enforces integrity as entries are added: duplicate canonical
names and aliases that map to more than one canonical skill raise
`SkillTaxonomyError`, so the alias lookup stays unambiguous. A skill in a
JSON file must reference one of the declared categories (unknown categories
fail fast — this also catches typos); runtime `register_skill` calls may
introduce new categories deliberately.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TAXONOMY_PATH = Path(__file__).parent / "skill_taxonomy.json"
_ENV_TAXONOMY_PATH = "SKILL_TAXONOMY_PATH"


class SkillTaxonomyError(ValueError):
    """Raised when the taxonomy data is invalid or an extension conflicts."""


def _normalize_key(name: str) -> str:
    """Normalize an alias/canonical for lookup: lowercase, stripped."""
    return name.strip().lower()


@dataclass(frozen=True)
class CategoryDef:
    """Metadata for one skill category."""

    id: str
    label: str
    description: str = ""

    def as_dict(self) -> dict:
        return {"label": self.label, "description": self.description}


@dataclass(frozen=True)
class SkillDef:
    """One canonical skill entry in the taxonomy."""

    name: str
    category: str
    aliases: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"name": self.name, "category": self.category, "aliases": list(self.aliases)}


class SkillTaxonomy:
    """
    Loaded, validated skill vocabulary.

    Provides alias → canonical resolution, canonicalization, category lookup,
    and runtime extension via `register_skill`.
    """

    def __init__(self, data: dict | None = None):
        data = data if data is not None else {}
        self.schema_version = str(data.get("schema_version", ""))

        raw_categories = data.get("categories", {})
        raw_skills = data.get("skills", [])

        self._categories: dict[str, CategoryDef] = {}
        for cat_id, meta in raw_categories.items():
            meta = meta or {}
            self._categories[cat_id] = CategoryDef(
                id=cat_id,
                label=meta.get("label", cat_id),
                description=meta.get("description", ""),
            )

        self._skills: list[SkillDef] = []
        # name(lower) -> SkillDef and alias(lower) -> SkillDef
        self._by_name: dict[str, SkillDef] = {}
        self._alias_map: dict[str, SkillDef] = {}

        for entry in raw_skills:
            self._add_skill(
                name=entry.get("name"),
                category=entry.get("category"),
                aliases=entry.get("aliases", []),
                allow_new_category=False,
            )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> SkillTaxonomy:
        """Load a taxonomy from a JSON file (defaults to the bundled file)."""
        json_path = Path(path) if path else Path(
            os.environ.get(_ENV_TAXONOMY_PATH, DEFAULT_TAXONOMY_PATH)
        )
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    @property
    def categories(self) -> tuple[CategoryDef, ...]:
        return tuple(self._categories.values())

    @property
    def skills(self) -> tuple[SkillDef, ...]:
        return tuple(self._skills)

    @property
    def category_ids(self) -> tuple[str, ...]:
        return tuple(self._categories.keys())

    def get_category(self, category_id: str) -> CategoryDef | None:
        return self._categories.get(category_id)

    def get_skill(self, canonical_name: str) -> SkillDef | None:
        """Return the skill by exact canonical name (case-insensitive)."""
        return self._by_name.get(_normalize_key(canonical_name))

    def resolve(self, alias: str) -> SkillDef | None:
        """Resolve any alias or canonical name to its canonical SkillDef."""
        return self._alias_map.get(_normalize_key(alias))

    def canonicalize(self, name: str) -> str | None:
        """Map an alias/canonical spelling to the canonical skill name."""
        skill = self.resolve(name)
        return skill.name if skill else None

    def skills_in_category(self, category_id: str) -> tuple[SkillDef, ...]:
        return tuple(s for s in self._skills if s.category == category_id)

    def lookup_items(self) -> dict[str, tuple[str, str]]:
        """Flat alias → (canonical_name, category) map for fast scanning."""
        return {
            alias: (skill.name, skill.category) for alias, skill in self._alias_map.items()
        }

    # ------------------------------------------------------------------
    # Extension
    # ------------------------------------------------------------------

    def register_skill(
        self,
        name: str,
        category: str,
        aliases: list[str] | tuple[str, ...] = (),
    ) -> SkillDef:
        """
        Add a skill to the taxonomy at runtime.

        Adding an alias already claimed by a *different* canonical skill raises
        `SkillTaxonomyError`; re-registering the same canonical merges aliases.
        Unknown categories are created automatically (with a title-cased label).
        """
        return self._add_skill(name=name, category=category, aliases=list(aliases))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize back to the JSON file shape (for round-tripping/tooling)."""
        return {
            "schema_version": self.schema_version,
            "categories": {
                cat.id: cat.as_dict() for cat in self._categories.values()
            },
            "skills": [skill.as_dict() for skill in self._skills],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _add_skill(
        self, name: str, category: str, aliases: list[str], allow_new_category: bool = True
    ) -> SkillDef:
        if not name or not str(name).strip():
            raise SkillTaxonomyError("Skill entry missing a 'name'")
        if not category or not str(category).strip():
            raise SkillTaxonomyError(f"Skill '{name}' missing a 'category'")

        name = str(name).strip()
        category = str(category).strip()
        name_key = _normalize_key(name)

        # Unknown category: only allowed via runtime registration, so a typo in
        # the taxonomy JSON fails fast instead of creating a bogus category
        if category not in self._categories:
            if not allow_new_category:
                raise SkillTaxonomyError(
                    f"Skill '{name}' references unknown category '{category}'"
                )
            self._categories[category] = CategoryDef(
                id=category, label=category.replace("_", " ").title()
            )

        existing = self._by_name.get(name_key)
        if existing is not None:
            if existing.category != category:
                raise SkillTaxonomyError(
                    f"Skill '{name}' already registered under category "
                    f"'{existing.category}', cannot re-register as '{category}'"
                )
            # Merge new aliases into the existing skill (same canonical)
            merged = tuple(dict.fromkeys(existing.aliases + tuple(aliases)))
            skill = SkillDef(name=name, category=category, aliases=merged)
            self._skills[self._skills.index(existing)] = skill
            self._by_name[name_key] = skill
            self._alias_map[name_key] = skill
            for alias in merged:
                self._claim_alias(alias, name)
                self._alias_map[_normalize_key(alias)] = skill
            return skill

        skill = SkillDef(name=name, category=category, aliases=tuple(aliases))
        self._skills.append(skill)
        self._by_name[name_key] = skill
        # The canonical name is itself always resolvable
        self._claim_alias(name, name)
        self._alias_map[name_key] = skill
        for alias in aliases:
            self._claim_alias(alias, name)
            self._alias_map[_normalize_key(alias)] = skill
        return skill

    def _claim_alias(self, alias: str, skill_name: str) -> None:
        """
        Enforce that every alias maps to exactly one canonical skill.

        Called on load and on runtime registration so ambiguity fails fast
        instead of silently overwriting an existing mapping.
        """
        alias_key = _normalize_key(alias)
        if not alias_key:
            raise SkillTaxonomyError(f"Skill '{skill_name}' contains an empty alias")
        owner = self._alias_map.get(alias_key)
        if owner is not None and owner.name != skill_name:
            raise SkillTaxonomyError(
                f"Cannot register alias '{alias}' for skill '{skill_name}': "
                f"already claimed by '{owner.name}'"
            )


# Cached default taxonomy so extraction code doesn't re-read the file per call
_default_taxonomy: SkillTaxonomy | None = None


def load_taxonomy(path: str | Path | None = None) -> SkillTaxonomy:
    """
    Load a taxonomy.

    Without `path`, returns the cached default (env var `SKILL_TAXONOMY_PATH`
    overrides the bundled file). Pass an explicit `path` to load a custom file.
    """
    global _default_taxonomy
    if path is None:
        if _default_taxonomy is None:
            _default_taxonomy = SkillTaxonomy.from_json()
        return _default_taxonomy
    return SkillTaxonomy.from_json(path)


# Convenience module-level default instance
TAXONOMY = load_taxonomy()
