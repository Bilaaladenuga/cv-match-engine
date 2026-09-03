"""
Seed the `skills` table from the skill taxonomy.

Usage (from `backend/` with the venv active):

    python -m scripts.seed_skills [--dry-run] [--taxonomy-path PATH]

Every canonical skill in the taxonomy is upserted into PostgreSQL:

* new name                      -> INSERT
* same name, different category -> UPDATE (the taxonomy is authoritative)
* same name, same category      -> untouched (reruns are no-ops)

Rows in the database that no longer exist in the taxonomy are reported but
never deleted — they may be referenced by candidate_skills or
job_requirements rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from sqlalchemy import select

from app.core.database import SessionLocal
from app.models.skill import Skill
from app.nlp.taxonomy import SkillTaxonomy, load_taxonomy


@dataclass
class SeedPlan:
    """What an upsert run would do, before any row is touched."""

    inserts: list[Skill]
    updates: list[Skill]  # existing rows whose category must change
    unchanged: list[str]  # existing canonical names already in sync
    orphaned: list[str]  # in DB but not in the taxonomy (left alone)

    @property
    def total_changes(self) -> int:
        return len(self.inserts) + len(self.updates)


def plan_upserts(taxonomy: SkillTaxonomy, existing_by_name: dict[str, Skill]) -> SeedPlan:
    """
    Compute the insert/update/unchanged/orphan split for one taxonomy.

    Pure function — no database access — so it is easy to unit test. `updates`
    contains the *new* desired values as transient Skill rows; the caller
    applies category changes onto the existing rows.
    """
    taxonomy_names = {skill.name for skill in taxonomy.skills}

    inserts: list[Skill] = []
    updates: list[Skill] = []
    unchanged: list[str] = []

    for skill in taxonomy.skills:
        existing = existing_by_name.get(skill.name)
        if existing is None:
            inserts.append(Skill(name=skill.name, category=skill.category))
        elif existing.category != skill.category:
            updates.append(Skill(name=skill.name, category=skill.category))
        else:
            unchanged.append(skill.name)

    orphaned = sorted(set(existing_by_name) - taxonomy_names)
    return SeedPlan(
        inserts=inserts,
        updates=updates,
        unchanged=unchanged,
        orphaned=orphaned,
    )


def _apply_plan(session, plan: SeedPlan, existing_by_name: dict[str, Skill]) -> None:
    """Persist inserts and category updates from a plan."""
    for new_skill in plan.inserts:
        session.add(new_skill)
    for desired in plan.updates:
        existing_by_name[desired.name].category = desired.category


def _print_plan(plan: SeedPlan) -> None:
    print(f"Taxonomy skills : {len(plan.inserts) + len(plan.updates) + len(plan.unchanged)}")
    print(f"  to insert     : {len(plan.inserts)}")
    print(f"  to update     : {len(plan.updates)}")
    print(f"  in sync       : {len(plan.unchanged)}")
    print(f"DB rows not in taxonomy (kept, not deleted): {len(plan.orphaned)}")
    for name in plan.orphaned:
        print(f"  [orphan] {name}")

    for skill in plan.inserts:
        print(f"  [insert] {skill.name} ({skill.category})")
    for skill in plan.updates:
        print(f"  [update] {skill.name} -> ({skill.category})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without touching the database",
    )
    parser.add_argument(
        "--taxonomy-path",
        default=None,
        help="Path to a taxonomy JSON file (default: bundled taxonomy or $SKILL_TAXONOMY_PATH)",
    )
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy_path)
    print(f"Loaded taxonomy: {len(taxonomy.skills)} skills from {taxonomy.schema_version}")

    with SessionLocal() as session:
        existing = {skill.name: skill for skill in session.scalars(select(Skill)).all()}
        plan = plan_upserts(taxonomy, existing)

        if args.dry_run:
            print("\nDRY RUN - no changes written\n")
            _print_plan(plan)
            return

        if plan.total_changes == 0:
            print("\nAlready in sync - nothing to do.")
            return

        _apply_plan(session, plan, existing)
        session.commit()
        print(f"\nSeeded {len(plan.inserts)} new skills, updated {len(plan.updates)}.")
        _print_plan(plan)

    print("\nDone.")


if __name__ == "__main__":
    main()
