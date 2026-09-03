"""
Tests for the skill-seeding planner (`scripts.seed_skills`).

The planner is a pure function (no database), so these tests run without a
PostgreSQL connection.
"""

from app.models.skill import Skill
from app.nlp.taxonomy import load_taxonomy
from scripts.seed_skills import plan_upserts


def make_skill(name: str, category: str) -> Skill:
    return Skill(name=name, category=category)


def test_empty_db_plans_everything_as_insert():
    tax = load_taxonomy()
    plan = plan_upserts(tax, existing_by_name={})
    assert len(plan.inserts) == len(tax.skills)
    assert plan.updates == []
    assert plan.unchanged == []
    assert plan.orphaned == []
    # Every insert carries the canonical name + category from the taxonomy
    inserted = {s.name: s.category for s in plan.inserts}
    assert inserted["Python"] == "programming"
    assert inserted["PostgreSQL"] == "database"
    assert inserted["GIS"] == "gis"


def test_in_sync_db_is_noop():
    tax = load_taxonomy()
    existing = {skill.name: make_skill(skill.name, skill.category) for skill in tax.skills}
    plan = plan_upserts(tax, existing)
    assert plan.total_changes == 0
    assert len(plan.unchanged) == len(tax.skills)


def test_category_change_is_an_update():
    tax = load_taxonomy()
    # A stale row: React was stored under "frontend frameworks"
    existing = {"React": make_skill("React", "frontend frameworks")}
    plan = plan_upserts(tax, existing)
    assert [(s.name, s.category) for s in plan.updates] == [("React", "frontend")]
    # Everything else in the taxonomy is a new insert
    assert len(plan.inserts) == len(tax.skills) - 1


def test_mix_of_insert_update_unchanged():
    tax = load_taxonomy()
    existing = {
        "Python": make_skill("Python", "programming"),  # in sync
        "AWS": make_skill("AWS", "infra"),  # stale category
    }
    plan = plan_upserts(tax, existing)
    inserted = {s.name for s in plan.inserts}
    assert "Python" not in inserted
    assert len(plan.unchanged) == 1
    assert [(s.name, s.category) for s in plan.updates] == [("AWS", "cloud")]


def test_orphans_reported_not_touched():
    tax = load_taxonomy()
    existing = {"Python": make_skill("Python", "programming"), "COBOL": make_skill("COBOL", "programming")}
    plan = plan_upserts(tax, existing)
    assert plan.orphaned == ["COBOL"]
    assert "COBOL" not in [s.name for s in plan.inserts]
