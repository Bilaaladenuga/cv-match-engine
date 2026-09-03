"""
Tests for Phase 5 — Skill Taxonomy.

Covers loading, alias resolution, canonicalization, data integrity, runtime
extension, and extraction of skills from the new categories (ML, GIS,
design, soft skills).
"""

import json

import pytest

from app.nlp.skill_extractor import extract_skills, extract_skills_from_list
from app.nlp.taxonomy import (
    DEFAULT_TAXONOMY_PATH,
    SkillTaxonomy,
    SkillTaxonomyError,
    load_taxonomy,
)

# Categories the spec requires the taxonomy to support
REQUIRED_CATEGORIES = {
    "programming",
    "frontend",
    "backend",
    "database",
    "cloud",
    "devops",
    "data_science",
    "machine_learning",
    "gis",
    "design",
    "soft_skills",
}


class TestTaxonomyLoading:
    def test_default_loads(self):
        tax = load_taxonomy()
        assert len(tax.skills) >= 100
        assert tax.schema_version

    def test_required_categories_present(self):
        tax = load_taxonomy()
        assert REQUIRED_CATEGORIES.issubset(set(tax.category_ids))

    def test_categories_have_labels(self):
        tax = load_taxonomy()
        for cat in tax.categories:
            assert cat.label, f"category {cat.id} missing label"

    def test_every_skill_has_at_least_one_alias(self):
        tax = load_taxonomy()
        for skill in tax.skills:
            assert len(skill.aliases) >= 1, f"{skill.name} has no aliases"
            assert skill.category in tax.category_ids, (
                f"{skill.name} references unknown category {skill.category}"
            )

    def test_ci_alias_collision_gate(self):
        """
        Gate that fails CI whenever the taxonomy file is ambiguous.

        Reads the raw JSON directly (not the loader) so *all* conflicting
        aliases are reported in one message instead of failing on the first.
        """
        with open(DEFAULT_TAXONOMY_PATH, encoding="utf-8") as f:
            data = json.load(f)

        claimed: dict[str, str] = {}
        collisions: list[str] = []
        for skill in data["skills"]:
            for key in [skill["name"]] + skill.get("aliases", []):
                norm = key.strip().lower()
                if norm in claimed and claimed[norm] != skill["name"]:
                    collisions.append(f"'{key}' -> both '{claimed[norm]}' and '{skill['name']}'")
                else:
                    claimed[norm] = skill["name"]

        assert not collisions, (
            "Ambiguous aliases in skill_taxonomy.json "
            "(each alias must map to exactly one canonical):\n  "
            + "\n  ".join(sorted(set(collisions)))
        )

    def test_round_trip(self):
        tax = load_taxonomy()
        reloaded = SkillTaxonomy(tax.to_dict())
        assert {s.name for s in reloaded.skills} == {s.name for s in tax.skills}
        assert reloaded.schema_version == tax.schema_version

    def test_unknown_category_in_json_fails_fast(self):
        # A typo like "dev-ops" must raise instead of silently creating a category
        tax = load_taxonomy()
        bad = tax.to_dict()
        bad["skills"].append(
            {"name": "Meson", "category": "dev-ops", "aliases": ["meson build"]}
        )
        with pytest.raises(SkillTaxonomyError, match="unknown category 'dev-ops'"):
            SkillTaxonomy(bad)


class TestAliasResolution:
    @pytest.fixture
    def tax(self):
        return load_taxonomy()

    def test_alias_to_canonical(self, tax):
        assert tax.resolve("js").name == "JavaScript"
        assert tax.resolve("postgres").name == "PostgreSQL"
        assert tax.resolve("reactjs").name == "React"
        assert tax.resolve("sklearn").name == "Scikit-learn"
        assert tax.resolve("k8s").name == "Kubernetes"

    def test_canonical_name_resolves_to_itself(self, tax):
        assert tax.resolve("Python").name == "Python"
        assert tax.resolve("AWS").name == "AWS"

    def test_case_insensitive(self, tax):
        assert tax.resolve("JS").name == "JavaScript"
        assert tax.resolve("ReAcT").name == "React"

    def test_canonicalize(self, tax):
        assert tax.canonicalize("JS") == "JavaScript"
        assert tax.canonicalize("Javascript") == "JavaScript"
        assert tax.canonicalize("Java Script") == "JavaScript"
        assert tax.canonicalize("Postgres") == "PostgreSQL"
        assert tax.canonicalize("python3") == "Python"

    def test_unknown_returns_none(self, tax):
        assert tax.resolve("quantum knitting") is None
        assert tax.canonicalize("quantum knitting") is None

    def test_category_assignment(self, tax):
        assert tax.resolve("AWS").category == "cloud"
        assert tax.resolve("React").category == "frontend"
        assert tax.resolve("PostgreSQL").category == "database"
        assert tax.resolve("TensorFlow").category == "machine_learning"
        assert tax.resolve("GIS").category == "gis"
        assert tax.resolve("Figma").category == "design"

    def test_sql_maps_to_database(self, tax):
        # Phase 4 cleanup: SQL is a database skill; MySQL stays distinct
        assert tax.canonicalize("SQL") == "SQL"
        assert tax.resolve("SQL").category == "database"
        assert tax.canonicalize("mysql") == "MySQL"

    def test_worked_example_skills_resolve(self, tax):
        # Regression coverage for the niche skills added as a worked example
        assert tax.canonicalize("celery") == "Celery"
        assert tax.canonicalize("rabbit mq") == "RabbitMQ"
        assert tax.canonicalize("rest apis") == "REST API"
        assert tax.resolve("REST API").category == "backend"
        assert tax.canonicalize("websockets") == "WebSocket"
        assert tax.canonicalize("jest") == "Jest"
        assert tax.canonicalize("py.test") == "pytest"
        assert tax.resolve("playwright").category == "tools"
        assert tax.resolve("cypress").category == "tools"
        assert (
            tax.resolve("aws certified solutions architect").name
            == "AWS Certified Solutions Architect"
        )
        assert tax.resolve("google cloud certified").category == "cloud"

    def test_ruby_on_rails_maps_to_framework(self, tax):
        assert tax.canonicalize("rails") == "Ruby on Rails"
        assert tax.canonicalize("ruby") == "Ruby"


class TestExtension:
    def test_register_new_skill(self):
        tax = SkillTaxonomy(load_taxonomy().to_dict())
        tax.register_skill("Phoenix", "backend", ["phoenix", "phoenix framework"])
        assert tax.resolve("phoenix framework").name == "Phoenix"
        assert tax.canonicalize("Phoenix") == "Phoenix"
        assert tax.resolve("Phoenix").category == "backend"

    def test_register_new_category_auto_creates(self):
        tax = SkillTaxonomy(load_taxonomy().to_dict())
        tax.register_skill("Tableau Prep", "data_engineering", ["tableau prep"])
        assert tax.get_category("data_engineering") is not None

    def test_merge_into_existing_skill(self):
        tax = SkillTaxonomy(load_taxonomy().to_dict())
        before = len(tax.skills)
        tax.register_skill("Python", "programming", ["monty"])
        assert len(tax.skills) == before
        assert tax.canonicalize("monty") == "Python"
        assert "monty" in tax.get_skill("Python").aliases

    def test_conflicting_alias_raises(self):
        tax = SkillTaxonomy(load_taxonomy().to_dict())
        with pytest.raises(SkillTaxonomyError):
            tax.register_skill("Something Else", "programming", ["js"])

    def test_duplicate_canonical_in_other_category_raises(self):
        tax = SkillTaxonomy(load_taxonomy().to_dict())
        with pytest.raises(SkillTaxonomyError):
            tax.register_skill("Python", "data_science", ["python (ds)"])


class TestExtractionFromTaxonomy:
    def test_machine_learning_skills(self):
        skills = extract_skills("I train models with tensorflow, keras and sklearn")
        names = [s.name for s in skills]
        assert "TensorFlow" in names
        assert "Keras" in names
        assert "Scikit-learn" in names
        assert all(s.category == "machine_learning" for s in skills)

    def test_gis_skills(self):
        skills = extract_skills("Built maps with qgis, postgis and leaflet")
        names = {s.name: s.category for s in skills}
        assert names.get("QGIS") == "gis"
        assert names.get("PostGIS") == "gis"
        assert names.get("Leaflet") == "gis"

    def test_design_skills(self):
        skills = extract_skills_from_list("Figma, Adobe Photoshop, Canva")
        names = {s.name for s in skills}
        assert names == {"Figma", "Adobe Photoshop", "Canva"}
        assert all(s.category == "design" for s in skills)

    def test_soft_skills(self):
        skills = extract_skills("Strong communication, leadership and agile delivery")
        names = {s.name: s.category for s in skills}
        assert names.get("Communication") == "soft_skills"
        assert names.get("Leadership") == "soft_skills"
        assert names.get("Agile") == "soft_skills"

    def test_single_letter_r_is_safe(self):
        # "R&D" must not be extracted as the R programming language
        skills = extract_skills("Led R&D for 2 years using Python")
        names = [s.name for s in skills]
        assert "R" not in names
        assert "Python" in names

    def test_short_alias_word_boundaries(self):
        # "ml" must not match inside "mlops", "go" not inside "gopher"
        # ("golang" is itself a Go alias and legitimately matches)
        skills = extract_skills("mlops pipelines and gopher services")
        names = [s.name for s in skills]
        assert "Machine Learning" not in names
        assert "Go" not in names

    def test_golang_alias_still_matches(self):
        skills = extract_skills("golang services")
        assert "Go" in [s.name for s in skills]

    def test_sample_cv_still_extracts(self):
        from pathlib import Path

        sample = (
            Path(__file__).resolve().parent.parent.parent / "data" / "sample" / "sample_cv.txt"
        )
        with open(sample, encoding="utf-8") as f:
            text = f.read()
        names = [s.name for s in extract_skills(text)]
        assert "Python" in names
        assert "React" in names
        assert "PostgreSQL" in names
        assert "Docker" in names
        assert "AWS" in names
        assert "Git" in names

    def test_sample_cv_extracts_new_backend_skills(self):
        # The sample CV lists "REST APIs" in skills and Celery in experience;
        # both should now be recognized via the taxonomy
        from pathlib import Path

        sample = (
            Path(__file__).resolve().parent.parent.parent / "data" / "sample" / "sample_cv.txt"
        )
        with open(sample, encoding="utf-8") as f:
            text = f.read()
        names = [s.name for s in extract_skills(text)]
        assert "REST API" in names
        assert "Celery" in names
