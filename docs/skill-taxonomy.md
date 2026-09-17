# Skill Taxonomy

The skill taxonomy is the canonical, machine-readable vocabulary of skills the
engine recognizes. It is the single source of truth shared by:

- the **skill extractor** (`backend/app/nlp/skill_extractor.py`)
- the future **job description parser** and **skill matching engine**
- the **`skills` database table** (populated by `scripts/seed_skills.py`)

## Where it lives

| Thing | Path |
|-------|------|
| Taxonomy data | `backend/app/nlp/skill_taxonomy.json` |
| Loader / API | `backend/app/nlp/taxonomy.py` (`SkillTaxonomy`) |
| Extractor | `backend/app/nlp/skill_extractor.py` |
| DB seeding | `backend/scripts/seed_skills.py` |

## File format

```json
{
  "schema_version": "1.0",
  "categories": {
    "database": {
      "label": "Database",
      "description": "SQL and NoSQL databases, query languages"
    }
  },
  "skills": [
    {
      "name": "PostgreSQL",
      "category": "database",
      "aliases": ["postgresql", "postgres", "psql"]
    }
  ]
}
```

| Field | Meaning |
|-------|---------|
| `schema_version` | Version of the taxonomy schema/data. Bump it when you make a *breaking* change (e.g. removing or renaming a canonical skill). |
| `categories` | Map of category id → `{label, description}`. A skill's `category` must reference one of these ids. |
| `skills[]` | One entry per canonical skill. |
| `skills[].name` | The canonical name — this is what appears in reports, the database, and match results. |
| `skills[].category` | Category id the skill belongs to (see table below). |
| `skills[].aliases` | Common spellings, abbreviations, and variants that should resolve to this skill. Matching is case-insensitive. |

### Category ids

Only these ids are valid in the JSON. A typo (e.g. `"dev-ops"`) makes loading
fail immediately rather than silently creating a bogus category.

| id | Label |
|----|-------|
| `programming` | Programming |
| `frontend` | Frontend |
| `backend` | Backend |
| `database` | Database |
| `cloud` | Cloud |
| `devops` | DevOps |
| `data_science` | Data Science |
| `machine_learning` | Machine Learning |
| `gis` | GIS |
| `design` | Design |
| `tools` | Tools |
| `soft_skills` | Soft Skills |
| `healthcare` | Healthcare |
| `finance` | Finance |
| `engineering` | Engineering |
| `law` | Law & Legal |
| `education` | Education & Training |
| `marketing` | Marketing |
| `sales` | Sales |
| `hr` | Human Resources |
| `media` | Media & Creative |
| `trades` | Skilled Trades |
| `logistics` | Logistics & Supply Chain |
| `science` | Laboratory Science |
| `hospitality` | Hospitality & Culinary |
| `public_sector` | Public Sector & Nonprofit |

## Adding a skill

Edit `skill_taxonomy.json` and add an entry. Example — adding Redis' caching
cousin Memcached under `database`:

```json
{
  "name": "Memcached",
  "category": "database",
  "aliases": ["memcached", "mem cache"]
}
```

Checklist:

1. **Canonical name** — the clean, human-facing name: `JavaScript` (never
   `JS`), `PostgreSQL` (never `postgres`), `Scikit-learn`. Keep it unique:
   two skills may not share a name, even across categories (SQL lives in
   `database` only — it is not also a `programming` skill).

> **Career-field coverage note.** The taxonomy now spans 26 categories
> covering IT, healthcare, finance, engineering, law, education, marketing,
> sales, HR, media, skilled trades, logistics, laboratory science,
> hospitality, and the public sector, plus cross-field tools
> (Microsoft Office, Google Workspace) and soft skills. New categories
> extend `cov_other` coverage until a model retrain adds dedicated
> `cov_<cat>` features — the serving schema is fixed, so expansion never
> breaks the trained model.
2. **Category** — pick one of the 12 ids above. Every entry needs one; a
   missing or unknown category raises an error on load.
3. **Aliases** — include the variants people actually write:
   - case variants: not needed (`python` and `Python` both match)
   - abbreviations: `js`, `k8s`, `sklearn`
   - alternate spellings: `reactjs` / `react.js`, `postgres`, `postgresql`
   - spaced forms: `node js`, `docker compose`
   - the canonical name itself does **not** need to be listed — it always
     matches.
4. **Never reuse an alias already claimed by another skill.** Aliases must
   map to exactly one canonical skill. If `react` is already an alias of
   `React`, you cannot also list it for `Vue.js` — loading raises an error
   telling you which skills conflict. Search the file first:
   ```bash
   cd backend && grep -i '"react"' app/nlp/skill_taxonomy.json
   ```
5. **Watch out for generic English words.** Whole-word matching means a soft
   skill like `Communication` will trigger on any CV that says
   "communication". For the same reason, avoid verbs that appear in prose
   (`leading`, `mentor` is fine but risky words like `design`, `sketch` are
   not) unless you are deliberately modeling a soft skill.
6. **Short aliases are safe but loud.** Word boundaries prevent `ml` from
   matching inside `mlops` and `go` inside `golang`, but any standalone
   1–2 letter token still matches, so only add them for genuinely common
   abbreviations. Single letters require whitespace isolation (an `r` alias
   does not match inside `R&D`).
7. **Phrase aliases that contain another alias** will match both skills
   (e.g. text "Ruby on Rails" yields both `Ruby` and `Ruby on Rails`). That
   is accepted behavior — prefer distinct aliases where practical.

## Removing or renaming a skill

- **Removing**: delete the entry. The extractor stops recognizing it. DB rows
  for skills that disappear from the taxonomy are reported as orphans by the
  seed script but are **not deleted**, because `candidate_skills` and
  `job_requirements` rows may reference them.
- **Renaming**: add the new entry and delete the old one, then bump
  `schema_version`. The seed script will insert the new row and stop managing
  the old one.
- After either, re-run the seeder if you keep the `skills` table in sync:
  ```bash
  cd backend && python -m scripts.seed_skills
  ```

## Verifying a change

```bash
cd backend
source venv/Scripts/activate        # Windows: venv\Scripts\activate

# 1. Load + integrity checks (duplicate names, stolen aliases, bad categories)
python -m pytest tests/test_taxonomy.py -q

# 2. Sanity-check resolution of your new alias
python -c "from app.nlp.taxonomy import load_taxonomy; print(load_taxonomy().canonicalize('memcache'))"

# 3. Full suite + lint
python -m pytest tests/ -q
ruff check app/ tests/ scripts/

# 4. Sync the database (inserts new skills only)
python -m scripts.seed_skills
```

**Restart the backend** after editing the JSON: the extractor builds its
alias lookup once at import time.

## Alternatives to editing the JSON

**Runtime registration** — for skills discovered dynamically (e.g. a niche
tool from a specific job ad) without touching the data file. Unknown
categories are allowed here and auto-created:

```python
from app.nlp.taxonomy import TAXONOMY

TAXONOMY.register_skill("Meson", "tools", ["meson", "meson build"])
```

**Custom taxonomy file** — load a different taxonomy entirely (useful for
domain-specific vocabularies) by pointing the loader at it:

```bash
# via environment variable
export SKILL_TAXONOMY_PATH=/path/to/my_taxonomy.json

# or in code
from app.nlp.taxonomy import load_taxonomy
tax = load_taxonomy("/path/to/my_taxonomy.json")
```

Keep in mind the extractor (`skill_extractor.py`) uses the bundled default
taxonomy; a custom file affects code that calls `load_taxonomy(path)`
explicitly.
