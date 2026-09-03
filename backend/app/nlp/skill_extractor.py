"""
Skill Extractor — identifies skill mentions in CV text.

Uses a combination of:
1. Known skill dictionary matching
2. Pattern-based detection
3. Category classification
"""

import re
from dataclasses import dataclass


@dataclass
class ExtractedSkill:
    """A skill extracted from CV text."""
    name: str  # Normalized skill name
    raw_text: str  # Original text where skill was found
    category: str  # programming, frontend, backend, database, cloud, etc.
    confidence: float  # 0.0–1.0


# Known skills organized by category.
# Keys are canonical names; values are lists of aliases/variants.
KNOWN_SKILLS: dict[str, dict[str, list[str]]] = {
    "programming": {
        "Python": ["python", "python3", "python programming"],
        "JavaScript": ["javascript", "js", "ecmascript", "es6", "es2015"],
        "TypeScript": ["typescript", "ts"],
        "Java": ["java", "java se", "java ee"],
        "C++": ["c++", "cpp", "c plus plus"],
        "C#": ["c#", "csharp", "c sharp", ".net"],
        "Go": ["go", "golang"],
        "Rust": ["rust", "rustlang"],
        "Ruby": ["ruby", "ruby on rails"],
        "PHP": ["php"],
        "Swift": ["swift", "swiftui"],
        "Kotlin": ["kotlin"],
        "R": ["r programming", "r language", " r "],
        "Scala": ["scala"],
        "Perl": ["perl"],
        "Shell": ["shell", "bash", "zsh", "shell scripting"],
        "SQL": ["sql", "mysql", "postgresql", "plsql"],
    },
    "frontend": {
        "React": ["react", "reactjs", "react.js"],
        "Next.js": ["nextjs", "next.js", "next js"],
        "Vue.js": ["vue", "vuejs", "vue.js"],
        "Angular": ["angular", "angularjs"],
        "Svelte": ["svelte", "sveltekit"],
        "HTML": ["html", "html5"],
        "CSS": ["css", "css3", "scss", "sass", "less", "tailwind", "tailwindcss"],
        "jQuery": ["jquery"],
        "Redux": ["redux"],
        "GraphQL": ["graphql", "gql"],
    },
    "backend": {
        "Node.js": ["nodejs", "node.js", "node js", "node"],
        "Express": ["express", "expressjs", "express.js"],
        "FastAPI": ["fastapi", "fast api"],
        "Django": ["django"],
        "Flask": ["flask"],
        "Spring": ["spring", "spring boot", "springboot"],
        "ASP.NET": ["asp.net", "aspnet"],
        "Ruby on Rails": ["ruby on rails", "rails"],
        "Laravel": ["laravel"],
    },
    "database": {
        "PostgreSQL": ["postgresql", "postgres", "psql"],
        "MySQL": ["mysql"],
        "MongoDB": ["mongodb", "mongo"],
        "Redis": ["redis"],
        "Elasticsearch": ["elasticsearch", "elastic search"],
        "SQLite": ["sqlite", "sqlite3"],
        "DynamoDB": ["dynamodb", "dynamo db"],
        "Cassandra": ["cassandra"],
        "SQL": ["sql", "relational database", "rdbms"],
    },
    "cloud": {
        "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda", "cloudformation"],
        "Azure": ["azure", "microsoft azure", "azure devops"],
        "GCP": ["gcp", "google cloud", "google cloud platform"],
        "Heroku": ["heroku"],
        "Vercel": ["vercel"],
        "Netlify": ["netlify"],
    },
    "devops": {
        "Docker": ["docker", "dockerfile", "docker-compose"],
        "Kubernetes": ["kubernetes", "k8s", "kubectl"],
        "CI/CD": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
        "Terraform": ["terraform"],
        "Ansible": ["ansible"],
        "Jenkins": ["jenkins"],
        "GitHub Actions": ["github actions"],
        "Nginx": ["nginx"],
        "Linux": ["linux", "ubuntu", "debian", "centos"],
        "Git": ["git", "github", "gitlab", "bitbucket"],
    },
    "data_science": {
        "Machine Learning": ["machine learning", "ml", "deep learning"],
        "TensorFlow": ["tensorflow"],
        "PyTorch": ["pytorch"],
        "Pandas": ["pandas"],
        "NumPy": ["numpy", "numpy array"],
        "Scikit-learn": ["scikit-learn", "sklearn"],
        "Data Analysis": ["data analysis", "data analytics", "analytics"],
        "NLP": ["nlp", "natural language processing"],
        "Computer Vision": ["computer vision", "opencv"],
    },
    "tools": {
        "Jira": ["jira"],
        "Confluence": ["confluence"],
        "Figma": ["figma"],
        "Slack": ["slack"],
        "Notion": ["notion"],
        "VS Code": ["vs code", "visual studio code"],
    },
}


def _build_alias_lookup() -> dict[str, tuple[str, str]]:
    """Build a flat lookup: alias_lower → (canonical_name, category)."""
    lookup: dict[str, tuple[str, str]] = {}
    for category, skills in KNOWN_SKILLS.items():
        for canonical, aliases in skills.items():
            for alias in aliases:
                lookup[alias.lower().strip()] = (canonical, category)
    return lookup


# Module-level cached lookup
_ALIAS_LOOKUP = _build_alias_lookup()


def extract_skills(text: str) -> list[ExtractedSkill]:
    """
    Extract skill mentions from text.

    Scans the text for known skill aliases and returns
    deduplicated results with category and confidence.
    """
    text_lower = text.lower()
    seen: dict[str, ExtractedSkill] = {}

    for alias, (canonical, category) in _ALIAS_LOOKUP.items():
        # Require surrounding word boundaries so short aliases like "go" or
        # "r" never match inside unrelated words
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"

        if re.search(pattern, text_lower, re.IGNORECASE) and canonical not in seen:
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
        # Check if this item matches a known skill
        item_lower = item.lower().strip()
        if item_lower in _ALIAS_LOOKUP:
            canonical, category = _ALIAS_LOOKUP[item_lower]
            results.append(ExtractedSkill(
                name=canonical,
                raw_text=item,
                category=category,
                confidence=0.95,
            ))
        else:
            # Try partial matching
            skills = extract_skills(item)
            results.extend(skills)

    # Deduplicate
    seen = set()
    deduped = []
    for skill in results:
        if skill.name not in seen:
            seen.add(skill.name)
            deduped.append(skill)

    return deduped
