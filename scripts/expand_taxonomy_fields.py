"""One-shot migration: expand the skill taxonomy to 26 career-field categories.

Idempotency and collision safety: refuses to run if any new skill name or
alias collides with existing or intra-batch entries, and exits without
writing in that case. Run:  python scripts/expand_taxonomy_fields.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "backend" / "app" / "nlp" / "skill_taxonomy.json"

NEW_CATEGORIES = {
    "engineering": {"label": "Engineering", "description": "Civil, mechanical, electrical and general engineering skills and CAD/CAE tools"},
    "law": {"label": "Law & Legal", "description": "Legal practice, compliance and regulatory skills"},
    "education": {"label": "Education & Training", "description": "Teaching, curriculum and instructional skills"},
    "marketing": {"label": "Marketing", "description": "Digital and traditional marketing, SEO and content skills"},
    "sales": {"label": "Sales", "description": "Sales, account management and business development skills"},
    "hr": {"label": "Human Resources", "description": "Recruiting, people operations and HR systems"},
    "media": {"label": "Media & Creative", "description": "Video, audio, photography and journalism skills"},
    "trades": {"label": "Skilled Trades", "description": "Construction and trades: welding, plumbing, electrical, carpentry, HVAC"},
    "logistics": {"label": "Logistics & Supply Chain", "description": "Supply chain, warehousing, procurement and fleet skills"},
    "science": {"label": "Laboratory Science", "description": "Wet-lab and analytical laboratory techniques"},
    "hospitality": {"label": "Hospitality & Culinary", "description": "Food service, hotel operations and event skills"},
    "public_sector": {"label": "Public Sector & Nonprofit", "description": "Policy, community programs and public administration skills"},
}

NEW_SKILLS = [
    # engineering
    ("Civil Engineering", "engineering", ["civil eng", "infrastructure engineering"]),
    ("Mechanical Engineering", "engineering", ["mech eng", "mechanical design"]),
    ("Electrical Engineering", "engineering", ["electrical eng", "power systems"]),
    ("AutoCAD", "engineering", ["autodesk autocad", "autocad drafting"]),
    ("SolidWorks", "engineering", ["solid works", "solidworks cad"]),
    ("MATLAB", "engineering", ["matlab programming", "matlab/simulink"]),
    ("Revit", "engineering", ["autodesk revit", "revit bim"]),
    ("Structural Analysis", "engineering", ["structural design", "structural engineering analysis"]),
    ("Finite Element Analysis", "engineering", ["fea", "finite element modelling", "finite element modeling"]),
    ("PLC Programming", "engineering", ["plc", "siemens s7", "industrial automation"]),
    ("Engineering Drawing", "engineering", ["technical drawing", "engineering graphics"]),
    ("Thermodynamics", "engineering", ["heat transfer", "fluid mechanics"]),
    # law
    ("Legal Research", "law", ["legal analysis", "westlaw", "lexisnexis"]),
    ("Legal Writing", "law", ["legal drafting", "brief writing"]),
    ("Contract Law", "law", ["contract drafting", "contracts"]),
    ("Litigation", "law", ["litigator", "court proceedings"]),
    ("Compliance", "law", ["regulatory compliance", "compliance management"]),
    ("Due Diligence", "law", ["dd review", "transactional due diligence"]),
    ("Case Management", "law", ["case management systems", "docketing"]),
    ("Corporate Law", "law", ["company law", "corporate governance"]),
    ("Intellectual Property", "law", ["ip law", "patents", "trademarks"]),
    ("Regulatory Affairs", "law", ["regulatory submissions", "regulatory strategy"]),
    # education
    ("Curriculum Development", "education", ["curriculum planning", "curriculum design"]),
    ("Lesson Planning", "education", ["lesson plans", "unit planning"]),
    ("Classroom Management", "education", ["behaviour management", "behavior management"]),
    ("Student Assessment", "education", ["grading", "formative assessment"]),
    ("Teaching", "education", ["teacher", "lecturing", "instruction"]),
    ("Tutoring", "education", ["tutor", "one-on-one instruction"]),
    ("Educational Technology", "education", ["edtech", "google classroom", "moodle"]),
    ("Special Education", "education", ["send", "special needs education", "iep"]),
    ("Academic Advising", "education", ["student advising", "student mentoring"]),
    ("Instructional Design", "education", ["instructional designer", "e-learning development"]),
    # marketing
    ("SEO", "marketing", ["search engine optimization", "search engine optimisation"]),
    ("SEM", "marketing", ["search engine marketing", "paid search"]),
    ("Google Analytics", "marketing", ["ga4", "universal analytics"]),
    ("Content Marketing", "marketing", ["content strategy", "content creation"]),
    ("Social Media Marketing", "marketing", ["social media management", "smm"]),
    ("Email Marketing", "marketing", ["mailchimp", "email campaigns"]),
    ("Copywriting", "marketing", ["copy writing", "content writing"]),
    ("Brand Management", "marketing", ["branding", "brand strategy"]),
    ("Market Research", "marketing", ["market analysis", "marketing research"]),
    ("Digital Advertising", "marketing", ["paid media", "programmatic advertising"]),
    ("Google Ads", "marketing", ["adwords", "google adwords"]),
    ("Meta Ads", "marketing", ["facebook ads", "instagram ads"]),
    # sales
    ("CRM Software", "sales", ["crm", "crm systems"]),
    ("Salesforce", "sales", ["salesforce crm", "sfdc"]),
    ("Lead Generation", "sales", ["lead gen", "leadgen"]),
    ("Negotiation", "sales", ["negotiating", "contract negotiation"]),
    ("Account Management", "sales", ["key account management", "account manager"]),
    ("Sales Forecasting", "sales", ["pipeline management", "sales projections"]),
    ("B2B Sales", "sales", ["business-to-business sales", "b2b"]),
    ("Client Relationship Management", "sales", ["client relations", "relationship management"]),
    # hr
    ("Recruiting", "hr", ["recruitment", "recruiter", "technical recruiting"]),
    ("Talent Acquisition", "hr", ["talent sourcing", "candidate sourcing"]),
    ("Onboarding", "hr", ["employee onboarding", "new hire orientation"]),
    ("Payroll Administration", "hr", ["payroll", "payroll processing"]),
    ("Performance Management", "hr", ["appraisals", "performance reviews"]),
    ("Employee Relations", "hr", ["er", "grievance handling"]),
    ("HRIS", "hr", ["hr information systems", "workday", "bamboohr"]),
    ("Training & Development", "hr", ["training and development", "staff training"]),
    # media
    ("Video Editing", "media", ["video editor", "video post-production"]),
    ("Adobe Premiere Pro", "media", ["premiere pro", "premiere"]),
    ("Final Cut Pro", "media", ["fcp", "final cut"]),
    ("Adobe After Effects", "media", ["after effects", "ae"]),
    ("Photography", "media", ["photographer", "photo shooting"]),
    ("Podcast Production", "media", ["podcasting", "podcast editing"]),
    ("Journalism", "media", ["journalist", "news writing"]),
    ("Screenwriting", "media", ["script writing", "scriptwriting"]),
    ("Audio Editing", "media", ["audio post-production", "audio mixing"]),
    ("Sound Design", "media", ["sound designer", "audio design"]),
    # trades
    ("Welding", "trades", ["mig welding", "tig welding", "arc welding"]),
    ("Plumbing", "trades", ["plumber", "pipefitting"]),
    ("Electrical Wiring", "trades", ["wiring", "electrician"]),
    ("Carpentry", "trades", ["carpenter", "woodworking"]),
    ("Masonry", "trades", ["bricklaying", "stonemasonry"]),
    ("HVAC", "trades", ["heating, ventilation and air conditioning", "heating and air conditioning", "climate control"]),
    ("Construction Management", "trades", ["construction project management", "site management"]),
    ("Blueprint Reading", "trades", ["blueprints", "reading blueprints"]),
    ("Forklift Operation", "trades", ["forklift", "forklift certified"]),
    ("Scaffolding", "trades", ["scaffold erection", "scaffolder"]),
    # logistics
    ("Supply Chain Management", "logistics", ["supply chain", "scm"]),
    ("Warehouse Operations", "logistics", ["warehouse management", "warehousing"]),
    ("Inventory Management", "logistics", ["inventory control", "stock control"]),
    ("Freight Forwarding", "logistics", ["freight", "customs brokerage"]),
    ("Route Planning", "logistics", ["route optimization", "route optimisation"]),
    ("Procurement", "logistics", ["purchasing", "vendor management"]),
    ("Fleet Management", "logistics", ["fleet operations", "vehicle fleet"]),
    ("Demand Planning", "logistics", ["demand forecasting", "s&op"]),
    ("Customs Compliance", "logistics", ["customs", "import/export compliance"]),
    # science
    ("Laboratory Techniques", "science", ["lab techniques", "bench work"]),
    ("PCR", "science", ["polymerase chain reaction", "qpcr"]),
    ("Cell Culture", "science", ["cell culturing", "mammalian cell culture"]),
    ("Chromatography", "science", ["hplc", "gc-ms"]),
    ("Spectroscopy", "science", ["nmr", "mass spectrometry"]),
    ("Microscopy", "science", ["light microscopy", "electron microscopy"]),
    ("Experimental Design", "science", ["doe", "experiment design"]),
    ("Data Collection", "science", ["field data collection", "survey administration"]),
    ("Wet Lab", "science", ["wet-lab", "wet laboratory"]),
    # hospitality
    ("Food Safety", "hospitality", ["food hygiene", "haccp"]),
    ("Culinary Arts", "hospitality", ["cooking", "chef", "culinary"]),
    ("Hotel Management", "hospitality", ["hospitality management", "front office management"]),
    ("Bartending", "hospitality", ["bartender", "mixology"]),
    ("Barista", "hospitality", ["coffee preparation", "espresso"]),
    ("Guest Services", "hospitality", ["guest relations", "front desk"]),
    ("Housekeeping Management", "hospitality", ["housekeeping", "housekeeping operations"]),
    ("Event Planning", "hospitality", ["event management", "event coordination"]),
    ("Catering", "hospitality", ["caterer", "banquet service"]),
    ("Food & Beverage Service", "hospitality", ["f&b service", "food and beverage"]),
    # public sector
    ("Public Policy", "public_sector", ["policy analysis", "policy development"]),
    ("Grant Writing", "public_sector", ["grant proposals", "grantwriting"]),
    ("Community Outreach", "public_sector", ["community engagement", "outreach programs"]),
    ("Case Work", "public_sector", ["casework", "client casework"]),
    ("Urban Planning", "public_sector", ["town planning", "city planning"]),
    ("Emergency Management", "public_sector", ["disaster management", "emergency preparedness"]),
    ("Legislative Analysis", "public_sector", ["legislative affairs", "bill analysis"]),
    ("Public Administration", "public_sector", ["public sector administration", "government administration"]),
    # cross-field additions to existing categories
    ("Customer Service", "soft_skills", ["client service", "customer support", "customer care"]),
    ("Public Speaking", "soft_skills", ["presentations", "presentation skills"]),
    ("Microsoft Excel", "tools", ["excel", "ms excel", "spreadsheets"]),
    ("Microsoft Word", "tools", ["word", "ms word"]),
    ("PowerPoint", "tools", ["ms powerpoint", "slides"]),
    ("Microsoft Office", "tools", ["ms office", "office suite"]),
    ("Google Workspace", "tools", ["g suite", "google docs"]),
]


def main() -> None:
    tax = json.loads(PATH.read_text(encoding="utf-8"))

    existing_names = {s["name"] for s in tax["skills"]}
    alias_map: dict[str, str] = {}
    for s in tax["skills"]:
        alias_map[s["name"].lower()] = s["name"]
        for a in s["aliases"]:
            alias_map[a.lower()] = s["name"]

    # Idempotency: skip entries that already exist.
    pending = [e for e in NEW_SKILLS if e[0] not in existing_names]
    if not pending:
        print("Nothing to add — taxonomy already expanded.")
        return

    errors = []
    for name, _cat, aliases in pending:
        all_aliases = {name.lower(), *{a.lower() for a in aliases}}
        for al in all_aliases:
            if al in alias_map and alias_map[al] != name:
                errors.append(f"alias {al!r} of {name!r} already maps to {alias_map[al]!r}")
        for al in all_aliases:
            alias_map.setdefault(al, name)  # catch intra-batch collisions too

    if errors:
        print("COLLISIONS — nothing written:")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)

    tax["categories"].update(NEW_CATEGORIES)
    for name, cat, aliases in pending:
        als = list(aliases)
        if name.lower() not in als:
            als.insert(0, name.lower())
        tax["skills"].append({"name": name, "category": cat, "aliases": als})

    out = {
        "schema_version": tax["schema_version"],
        "categories": dict(sorted(tax["categories"].items())),
        "skills": tax["skills"],
    }
    PATH.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    by_cat = Counter(s["category"] for s in out["skills"])
    print(f"OK: {len(out['skills'])} skills across {len(out['categories'])} categories")
    for c in sorted(NEW_CATEGORIES):
        print(f"  {c}: {by_cat[c]} skills")


if __name__ == "__main__":
    main()
