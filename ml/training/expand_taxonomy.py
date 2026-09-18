"""
Expand skill taxonomy with more natural language aliases.

Adds common phrases used in CVs for each field so the skill extractor
can detect skills from natural language descriptions.
"""

import json
from pathlib import Path

TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "backend" / "app" / "nlp" / "skill_taxonomy.json"

# Additional aliases for each skill (key = canonical name, value = list of new aliases)
EXTRA_ALIASES = {
    # Healthcare
    "Patient Care": [
        "patient intake", "patient care", "patient interaction",
        "patient support", "patient assistance", "bedside manner",
        "patient advocacy", "patient education", "patient comfort",
    ],
    "Clinical Documentation": [
        "documentation", "charting", "medical notes",
        "patient records", "clinical notes", "charting",
    ],
    "Medical Terminology": [
        "medical terms", "healthcare vocabulary", "clinical language",
    ],
    "Phlebotomy": [
        "blood draw", "blood collection", "specimen collection",
        "venipuncture", "blood sampling",
    ],
    "Medication Administration": [
        "medication", "med administration", "drug administration",
        "medication management", "pharmaceutical",
    ],
    "Care Planning": [
        "care plans", "treatment planning", "care coordination",
        "patient care planning", "discharge planning",
    ],
    "Triage": [
        "triage", "patient triage", "emergency assessment",
        "acuity assessment", "priority assessment",
    ],
    "Electronic Health Records": [
        "ehr", "emr", "electronic health record", "electronic medical record",
        "epic", "cerner", "health information", "medical records",
    ],
    "ICU Care": [
        "icu", "intensive care", "critical care", "critical care nursing",
        "life support", "ventilator management",
    ],
    "Infection Control": [
        "infection prevention", "infection prevention and control",
        "infection control", "ipc", "sterile technique", "aseptic technique",
    ],
    "Clinical Research": [
        "clinical trials", "clinical research", "research study",
        "clinical study", "research protocol", "gcp",
    ],
    "Medical Coding": [
        "medical coding", "icd-10", "cpt", "billing codes",
        "insurance coding", "diagnosis coding", "procedure coding",
    ],
    
    # Finance
    "Financial Analysis": [
        "financial analysis", "financial modeling", "financial assessment",
        "financial evaluation", "financial review", "financial planning",
        "budget analysis", "cost analysis", "variance analysis",
    ],
    "Financial Reporting": [
        "financial reporting", "management reporting", "financial statements",
        "reporting", "financial summaries", "board reporting",
    ],
    "Bookkeeping": [
        "bookkeeping", "double-entry bookkeeping", "ledger management",
        "accounts payable", "accounts receivable", "journal entries",
    ],
    "Account Reconciliation": [
        "reconciliation", "bank reconciliation", "account reconciliation",
        "reconciling", "balance reconciliation",
    ],
    "Budgeting & Forecasting": [
        "budgeting", "forecasting", "budget management", "financial forecasting",
        "budget planning", "budget control", "cost forecasting",
    ],
    "Tax Preparation": [
        "tax returns", "taxation", "tax preparation", "tax filing",
        "tax compliance", "income tax", "tax planning",
    ],
    "Auditing": [
        "audit", "auditing", "internal audit", "external audit",
        "audit management", "compliance audit", "financial audit",
    ],
    "Risk Analysis": [
        "risk assessment", "risk management", "risk analysis",
        "risk evaluation", "risk mitigation", "enterprise risk",
    ],
    "Financial Modeling": [
        "financial modeling", "financial modelling", "dcf",
        "discounted cash flow", "excel modeling", "valuation modeling",
    ],
    "GAAP": [
        "us gaap", "gaap", "generally accepted accounting principles",
        "gaap compliance", "gaap standards",
    ],
    "IFRS": [
        "ifrs", "international financial reporting standards",
        "ifrs standards", "ifrs compliance",
    ],
    "QuickBooks": [
        "quickbooks", "quickbooks online", "quickbooks desktop",
        "quickbooks pro",
    ],
    "Portfolio Management": [
        "portfolio management", "asset management", "investment management",
        "portfolio allocation", "investment portfolio",
    ],
    "Credit Analysis": [
        "credit analysis", "credit risk", "credit risk assessment",
        "credit evaluation", "credit review", "lending analysis",
    ],
    
    # Marketing
    "SEO": [
        "seo", "search engine optimization", "organic search",
        "keyword research", "on-page seo", "technical seo",
    ],
    "SEM": [
        "sem", "search engine marketing", "paid search",
        "ppc", "pay-per-click", "google ads", "adwords",
    ],
    "Google Analytics": [
        "google analytics", "analytics", "web analytics",
        "ga4", "traffic analysis", "conversion tracking",
    ],
    "Content Marketing": [
        "content marketing", "content strategy", "content creation",
        "content planning", "editorial calendar", "blog management",
    ],
    "Social Media Marketing": [
        "social media", "social media marketing", "social media management",
        "community management", "social media strategy", "social media ads",
    ],
    "Email Marketing": [
        "email marketing", "email campaigns", "newsletter management",
        "email automation", "drip campaigns", "email newsletters",
    ],
    "Brand Management": [
        "brand management", "brand strategy", "brand development",
        "brand identity", "brand guidelines", "brand positioning",
    ],
    "Marketing Strategy": [
        "marketing strategy", "marketing planning", "go-to-market",
        "gtm", "marketing campaigns", "marketing programs",
    ],
    "Copywriting": [
        "copywriting", "copy writing", "advertising copy",
        "creative writing", "persuasive writing", "marketing copy",
    ],
    "Market Research": [
        "market research", "competitive analysis", "market analysis",
        "consumer research", "industry analysis", "market intelligence",
    ],
    "Public Relations": [
        "public relations", "pr", "media relations", "press releases",
        "crisis communications", "media outreach",
    ],
    "Google Ads": [
        "google ads", "google adwords", "ppc campaigns",
        "paid advertising", "display advertising",
    ],
    
    # HR
    "Recruiting": [
        "recruiting", "recruitment", "talent acquisition", "hiring",
        "sourcing", "interviewing", "candidate screening",
    ],
    "Talent Acquisition": [
        "talent acquisition", "talent sourcing", "talent recruitment",
        "full-cycle recruiting", "executive recruiting",
    ],
    "Onboarding": [
        "onboarding", "employee onboarding", "new hire orientation",
        "onboarding programs", "orientation programs",
    ],
    "Payroll Administration": [
        "payroll", "payroll administration", "payroll processing",
        "payroll management", "compensation administration",
    ],
    "Performance Management": [
        "performance management", "performance reviews", "performance evaluations",
        "goal setting", "performance improvement", "kpi management",
    ],
    "Employee Relations": [
        "employee relations", "workplace relations", "conflict resolution",
        "employee engagement", "employee retention",
    ],
    "Compensation & Benefits": [
        "compensation", "benefits", "compensation and benefits",
        "total rewards", "benefits administration", "compensation analysis",
    ],
    "Training & Development": [
        "training", "training and development", "learning and development",
        "employee training", "professional development", "workshop facilitation",
    ],
    
    # Sales
    "CRM Software": [
        "crm", "crm software", "customer relationship management",
        "salesforce", "hubspot", "crm systems",
    ],
    "Salesforce": [
        "salesforce", "salesforce crm", "salesforce administrator",
        "salesforce platform",
    ],
    "Lead Generation": [
        "lead generation", "lead gen", "prospecting", "pipeline development",
        "lead qualification", "lead nurturing",
    ],
    "Negotiation": [
        "negotiation", "contract negotiation", "deal negotiation",
        "price negotiation", "sales negotiation",
    ],
    "Account Management": [
        "account management", "key account management", "client management",
        "customer success", "account retention",
    ],
    "Sales Strategy": [
        "sales strategy", "sales planning", "revenue strategy",
        "sales forecasting", "territory management",
    ],
    "Business Development": [
        "business development", "biz dev", "partnership development",
        "strategic partnerships", "alliance management",
    ],
    "Sales Presentations": [
        "sales presentations", "presentations", "demo", "product demo",
        "sales demos", "client presentations",
    ],
    
    # Education
    "Curriculum Development": [
        "curriculum development", "curriculum design", "course development",
        "instructional design", "syllabus development",
    ],
    "Lesson Planning": [
        "lesson planning", "lesson plans", "unit planning",
        "instructional planning", "teaching plans",
    ],
    "Classroom Management": [
        "classroom management", "classroom control", "student management",
        "behavior management", "classroom discipline",
    ],
    "Student Assessment": [
        "student assessment", "assessment", "grading", "evaluation",
        "test development", "rubric development",
    ],
    "Teaching": [
        "teaching", "instruction", "educating", "lecturing",
        "tutoring", "mentoring",
    ],
    
    # Legal
    "Legal Research": [
        "legal research", "case research", "legal analysis",
        "legal investigation", "legal writing", "brief writing",
    ],
    "Contract Law": [
        "contract law", "contract drafting", "contract review",
        "contract management", "agreement review",
    ],
    "Litigation": [
        "litigation", "trial preparation", "trial advocacy",
        "court proceedings", "dispute resolution",
    ],
    "Compliance": [
        "compliance", "regulatory compliance", "compliance management",
        "compliance monitoring", "audit compliance",
    ],
    "Legal Writing": [
        "legal writing", "legal drafting", "memo writing",
        "brief writing", "legal documentation",
    ],
    
    # Engineering
    "Civil Engineering": [
        "civil engineering", "structural engineering", "construction engineering",
        "transportation engineering", "geotechnical engineering",
    ],
    "Mechanical Engineering": [
        "mechanical engineering", "machine design", "thermodynamics",
        "fluid mechanics", "manufacturing engineering",
    ],
    "Electrical Engineering": [
        "electrical engineering", "circuit design", "power systems",
        "control systems", "electronics engineering",
    ],
    "AutoCAD": [
        "autocad", "cad", "computer-aided design", "technical drawing",
        "drafting",
    ],
    "SolidWorks": [
        "solidworks", "3d modeling", "cad design", "solid modeling",
    ],
    
    # Trades
    "Welding": [
        "welding", "arc welding", "mig welding", "tig welding",
        "certified welder", "welding techniques",
    ],
    "Plumbing": [
        "plumbing", "pipe fitting", "pipe installation",
        "plumbing repair", "plumbing systems",
    ],
    "Electrical Wiring": [
        "electrical wiring", "wiring", "electrical installation",
        "electrical systems", "electrical repair",
    ],
    "Carpentry": [
        "carpentry", "woodworking", "finish carpentry",
        "rough carpentry", "cabinet making",
    ],
    "HVAC": [
        "hvac", "heating ventilation and air conditioning",
        "hvac systems", "hvac installation", "hvac maintenance",
    ],
    
    # Logistics
    "Supply Chain Management": [
        "supply chain", "supply chain management", "scm",
        "logistics management", "procurement",
    ],
    "Warehouse Operations": [
        "warehouse operations", "warehouse management", "inventory control",
        "warehouse logistics", "distribution center",
    ],
    "Inventory Management": [
        "inventory management", "inventory control", "stock management",
        "inventory planning", "materials management",
    ],
    "Route Planning": [
        "route planning", "route optimization", "dispatching",
        "fleet management", "delivery planning",
    ],
    
    # Hospitality
    "Food Safety": [
        "food safety", "food handling", "food sanitation",
        "haccp", "food hygiene", "food safety certification",
    ],
    "Culinary Arts": [
        "culinary arts", "cooking", "food preparation",
        "culinary skills", "chef", "kitchen management",
    ],
    "Hotel Management": [
        "hotel management", "hospitality management", "front desk",
        "guest services", "hotel operations",
    ],
    
    # Public Sector
    "Grant Writing": [
        "grant writing", "grant proposals", "grant applications",
        "funding proposals", "grant management",
    ],
    "Community Outreach": [
        "community outreach", "community engagement", "community organizing",
        "community relations", "public engagement",
    ],
    "Public Policy": [
        "public policy", "policy analysis", "policy development",
        "government affairs", "legislative analysis",
    ],
    
    # Science
    "Laboratory Techniques": [
        "laboratory techniques", "lab techniques", "wet lab",
        "laboratory skills", "lab procedures",
    ],
    "PCR": [
        "pcr", "polymerase chain reaction", "pcr testing",
        "molecular diagnostics",
    ],
    "Cell Culture": [
        "cell culture", "tissue culture", "cell biology",
        "mammalian cell culture",
    ],
    
    # Soft Skills
    "Leadership": [
        "leadership", "team leadership", "leadership skills",
        "people management", "team management",
    ],
    "Communication": [
        "communication", "verbal communication", "written communication",
        "interpersonal communication", "presentation skills",
    ],
    "Problem Solving": [
        "problem solving", "problem-solving", "analytical thinking",
        "troubleshooting", "root cause analysis",
    ],
    "Project Management": [
        "project management", "project planning", "project coordination",
        "project delivery", "project leadership",
    ],
}


def expand_taxonomy():
    """Add extra aliases to the taxonomy."""
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    skills = data.get("skills", [])
    updated = 0
    
    for skill in skills:
        name = skill["name"]
        if name in EXTRA_ALIASES:
            existing = set(skill.get("aliases", []))
            new_aliases = [a for a in EXTRA_ALIASES[name] if a.lower() not in {e.lower() for e in existing}]
            if new_aliases:
                skill["aliases"] = list(existing) + new_aliases
                updated += 1
    
    with open(TAXONOMY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Updated {updated} skills with new aliases")
    print(f"Total skills: {len(skills)}")


if __name__ == "__main__":
    expand_taxonomy()
