"""
CandidateProfileBuilder — orchestrates all extraction modules to build
a structured candidate profile from raw CV text.
"""

from dataclasses import dataclass, field

from app.nlp.certification_extractor import Certification, extract_certifications
from app.nlp.education_extractor import EducationEntry, extract_education
from app.nlp.experience_extractor import (
    WorkExperience,
    calculate_total_years_experience,
    extract_experience,
)
from app.nlp.skill_extractor import ExtractedSkill, extract_skills, extract_skills_from_list
from app.nlp.title_extractor import extract_titles
from app.parsers.section_detector import DetectedSections, detect_sections


@dataclass
class CandidateProfile:
    """Complete structured profile extracted from a CV."""

    # Identity
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    linkedin: str | None = None
    github: str | None = None

    # Summary
    summary: str | None = None

    # Skills
    skills: list[ExtractedSkill] = field(default_factory=list)

    # Experience
    work_experience: list[WorkExperience] = field(default_factory=list)
    total_years_experience: float | None = None

    # Titles
    job_titles: list[str] = field(default_factory=list)

    # Education
    education: list[EducationEntry] = field(default_factory=list)

    # Certifications
    certifications: list[Certification] = field(default_factory=list)

    # Raw data for debugging
    raw_text: str = ""
    sections: DetectedSections | None = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "linkedin": self.linkedin,
            "github": self.github,
            "summary": self.summary,
            "skills": [
                {"name": s.name, "category": s.category, "confidence": s.confidence}
                for s in self.skills
            ],
            "work_experience": [
                {
                    "company": e.company,
                    "role": e.role,
                    "start_date": e.start_date,
                    "end_date": e.end_date,
                    "duration_months": e.duration_months,
                    "duration_years": e.duration_years,
                }
                for e in self.work_experience
            ],
            "total_years_experience": self.total_years_experience,
            "job_titles": self.job_titles,
            "education": [
                {
                    "degree": e.degree,
                    "field_of_study": e.field_of_study,
                    "institution": e.institution,
                    "year": e.year,
                }
                for e in self.education
            ],
            "certifications": [
                {"name": c.name, "issuer": c.issuer, "year": c.year}
                for c in self.certifications
            ],
        }


def build_candidate_profile(cv_text: str) -> CandidateProfile:
    """
    Build a complete candidate profile from raw CV text.

    Pipeline:
    1. Parse sections (contact, skills, experience, education, etc.)
    2. Extract skills
    3. Extract work experience
    4. Extract education
    5. Extract certifications
    6. Extract job titles
    7. Calculate total experience
    """
    # Step 1: Detect sections
    sections = detect_sections(cv_text)

    # Step 2: Extract skills from skills section and full text
    skills: list[ExtractedSkill] = []

    skills_section = sections.get_section_text("skills")
    if skills_section:
        skills.extend(extract_skills_from_list(skills_section))

    # Also check full text for skills not in skills section
    full_text_skills = extract_skills(cv_text)
    seen_names = {s.name for s in skills}
    for s in full_text_skills:
        if s.name not in seen_names:
            skills.append(s)

    # Step 3: Extract work experience
    exp_section = sections.get_section_text("experience")
    work_experience: list[WorkExperience] = []
    if exp_section:
        work_experience = extract_experience(exp_section)

    total_years = calculate_total_years_experience(work_experience)

    # Step 4: Extract education
    edu_section = sections.get_section_text("education")
    education: list[EducationEntry] = []
    if edu_section:
        education = extract_education(edu_section)

    # Step 5: Extract certifications
    cert_section = sections.get_section_text("certifications")
    certifications: list[Certification] = []
    if cert_section:
        certifications = extract_certifications(cert_section)

    # Step 6: Extract job titles
    job_titles = extract_titles(cv_text)

    # Step 7: Build profile
    profile = CandidateProfile(
        name=sections.name,
        email=sections.contact.emails[0] if sections.contact.emails else None,
        phone=sections.contact.phones[0] if sections.contact.phones else None,
        linkedin=sections.contact.linkedin,
        github=sections.contact.github,
        summary=sections.summary,
        skills=skills,
        work_experience=work_experience,
        total_years_experience=total_years,
        job_titles=job_titles,
        education=education,
        certifications=certifications,
        raw_text=cv_text,
        sections=sections,
    )

    return profile
