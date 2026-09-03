"""
Skill and CandidateSkill models — normalized skill taxonomy.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Skill(Base):
    """Normalized skill in the taxonomy."""

    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    category: Mapped[str] = mapped_column(
        String(100), nullable=False, default="general"
    )  # programming, frontend, backend, database, cloud, devops, etc.

    # Relationships
    candidates = relationship("CandidateSkill", back_populates="skill")
    job_requirements = relationship("JobRequirement", back_populates="skill")

    def __repr__(self) -> str:
        return f"<Skill id={self.id} name={self.name!r} category={self.category!r}>"


class CandidateSkill(Base):
    """Association between a candidate and a skill, with metadata."""

    __tablename__ = "candidate_skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    candidate_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("candidate_profiles.id", ondelete="CASCADE"), nullable=False
    )
    skill_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("skills.id", ondelete="CASCADE"), nullable=False
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0.0–1.0
    years_experience: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    candidate = relationship("CandidateProfile", back_populates="skills")
    skill = relationship("Skill", back_populates="candidates")

    def __repr__(self) -> str:
        return (
            f"<CandidateSkill candidate_id={self.candidate_id} "
            f"skill_id={self.skill_id} confidence={self.confidence}>"
        )
