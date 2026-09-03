"""
CandidateProfile model — structured data extracted from a CV.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class CandidateProfile(Base):
    __tablename__ = "candidate_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    resume_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("resumes.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    years_experience: Mapped[float | None] = mapped_column(nullable=True)
    education: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    extracted_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    resume = relationship("Resume", back_populates="candidate_profile")
    skills = relationship(
        "CandidateSkill", back_populates="candidate", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<CandidateProfile id={self.id} name={self.name!r}>"
