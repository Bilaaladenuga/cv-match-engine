"""
Job and JobRequirement models — job descriptions and requirements.
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Job(Base):
    """A job description."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    company: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    requirements = relationship(
        "JobRequirement", back_populates="job", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Job id={self.id} title={self.title!r}>"


class JobRequirement(Base):
    """A single requirement extracted from a job description."""

    __tablename__ = "job_requirements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    skill_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("skills.id", ondelete="CASCADE"), nullable=False
    )
    required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)  # 0.0–1.0
    minimum_experience: Mapped[float | None] = mapped_column(Float, nullable=True)  # years

    # Relationships
    job = relationship("Job", back_populates="requirements")
    skill = relationship("Skill", back_populates="job_requirements")

    def __repr__(self) -> str:
        return (
            f"<JobRequirement job_id={self.job_id} skill_id={self.skill_id} "
            f"required={self.required}>"
        )
