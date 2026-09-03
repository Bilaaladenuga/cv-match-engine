"""
Match and MatchExplanation models — CV–Job compatibility results.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Match(Base):
    """Result of a CV–Job compatibility analysis."""

    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    candidate_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # Scores (0.0–100.0)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    semantic_score: Mapped[float] = mapped_column(Float, nullable=False)
    skills_score: Mapped[float] = mapped_column(Float, nullable=False)
    experience_score: Mapped[float] = mapped_column(Float, nullable=False)
    education_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Metadata
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_values: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    explanation = relationship(
        "MatchExplanation", back_populates="match", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Match id={self.id} overall_score={self.overall_score} "
            f"model_version={self.model_version!r}>"
        )


class MatchExplanation(Base):
    """Human-readable explanation of a match result."""

    __tablename__ = "match_explanations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id", ondelete="CASCADE"),
        unique=True, nullable=False,
    )
    matched_skills: Mapped[list | None] = mapped_column(JSON, nullable=True)
    missing_skills: Mapped[list | None] = mapped_column(JSON, nullable=True)
    partial_skills: Mapped[list | None] = mapped_column(JSON, nullable=True)
    recommendations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    positive_factors: Mapped[list | None] = mapped_column(JSON, nullable=True)
    negative_factors: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Relationships
    match = relationship("Match", back_populates="explanation")

    def __repr__(self) -> str:
        return f"<MatchExplanation id={self.id} match_id={self.match_id}>"
