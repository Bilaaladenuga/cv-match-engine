"""
Resume model — uploaded CV documents and their parsed data.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="uploaded"
    )  # uploaded | parsed | error
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="resumes")
    candidate_profile = relationship(
        "CandidateProfile", back_populates="resume", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Resume id={self.id} filename={self.filename!r}>"
