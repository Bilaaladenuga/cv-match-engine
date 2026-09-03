"""
Database models — all ORM models are imported here for convenience.
"""

from app.models.user import User
from app.models.resume import Resume
from app.models.candidate import CandidateProfile
from app.models.skill import Skill, CandidateSkill
from app.models.job import Job, JobRequirement
from app.models.match import Match, MatchExplanation

__all__ = [
    "User",
    "Resume",
    "CandidateProfile",
    "Skill",
    "CandidateSkill",
    "Job",
    "JobRequirement",
    "Match",
    "MatchExplanation",
]
