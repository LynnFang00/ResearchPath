from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, default="default")
    background_level: Mapped[str] = mapped_column(String(64), nullable=False, default="basic_ml")
    saved_paper_ids: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    skipped_paper_ids: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    too_easy_paper_ids: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    too_hard_paper_ids: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    preferred_topics: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    avoid_topics: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    current_status: Mapped[str] = mapped_column(String(64), nullable=False, default="exploring")
    research_goal: Mapped[str] = mapped_column(String(64), nullable=False, default="learn_topic")
    paper_taste: Mapped[str] = mapped_column(String(64), nullable=False, default="balanced")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
