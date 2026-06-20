from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[str] = mapped_column(Text, nullable=False, default="")
    year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    venue: Mapped[str | None] = mapped_column(String(256), nullable=True)
    categories: Mapped[str] = mapped_column(Text, nullable=False, default="")
    citation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    external_id: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    source: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    doi: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    references_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    influential_citation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    abstract_word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    @property
    def searchable_text(self) -> str:
        return f"{self.title}\n\n{self.abstract}"
