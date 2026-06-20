from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class CitationEdge(Base):
    __tablename__ = "citation_edges"
    __table_args__ = (
        UniqueConstraint(
            "source_paper_id",
            "target_paper_id",
            "relationship_type",
            name="uq_citation_edge_relationship",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    source_paper_id: Mapped[int] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_paper_id: Mapped[int] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relationship_type: Mapped[str] = mapped_column(String(64), nullable=False, default="citation")
    source: Mapped[str | None] = mapped_column(String(128), nullable=True)
