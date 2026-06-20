from dataclasses import dataclass
from typing import Protocol

from app.models.paper import Paper


@dataclass(frozen=True)
class ScoredDocument:
    document_id: int
    score: float


@dataclass(frozen=True)
class PaperDocument:
    document_id: int
    text: str
    title: str = ""
    abstract: str = ""
    year: int | None = None
    citation_count: int = 0


class Retriever(Protocol):
    method_name: str

    def fit(self, papers: list[PaperDocument]) -> None:
        ...

    def search(
        self,
        query: str,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[ScoredDocument]:
        ...


def papers_to_documents(papers: list[Paper]) -> list[PaperDocument]:
    return [
        PaperDocument(
            document_id=paper.id,
            text=paper.searchable_text,
            title=paper.title,
            abstract=paper.abstract,
            year=paper.year,
            citation_count=paper.citation_count,
        )
        for paper in papers
    ]


def coerce_documents(
    documents: list[PaperDocument] | list[tuple[int, str]],
) -> list[PaperDocument]:
    coerced: list[PaperDocument] = []
    for item in documents:
        if isinstance(item, PaperDocument):
            coerced.append(item)
        else:
            document_id, text = item
            coerced.append(PaperDocument(document_id=document_id, text=text))
    return coerced
