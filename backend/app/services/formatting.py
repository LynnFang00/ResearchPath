import re

from app.models.paper import Paper
from app.schemas.paper import RecommendationResponse


def split_semicolon_field(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def make_snippet(text: str, max_chars: int = 260) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def normalize_doi_url(value: str | None) -> str | None:
    if not value:
        return None
    doi = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :].strip()
            break
    return f"https://doi.org/{doi}" if doi else None


def extract_arxiv_id(*values: str | None) -> str | None:
    for value in values:
        if not value:
            continue
        text = value.strip()
        url_match = re.search(
            r"arxiv\.org/(?:abs|pdf)/([A-Za-z\-]+/[0-9]{7}|[0-9]{4}\.[0-9]{4,5})(v[0-9]+)?",
            text,
            flags=re.IGNORECASE,
        )
        if url_match:
            return f"{url_match.group(1)}{url_match.group(2) or ''}"
        doi_match = re.search(
            r"10\.48550/arxiv\.([A-Za-z\-]+/[0-9]{7}|[0-9]{4}\.[0-9]{4,5})(v[0-9]+)?",
            text,
            flags=re.IGNORECASE,
        )
        if doi_match:
            return f"{doi_match.group(1)}{doi_match.group(2) or ''}"
        normalized = text.lower().removeprefix("arxiv:").strip()
        direct_match = re.fullmatch(r"([a-z\-]+/[0-9]{7}|[0-9]{4}\.[0-9]{4,5})(v[0-9]+)?", normalized)
        if direct_match:
            return text.lower().removeprefix("arxiv:").strip()
    return None


def paper_links(paper: Paper) -> tuple[str | None, str | None, str | None]:
    arxiv_id = extract_arxiv_id(paper.url, paper.source_url, paper.external_id, paper.doi)
    doi_url = normalize_doi_url(paper.doi)
    source_url = paper.url or paper.source_url
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None
    if not pdf_url and source_url and source_url.lower().split("?", 1)[0].endswith(".pdf"):
        pdf_url = source_url
    paper_url = source_url or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None) or doi_url
    return paper_url, pdf_url, doi_url


def paper_to_recommendation(
    paper: Paper,
    score: float,
    method: str,
    explanation: str,
) -> RecommendationResponse:
    paper_url, pdf_url, doi_url = paper_links(paper)
    return RecommendationResponse(
        paper_id=paper.id,
        title=paper.title,
        abstract_snippet=make_snippet(paper.abstract),
        year=paper.year,
        authors=split_semicolon_field(paper.authors),
        venue=paper.venue,
        paper_url=paper_url,
        pdf_url=pdf_url,
        doi_url=doi_url,
        source_url=paper.source_url or paper.url,
        doi=paper.doi,
        score=round(float(score), 6),
        method=method,
        explanation=explanation,
    )
