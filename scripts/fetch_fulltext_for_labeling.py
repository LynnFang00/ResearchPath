import argparse
from dataclasses import dataclass
from html import unescape
import json
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.schema import ensure_runtime_schema  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.models.paper import Paper  # noqa: E402


DEFAULT_POOL = REPO_ROOT / "data" / "eval" / "manual_label_pool_v1.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "fulltext"
MANIFEST_NAME = "fulltext_manifest_v1.jsonl"
USER_AGENT = "ResearchPath/0.1 fulltext labeling helper (open-access PDFs only)"


@dataclass(frozen=True)
class FullTextSource:
    source_type: str
    source_url: str


@dataclass(frozen=True)
class SourceLookupResult:
    source: FullTextSource | None
    lookup_attempts: list[str]
    error: str = ""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File was not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def unique_pool_rows(rows: list[dict[str, Any]], *, query_id: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    seen: set[int] = set()
    selected: list[dict[str, Any]] = []
    for row in rows:
        if query_id and row.get("query_id") != query_id:
            continue
        paper_id = int(row["paper_id"])
        if paper_id in seen:
            continue
        seen.add(paper_id)
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def find_fulltext_source(
    row: dict[str, Any],
    paper: Paper | None = None,
    *,
    use_openalex: bool = False,
    use_arxiv_title_search: bool = False,
    sleep_seconds: float = 0.2,
) -> SourceLookupResult:
    lookup_attempts = ["local_metadata"]
    lookup_errors: list[str] = []
    doi = str(row.get("doi") or (paper.doi if paper is not None else "") or "")
    external_id = str(row.get("external_id") or (paper.external_id if paper is not None else "") or "")
    arxiv_id = extract_arxiv_id(doi) or extract_arxiv_id(external_id)
    if arxiv_id:
        return SourceLookupResult(
            source=FullTextSource(source_type="local_arxiv", source_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf"),
            lookup_attempts=lookup_attempts,
        )

    for field_name in ("full_text_url", "pdf_url", "oa_url", "source_url", "url"):
        value = row.get(field_name)
        if value and is_allowed_direct_pdf_url(str(value)):
            return SourceLookupResult(
                source=FullTextSource(source_type=field_name, source_url=str(value)),
                lookup_attempts=lookup_attempts,
            )

    if paper is not None:
        for field_name in ("source_url", "url"):
            value = getattr(paper, field_name, None)
            if value and is_allowed_direct_pdf_url(str(value)):
                return SourceLookupResult(
                    source=FullTextSource(source_type=field_name, source_url=str(value)),
                    lookup_attempts=lookup_attempts,
                )
            arxiv_id = extract_arxiv_id(str(value or ""))
            if arxiv_id:
                return SourceLookupResult(
                    source=FullTextSource(source_type="local_arxiv", source_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf"),
                    lookup_attempts=lookup_attempts,
                )

    if use_openalex:
        openalex_id = extract_openalex_work_id(external_id)
        if openalex_id:
            lookup_attempts.append("openalex_id")
            try:
                source = source_from_openalex_work(fetch_openalex_work_by_id(openalex_id))
                if source:
                    return SourceLookupResult(source=source, lookup_attempts=lookup_attempts)
            except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
                lookup_errors.append(f"openalex_id: {exc}")
            sleep_if_needed(sleep_seconds)

        normalized_doi = normalize_doi(doi)
        if normalized_doi:
            lookup_attempts.append("openalex_doi")
            try:
                source = source_from_openalex_work(fetch_openalex_work_by_doi(normalized_doi))
                if source:
                    return SourceLookupResult(source=source, lookup_attempts=lookup_attempts)
            except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
                lookup_errors.append(f"openalex_doi: {exc}")
            sleep_if_needed(sleep_seconds)

    if use_arxiv_title_search and looks_like_cs_ml_title(str(row.get("title") or "")):
        lookup_attempts.append("arxiv_title")
        try:
            source = find_arxiv_source_by_title(str(row.get("title") or ""), sleep_seconds=sleep_seconds)
            if source:
                return SourceLookupResult(source=source, lookup_attempts=lookup_attempts)
        except (HTTPError, URLError, OSError, ValueError, ElementTree.ParseError) as exc:
            lookup_errors.append(f"arxiv_title: {exc}")

    return SourceLookupResult(source=None, lookup_attempts=lookup_attempts, error="; ".join(lookup_errors))


def extract_arxiv_id(value: str) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None
    patterns = [
        r"10\.48550/arxiv\.([0-9]{4}\.[0-9]{4,5})(v[0-9]+)?",
        r"arxiv:([0-9]{4}\.[0-9]{4,5})(v[0-9]+)?",
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(v[0-9]+)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_openalex_work_id(value: str) -> str | None:
    match = re.search(r"(?:openalex:|openalex\.org/)(W[0-9]+)", value.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def normalize_doi(value: str) -> str:
    normalized = value.strip()
    normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^doi:", "", normalized, flags=re.IGNORECASE)
    return normalized.lower()


def is_allowed_direct_pdf_url(value: str) -> bool:
    parsed = urlparse(value)
    if parsed.scheme != "https":
        return False
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if host.endswith("arxiv.org"):
        return path.startswith("/pdf/")
    if host == "openreview.net" and path == "/pdf":
        return True
    return path.endswith(".pdf")


def sleep_if_needed(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def fetch_json(url: str, *, timeout: int = 30) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_openalex_work_by_id(work_id: str) -> dict[str, Any]:
    return fetch_json(f"https://api.openalex.org/works/{work_id}")


def fetch_openalex_work_by_doi(doi: str) -> dict[str, Any]:
    return fetch_json(f"https://api.openalex.org/works/doi:{quote(doi, safe='')}")


def source_from_openalex_work(work: dict[str, Any]) -> FullTextSource | None:
    if not bool((work.get("open_access") or {}).get("is_oa")):
        return None

    candidate_locations: list[dict[str, Any]] = []
    for key in ("best_oa_location", "primary_location"):
        location = work.get(key)
        if isinstance(location, dict):
            candidate_locations.append(location)
    locations = work.get("locations") or []
    if isinstance(locations, list):
        candidate_locations.extend(location for location in locations if isinstance(location, dict))

    seen: set[str] = set()
    for location in candidate_locations:
        is_oa_location = bool(location.get("is_oa")) or location == work.get("best_oa_location")
        if not is_oa_location:
            continue
        for field_name in ("pdf_url", "landing_page_url"):
            value = location.get(field_name)
            if not value:
                continue
            url = str(value)
            if url in seen:
                continue
            seen.add(url)
            if is_allowed_direct_pdf_url(url):
                return FullTextSource(source_type="openalex_pdf", source_url=url)
    return None


def normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", unescape(value).lower())).strip()


def title_similarity(left: str, right: str) -> float:
    left_tokens = set(normalize_title(left).split())
    right_tokens = set(normalize_title(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def looks_like_cs_ml_title(title: str) -> bool:
    title_terms = set(normalize_title(title).split())
    ml_terms = {
        "architecture",
        "attention",
        "bayesian",
        "benchmark",
        "contrastive",
        "diffusion",
        "graph",
        "image",
        "language",
        "learning",
        "model",
        "neural",
        "network",
        "recommendation",
        "retrieval",
        "transformer",
        "vision",
    }
    return bool(title_terms & ml_terms)


def find_arxiv_source_by_title(title: str, *, sleep_seconds: float = 0.2) -> FullTextSource | None:
    query = urlencode({"search_query": f'ti:"{title}"', "start": 0, "max_results": 5})
    request = Request(f"https://export.arxiv.org/api/query?{query}", headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        payload = response.read()
    sleep_if_needed(sleep_seconds)
    return source_from_arxiv_feed(payload, title)


def source_from_arxiv_feed(payload: bytes, title: str) -> FullTextSource | None:
    root = ElementTree.fromstring(payload)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    best_url: str | None = None
    best_score = 0.0
    for entry in root.findall("atom:entry", namespace):
        entry_title = entry.findtext("atom:title", default="", namespaces=namespace)
        score = title_similarity(title, entry_title)
        if score < 0.75 or score < best_score:
            continue
        pdf_url = None
        for link in entry.findall("atom:link", namespace):
            href = link.attrib.get("href", "")
            title_attr = link.attrib.get("title", "")
            if title_attr == "pdf" and is_allowed_direct_pdf_url(href):
                pdf_url = href
                break
        if pdf_url is None:
            entry_id = entry.findtext("atom:id", default="", namespaces=namespace)
            arxiv_id = extract_arxiv_id(entry_id)
            if arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if pdf_url:
            best_score = score
            best_url = pdf_url
    if best_url:
        return FullTextSource(source_type="arxiv_title", source_url=best_url)
    return None


def download_pdf(url: str, path: Path, *, timeout: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "").lower()
        payload = response.read()
    if not payload.startswith(b"%PDF") and "pdf" not in content_type:
        raise ValueError(f"URL did not return a PDF response: content-type={content_type or 'unknown'}")
    path.write_bytes(payload)


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is not installed. Install backend dependencies to extract PDF text.") from exc

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(page.strip() for page in pages if page.strip())


def manifest_row_for(
    row: dict[str, Any],
    *,
    source: FullTextSource | None,
    pdf_path: Path,
    text_path: Path,
    status: str,
    lookup_attempts: list[str] | None = None,
    error: str = "",
    text_char_count: int = 0,
) -> dict[str, Any]:
    return {
        "paper_id": int(row["paper_id"]),
        "title": row.get("title", ""),
        "doi": row.get("doi"),
        "external_id": row.get("external_id"),
        "full_text_available": status == "ok",
        "source_type": source.source_type if source else None,
        "source_url": source.source_url if source else None,
        "lookup_attempts": lookup_attempts or [],
        "pdf_path": str(pdf_path) if pdf_path.exists() else None,
        "text_path": str(text_path) if text_path.exists() else None,
        "status": status,
        "error": error,
        "text_char_count": text_char_count,
    }


def fetch_fulltext_for_pool(
    *,
    pool_path: Path = DEFAULT_POOL,
    out_dir: Path = DEFAULT_OUT_DIR,
    limit: int | None = None,
    query_id: str | None = None,
    use_openalex: bool = False,
    use_arxiv_title_search: bool = False,
    sleep_seconds: float = 0.2,
    max_downloads: int | None = None,
) -> dict[str, Any]:
    pool_rows = unique_pool_rows(load_jsonl(pool_path), query_id=query_id, limit=limit)
    pdf_dir = out_dir / "pdfs"
    text_dir = out_dir / "text"
    manifest_path = out_dir / MANIFEST_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)

    manifest_rows: list[dict[str, Any]] = []
    downloads_started = 0
    with SessionLocal() as db:
        for row in pool_rows:
            paper_id = int(row["paper_id"])
            paper = db.get(Paper, paper_id)
            lookup = find_fulltext_source(
                row,
                paper,
                use_openalex=use_openalex,
                use_arxiv_title_search=use_arxiv_title_search,
                sleep_seconds=sleep_seconds,
            )
            source = lookup.source
            pdf_path = pdf_dir / f"{paper_id}.pdf"
            text_path = text_dir / f"{paper_id}.txt"
            if source is None:
                manifest_rows.append(
                    manifest_row_for(
                        row,
                        source=None,
                        pdf_path=pdf_path,
                        text_path=text_path,
                        status="no_open_source",
                        lookup_attempts=lookup.lookup_attempts,
                        error=lookup.error,
                    )
                )
                continue

            if max_downloads is not None and not pdf_path.exists() and downloads_started >= max_downloads:
                manifest_rows.append(
                    manifest_row_for(
                        row,
                        source=source,
                        pdf_path=pdf_path,
                        text_path=text_path,
                        status="download_limit_reached",
                        lookup_attempts=lookup.lookup_attempts,
                    )
                )
                continue

            try:
                if not pdf_path.exists():
                    downloads_started += 1
                    download_pdf(source.source_url, pdf_path)
                    sleep_if_needed(sleep_seconds)
                text = extract_pdf_text(pdf_path)
                text_path.parent.mkdir(parents=True, exist_ok=True)
                text_path.write_text(text, encoding="utf-8")
                manifest_rows.append(
                    manifest_row_for(
                        row,
                        source=source,
                        pdf_path=pdf_path,
                        text_path=text_path,
                        status="ok",
                        lookup_attempts=lookup.lookup_attempts,
                        text_char_count=len(text),
                    )
                )
            except (HTTPError, URLError, OSError, RuntimeError, ValueError) as exc:
                manifest_rows.append(
                    manifest_row_for(
                        row,
                        source=source,
                        pdf_path=pdf_path,
                        text_path=text_path,
                        status="error",
                        lookup_attempts=lookup.lookup_attempts,
                        error=str(exc),
                    )
                )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in manifest_rows:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    return {
        "candidate_count": len(pool_rows),
        "available_count": sum(1 for item in manifest_rows if item["full_text_available"]),
        "downloads_started": downloads_started,
        "manifest_path": str(manifest_path),
    }


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch open-access PDFs and extract text for labeling evidence.")
    parser.add_argument("--pool", default=str(DEFAULT_POOL))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-id", default=None)
    parser.add_argument("--use-openalex", action="store_true")
    parser.add_argument("--use-arxiv-title-search", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-downloads", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = fetch_fulltext_for_pool(
        pool_path=resolve_repo_path(args.pool),
        out_dir=resolve_repo_path(args.out_dir),
        limit=args.limit,
        query_id=args.query_id,
        use_openalex=args.use_openalex,
        use_arxiv_title_search=args.use_arxiv_title_search,
        sleep_seconds=args.sleep_seconds,
        max_downloads=args.max_downloads,
    )
    print(f"Candidates checked: {report['candidate_count']}")
    print(f"Full text available: {report['available_count']}")
    print(f"Downloads started: {report['downloads_started']}")
    print(f"Manifest: {report['manifest_path']}")


if __name__ == "__main__":
    main()
