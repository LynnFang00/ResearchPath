import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.db.schema import ensure_runtime_schema  # noqa: E402
from app.services.ingestion import ingest_papers_from_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ResearchPath papers from JSONL or CSV.")
    parser.add_argument("--file", required=True, help="Path to a local JSONL or CSV file.")
    parser.add_argument("--dataset-name", default=None, help="Dataset name for the generated manifest.")
    parser.add_argument("--source", default="local_file", help="Dataset source label.")
    parser.add_argument("--notes", default=None, help="Optional manifest notes.")
    parser.add_argument("--max-warnings", type=int, default=25, help="Maximum warnings to print.")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)

    input_path = Path(args.file)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path

    with SessionLocal() as db:
        result = ingest_papers_from_file(
            input_path,
            db,
            dataset_name=args.dataset_name,
            source=args.source,
            notes=args.notes,
        )

    print(f"Inserted: {result['inserted']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Citation edges inserted: {result['citation_edges_inserted']}")
    if result["manifest_path"]:
        print(f"Manifest: {result['manifest_path']}")
    warnings_to_print = result["warnings"][: args.max_warnings]
    for warning in warnings_to_print:
        print(f"Warning: {warning}")
    remaining_warnings = len(result["warnings"]) - len(warnings_to_print)
    if remaining_warnings > 0:
        print(f"Warning: {remaining_warnings} additional warnings omitted. See manifest for full details.")
    for error in result["errors"]:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
