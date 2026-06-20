from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


PAPER_COLUMN_UPGRADES = {
    "external_id": "VARCHAR(256)",
    "source": "VARCHAR(128)",
    "doi": "VARCHAR(256)",
    "url": "TEXT",
    "references_count": "INTEGER DEFAULT 0",
    "influential_citation_count": "INTEGER DEFAULT 0",
    "abstract_word_count": "INTEGER DEFAULT 0",
    "updated_at": "TIMESTAMP",
}


def ensure_runtime_schema(engine: Engine) -> None:
    """Small dev-time bridge until the project adds Alembic migrations."""
    inspector = inspect(engine)
    if "papers" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("papers")}
    missing_columns = [
        (name, sql_type)
        for name, sql_type in PAPER_COLUMN_UPGRADES.items()
        if name not in existing_columns
    ]
    if not missing_columns:
        return

    with engine.begin() as connection:
        for column_name, sql_type in missing_columns:
            connection.execute(text(f"ALTER TABLE papers ADD COLUMN {column_name} {sql_type}"))
