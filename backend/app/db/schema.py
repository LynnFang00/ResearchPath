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


PAPER_POSTGRES_TYPE_UPGRADES = {
    "venue": "TEXT",
}

USER_PROFILE_COLUMN_UPGRADES = {
    "avoid_topics": "TEXT DEFAULT '[]' NOT NULL",
    "current_status": "VARCHAR(64) DEFAULT 'exploring' NOT NULL",
    "research_goal": "VARCHAR(64) DEFAULT 'learn_topic' NOT NULL",
    "paper_taste": "VARCHAR(64) DEFAULT 'balanced' NOT NULL",
}


def ensure_runtime_schema(engine: Engine) -> None:
    """Small dev-time bridge until the project adds Alembic migrations."""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if "papers" not in table_names:
        return

    existing_columns = {column["name"] for column in inspector.get_columns("papers")}
    missing_columns = [
        (name, sql_type)
        for name, sql_type in PAPER_COLUMN_UPGRADES.items()
        if name not in existing_columns
    ]
    with engine.begin() as connection:
        for column_name, sql_type in missing_columns:
            connection.execute(text(f"ALTER TABLE papers ADD COLUMN {column_name} {sql_type}"))
        if engine.dialect.name == "postgresql":
            for column_name, sql_type in PAPER_POSTGRES_TYPE_UPGRADES.items():
                if column_name in existing_columns:
                    connection.execute(text(f"ALTER TABLE papers ALTER COLUMN {column_name} TYPE {sql_type}"))
        if "user_profiles" in table_names:
            existing_profile_columns = {column["name"] for column in inspector.get_columns("user_profiles")}
            for column_name, sql_type in USER_PROFILE_COLUMN_UPGRADES.items():
                if column_name not in existing_profile_columns:
                    connection.execute(text(f"ALTER TABLE user_profiles ADD COLUMN {column_name} {sql_type}"))
