from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    cors_origin_regex: str | None = r"https://.*\.devtunnels\.ms"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_index_path: str = "data/processed/embeddings/all_minilm_l6_v2_5k.npz"
    faiss_index_path: str = "data/processed/faiss/all_minilm_l6_v2_5k.faiss"
    faiss_id_map_path: str = "data/processed/faiss/all_minilm_l6_v2_5k.ids.npz"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
