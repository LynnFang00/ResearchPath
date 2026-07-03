from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    cors_origin_regex: str | None = r"https://.*\.devtunnels\.ms"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_index_path: str = "data/processed/embeddings/all_minilm_l6_v2_50k.npz"
    faiss_index_path: str = "data/processed/faiss/all_minilm_l6_v2_50k.faiss"
    faiss_id_map_path: str = "data/processed/faiss/all_minilm_l6_v2_50k.ids.npz"
    learned_ranker_path: str = "data/processed/models/lightweight_ranker.json"
    learned_ranker_v2_2b_path: str = "data/processed/models/v2_2b_lightweight_learned_hybrid.json"
    learned_ranker_v2_6_path: str = "data/processed/models/v2_6_production_aware_learned_hybrid.json"
    learned_blend_v2_7_path: str = "data/processed/models/v2_7_score_blend.json"
    learned_ranker_v3_3_dir: str = "data/processed/models/v3_3_ltr_800"
    learned_ranker_v4_1_dir: str = "data/processed/models/v4_1_weighted_ltr_2400"
    learned_ranker_v4_1_blend_config_path: str = "data/processed/models/v4_1_calibrated_blend.json"
    learned_ranker_v4_3_text_dir: str = "data/processed/models/v4_3_text_reranker"
    learned_ranker_v4_9_guarded_text_config_path: str = "data/processed/models/v4_9_guarded_text_blend_candidate.json"
    learned_ranker_v6_4_safe_fusion_config_path: str = "data/processed/models/v6_4_safe_fusion_candidate.json"
    learned_ranker_v6_6_safe_fusion_ridge_scorer_path: str = "data/processed/models/v6_6_safe_fusion_ridge_scorer.json"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
