from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dimension of the embedding vectors. This is part of the database schema
# (chunks.embedding is vector(384)), not a runtime switch: changing it means updating
# this constant, adding an Alembic migration, and re-encoding every chunk.
EMBEDDING_DIM = 384
RetrievalMode = Literal["dense", "fts", "hybrid", "dense_rerank", "hybrid_rerank"]


class Settings(BaseSettings):
    """The single source of runtime configuration, read from the environment or .env."""

    # --- Database ---
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/paperqa"

    # --- Embedding ---
    # A QA-retrieval model rather than a general sentence-similarity one, but still
    # 384-dimensional so the schema is unchanged. It also scores unrelated passages
    # closer to zero, which leaves more room for a similarity threshold later.
    embedding_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

    retrieval_mode: RetrievalMode = "hybrid_rerank"
    retrieval_candidate_k: int = Field(default=30, ge=1, le=200)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"

    # --- LLM (OpenAI-compatible; DeepSeek by default) ---
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Process-wide singleton, so .env is parsed at most once."""
    return Settings()
