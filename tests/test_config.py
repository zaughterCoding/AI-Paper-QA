"""Configuration defaults and the embedding dimension constant."""

from app.core.config import EMBEDDING_DIM, Settings


def test_settings_have_local_defaults():
    # _env_file=None so only the defaults in code are checked, not the local .env
    settings = Settings(_env_file=None)

    assert "paperqa" in settings.database_url
    assert settings.embedding_model == "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    assert settings.llm_provider == "deepseek"


def test_embedding_dim_matches_minilm_model():
    # This constant sets the type of the chunks.embedding column. Switching to a model
    # with a different dimension requires a migration and a re-encode of every chunk.
    assert EMBEDDING_DIM == 384
