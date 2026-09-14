from app.core.config import EMBEDDING_DIM, Settings


def test_settings_have_local_defaults():
    # _env_file=None 表示测试只验证代码里的默认值，不被本机 .env 干扰
    settings = Settings(_env_file=None)

    assert "paperqa" in settings.database_url
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.llm_provider == "deepseek"


def test_embedding_dim_matches_minilm_model():
    # all-MiniLM-L6-v2 输出 384 维；这个常量决定 chunks.embedding 列的类型
    assert EMBEDDING_DIM == 384
