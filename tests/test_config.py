from app.core.config import EMBEDDING_DIM, Settings


def test_settings_have_local_defaults():
    # _env_file=None 表示测试只验证代码里的默认值，不被本机 .env 干扰
    settings = Settings(_env_file=None)

    assert "paperqa" in settings.database_url
    assert settings.embedding_model == "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    assert settings.llm_provider == "deepseek"


def test_embedding_dim_matches_minilm_model():
    # multi-qa-MiniLM-L6-cos-v1 输出 384 维；这个常量决定 chunks.embedding 列的类型。
    # 它和 all-MiniLM-L6-v2 同维度，所以换模型时数据库 schema 不用动——
    # 但换来维度不同的模型（如 768 维的 mpnet 系）就必须写迁移并重编码全部 chunk。
    assert EMBEDDING_DIM == 384
