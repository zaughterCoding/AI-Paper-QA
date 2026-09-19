from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# embedding 向量的维度。它是**数据库 schema 的一部分**（chunks.embedding 列的类型是
# vector(384)），不是可以随便改的运行时开关：换用别的 embedding 模型时，
# 必须同时更新这里的常量、写一个新的 Alembic 迁移，并重新生成所有 chunk 的向量。
EMBEDDING_DIM = 384


class Settings(BaseSettings):
    """全部运行时配置的唯一来源，字段值来自环境变量或 .env 文件。"""

    # --- 数据库 ---
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/paperqa"

    # --- Embedding ---
    # 选它而不是 all-MiniLM-L6-v2：同样是 384 维（schema 不用改），但它是**问答检索**
    # 模型（2.15 亿条 question-passage 对训练），不是通用句子相似度模型。
    # 实测见 docs/Task7_Embedding_Client.md：无关 chunk 的相似度被压到 0.03~0.11，
    # 而旧模型的地板高得多（0.11~0.19），噪音底噪低才好定检索阈值。
    embedding_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

    # --- LLM（OpenAI 兼容接口，默认 DeepSeek）---
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """进程内单例。lru_cache 保证 .env 只被解析一次。"""
    return Settings()
