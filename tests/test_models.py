"""验证表结构真的按预期建在了 PostgreSQL 里。

这些测试回答四个问题：
1. 主键、时间戳这类默认值是不是自动填上了？
2. 唯一约束、外键约束是不是真的生效？
3. ORM 的关联关系能不能正常用？
4. pgvector 的向量列能不能真的存进 384 维向量？
"""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.core.config import EMBEDDING_DIM
from app.models.tables import Chunk, Document


def _make_document(**overrides) -> Document:
    defaults = {
        "title": "Attention Is All You Need",
        "source": "https://arxiv.org/abs/1706.03762",
        "content_hash": uuid.uuid4().hex,  # 每次不同，避免撞上唯一约束
    }
    return Document(**{**defaults, **overrides})


def _vector_literal(values: list[float]) -> str:
    """转成 pgvector 的文本格式 '[0.1,0.1,...]'，用来在 SQL 里直接比较。"""
    return "[" + ",".join(str(v) for v in values) + "]"


def test_document_gets_uuid_and_timestamp(db_session):
    document = _make_document()
    db_session.add(document)
    db_session.commit()

    assert isinstance(document.id, uuid.UUID)
    assert document.created_at is not None
    # 存的是带时区的 UTC 时间，不是裸的本地时间
    assert document.created_at.tzinfo is not None


def test_document_content_hash_must_be_unique(db_session):
    db_session.add(_make_document(content_hash="same-hash"))
    db_session.commit()

    db_session.add(_make_document(content_hash="same-hash"))
    # 唯一约束应该拦住重复导入，而不是悄悄存两条
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_chunk_links_back_to_document(db_session):
    document = _make_document()
    db_session.add(document)
    db_session.flush()  # 拿到 document.id，但先不提交

    chunk = Chunk(
        document_id=document.id,
        chunk_index=0,
        text="Self-attention relates different positions of a single sequence.",
        token_count=9,
    )
    db_session.add(chunk)
    db_session.commit()

    # 反向关联：从 chunk 能走到它所属的 document
    assert chunk.document.id == document.id
    assert chunk.embedding is None  # 刚导入时还没有向量，这是合法状态


def test_chunk_accepts_vector_embedding(db_session):
    document = _make_document()
    db_session.add(document)
    db_session.flush()

    vector = [0.1] * EMBEDDING_DIM
    chunk = Chunk(
        document_id=document.id,
        chunk_index=0,
        text="dummy",
        token_count=1,
        embedding=vector,
    )
    db_session.add(chunk)
    db_session.commit()

    # 让数据库自己算距离，验证向量真的按 384 维存进去了。
    # <=> 是 pgvector 的余弦距离运算符；和自身比较结果必然是 0。
    distance = db_session.execute(
        text("SELECT embedding <=> CAST(:query AS vector) FROM chunks WHERE id = :id"),
        {"query": _vector_literal(vector), "id": chunk.id},
    ).scalar()

    assert distance == pytest.approx(0.0)


def test_chunk_index_must_be_unique_per_document(db_session):
    document = _make_document()
    db_session.add(document)
    db_session.flush()

    db_session.add(Chunk(document_id=document.id, chunk_index=0, text="a", token_count=1))
    db_session.add(Chunk(document_id=document.id, chunk_index=0, text="b", token_count=1))
    with pytest.raises(IntegrityError):
        db_session.commit()
