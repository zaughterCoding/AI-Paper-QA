"""文档导入服务的测试。

这些测试跑在真实的 PostgreSQL 上（由 conftest.py 的 db_session 提供），
所以它们验证的是"端到端真的写进库了"，而不是"我 mock 对了没有"。
"""

import hashlib

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.tables import Chunk, Document
from app.rag.chunking import TextChunk, TextChunker
from app.repositories.documents import DocumentRepository
from app.services.ingestion import DocumentIngestionService

PARAGRAPH = (
    "The dominant sequence transduction models are based on complex recurrent "
    "or convolutional neural networks that include an encoder and a decoder."
)

# 400 个词，配合默认的 chunk_size=180 / overlap=30，会切出 3 个片段：
# [0:180] [150:330] [300:400]，第三段切到结尾就停止。
LONG_CONTENT = " ".join(f"token{i}" for i in range(400))


def _count(session: Session, model: type) -> int:
    """数一张表有多少行。用 COUNT(*) 而不是 len(session.query(...).all())。"""
    return session.scalar(select(func.count()).select_from(model)) or 0


def test_ingest_saves_document_and_chunks(db_session: Session) -> None:
    service = DocumentIngestionService(db_session)

    result = service.ingest(
        title="Attention Is All You Need",
        source="arxiv:1706.03762",
        content=LONG_CONTENT,
    )

    assert result.created is True
    assert result.chunk_count == 3

    document = db_session.get(Document, result.document_id)
    assert document is not None
    assert document.title == "Attention Is All You Need"
    assert document.source == "arxiv:1706.03762"

    # 片段确实落库了，而且都挂在同一篇文档下
    stored = db_session.scalars(
        select(Chunk).where(Chunk.document_id == result.document_id)
    ).all()
    assert len(stored) == 3


def test_ingest_stores_chunks_in_order_with_content(db_session: Session) -> None:
    # 注入一个小切分器，让片段边界一眼能算出来
    service = DocumentIngestionService(db_session, chunker=TextChunker(chunk_size=5, overlap=1))

    result = service.ingest(title="Counting", source="test", content="0 1 2 3 4 5 6 7 8 9")

    rows = db_session.scalars(
        select(Chunk)
        .where(Chunk.document_id == result.document_id)
        .order_by(Chunk.chunk_index)
    ).all()

    assert result.chunk_count == 3
    assert [row.chunk_index for row in rows] == [0, 1, 2]
    assert [row.text for row in rows] == ["0 1 2 3 4", "4 5 6 7 8", "8 9"]
    assert [row.token_count for row in rows] == [5, 5, 2]


def test_ingest_stores_sha256_of_content(db_session: Session) -> None:
    service = DocumentIngestionService(db_session)

    result = service.ingest(title="Attention", source="arxiv", content=PARAGRAPH)

    document = db_session.get(Document, result.document_id)
    assert document.content_hash == hashlib.sha256(PARAGRAPH.encode("utf-8")).hexdigest()


def test_ingest_is_idempotent_for_same_content(db_session: Session) -> None:
    """同样内容导两次，不会产生第二份数据。

    这是 content_hash 唯一约束存在的意义，也是 service 层必须处理的场景：
    用户重复上传同一篇论文是常态，不应该报错、更不应该存两份。
    """
    service = DocumentIngestionService(db_session)

    first = service.ingest(title="First title", source="src-a", content=PARAGRAPH)
    second = service.ingest(title="Second title", source="src-b", content=PARAGRAPH)

    assert second.document_id == first.document_id
    assert second.created is False
    # 第二次也能拿到和第一次一样的片段数，而不是 0
    assert second.chunk_count == first.chunk_count

    assert _count(db_session, Document) == 1
    assert _count(db_session, Chunk) == first.chunk_count

    # 已存在时以**原来那份**为准，后来的标题不会覆盖它
    document = db_session.get(Document, first.document_id)
    assert document.title == "First title"
    assert document.source == "src-a"


@pytest.mark.parametrize(
    ("title", "content"),
    [
        ("", PARAGRAPH),
        ("   ", PARAGRAPH),
        ("\n\t", PARAGRAPH),
        ("Attention", ""),
        ("Attention", "   \n\t  "),
    ],
)
def test_ingest_rejects_empty_title_or_content(
    db_session: Session, title: str, content: str
) -> None:
    service = DocumentIngestionService(db_session)

    with pytest.raises(ValueError):
        service.ingest(title=title, source="arxiv", content=content)

    # 校验失败时必须什么都没写进去——不能留下半篇文档
    assert _count(db_session, Document) == 0
    assert _count(db_session, Chunk) == 0


def test_ingest_commits_so_data_survives_rollback(db_session: Session) -> None:
    """服务自己负责提交事务。

    这里回滚 session 之后数据仍在，说明 ingest 内部已经 commit 了。
    如果服务忘了提交，数据此时还挂在一个未提交的事务里，会被回滚掉，
    这个断言就会失败。
    """
    service = DocumentIngestionService(db_session)

    result = service.ingest(title="Attention", source="arxiv", content=PARAGRAPH)

    db_session.rollback()
    db_session.expunge_all()  # 清掉身份映射缓存，强制重新查库

    assert db_session.get(Document, result.document_id) is not None
    assert _count(db_session, Chunk) == result.chunk_count


def test_ingest_recovers_from_concurrent_insert(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """竞态兜底分支：查的时候还没有，插的时候已经有了。

    这种时序在真实环境里是"两个请求同时上传同一篇论文"，概率低但会发生，
    没法自然复现。这里把 repository 的第一次查询伪装成"没查到"来构造它：
    服务于是继续往下走、尝试插入，被唯一约束拦下，走 except 分支恢复。

    没有这个兜底，用户拿到的会是一个 500，而不是"这篇已经导过了"。
    """
    service = DocumentIngestionService(db_session)
    first = service.ingest(title="Attention", source="arxiv", content=PARAGRAPH)

    real_lookup = DocumentRepository.get_by_content_hash
    calls = {"count": 0}

    def lookup_missing_first(self: DocumentRepository, content_hash: str):
        calls["count"] += 1
        if calls["count"] == 1:
            return None  # 模拟：另一个请求还没提交，所以这边查不到
        return real_lookup(self, content_hash)

    monkeypatch.setattr(DocumentRepository, "get_by_content_hash", lookup_missing_first)

    second = service.ingest(title="Attention", source="arxiv", content=PARAGRAPH)

    assert calls["count"] == 2  # 确认确实走了"插入失败 → 重查"这条路
    assert second.document_id == first.document_id
    assert second.created is False
    assert _count(db_session, Document) == 1


class _RecordingChunker:
    """假的切分器：不真的切，只记录"被调用时收到了什么"，并返回固定片段。

    它存在的意义是让"服务把原始内容原样交给切分器"这件事可以被断言。
    """

    def __init__(self, chunks: list[TextChunk]) -> None:
        self._chunks = chunks
        self.received: list[str] = []

    def chunk(self, text: str) -> list[TextChunk]:
        self.received.append(text)
        return self._chunks


def test_ingest_uses_injected_chunker_and_persists_its_output(db_session: Session) -> None:
    chunker = _RecordingChunker(
        [
            TextChunk(index=0, text="alpha beta", token_count=2),
            TextChunk(index=1, text="gamma delta", token_count=2),
        ]
    )
    service = DocumentIngestionService(db_session, chunker=chunker)

    result = service.ingest(title="Fixed", source="test", content=PARAGRAPH)

    # 切分器收到的是原始内容（未被 title 之类的东西污染）
    assert chunker.received == [PARAGRAPH]

    rows = db_session.scalars(
        select(Chunk)
        .where(Chunk.document_id == result.document_id)
        .order_by(Chunk.chunk_index)
    ).all()
    assert [(row.text, row.token_count) for row in rows] == [
        ("alpha beta", 2),
        ("gamma delta", 2),
    ]
