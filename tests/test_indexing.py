"""索引服务（向量回填）的测试。

这个服务补的是设计书数据流的第 4、5 步——"生成 embedding"和"保存向量"。
它掉进过 Task 5 与 Task 8 之间的缝里（见 F-39），这组测试就是把它钉住。

和 test_retrieval.py 的分工：
- test_retrieval.py 测的是"给定向量，能不能查对"（读路径）
- 这里测的是"文本能不能变成向量并正确落库"（写路径）
两者在 `test_index_document_makes_chunks_searchable` 里合流——
回填的**唯一意义**就是让片段能被检索到。
"""

import uuid

import pytest
from sqlalchemy.orm import Session

from app.models.tables import Chunk, Document
from app.repositories.chunks import ChunkRepository
from app.services.indexing import IndexingResult, IndexingService
from tests.fakes import FakeEmbeddingClient


@pytest.fixture
def embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
def repo(db_session: Session) -> ChunkRepository:
    return ChunkRepository(db_session)


@pytest.fixture
def service(db_session: Session, embedder: FakeEmbeddingClient) -> IndexingService:
    return IndexingService(db_session, embedder)


def make_document(
    session: Session, title: str = "Attention Is All You Need", content_hash: str = "a" * 64
) -> Document:
    document = Document(title=title, source="test", content_hash=content_hash)
    session.add(document)
    session.flush()
    return document


def add_chunk(
    session: Session,
    document: Document,
    index: int,
    text: str,
    embedding: list[float] | None = None,
) -> Chunk:
    chunk = Chunk(
        document_id=document.id,
        chunk_index=index,
        text=text,
        token_count=len(text.split()),
        embedding=embedding,
    )
    session.add(chunk)
    session.flush()
    return chunk


def embedding_of(session: Session, chunk: Chunk) -> list[float] | None:
    """从数据库重新读一个片段的向量（不信内存里的对象）。

    `expire()` 让 SQLAlchemy 把这个对象标记为"过期"，下次访问属性时**重新查库**。
    没有这一步，读到的可能只是刚才写进内存、还没落库的值——
    那等于在验证自己刚写的代码，而不是验证"数据真的进数据库了"。
    """
    session.expire(chunk)
    stored = session.get(Chunk, chunk.id)
    assert stored is not None
    return list(stored.embedding) if stored.embedding is not None else None


def count_pending(session: Session) -> int:
    """全库还有多少个片段没有向量。"""
    return len(ChunkRepository(session).list_without_embedding())


# --- repository：找出待回填的片段 ---------------------------------------------


def test_list_without_embedding_skips_chunks_that_already_have_vectors(db_session, repo) -> None:
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "done", embedding=[0.5] * 384)
    pending = add_chunk(db_session, document, 1, "todo")

    assert [chunk.id for chunk in repo.list_without_embedding()] == [pending.id]


def test_list_without_embedding_can_be_limited_to_one_document(db_session, repo) -> None:
    first = make_document(db_session, content_hash="a" * 64)
    second = make_document(db_session, content_hash="b" * 64)
    chunk_of_first = add_chunk(db_session, first, 0, "first")
    add_chunk(db_session, second, 0, "second")

    assert [chunk.id for chunk in repo.list_without_embedding(document_id=first.id)] == [
        chunk_of_first.id
    ]
    assert len(repo.list_without_embedding()) == 2


def test_list_without_embedding_returns_a_deterministic_order(db_session, repo) -> None:
    """顺序必须确定，且与插入顺序无关。

    不写 ORDER BY 时，PostgreSQL 返回行的顺序是**未定义的**——取决于物理存储、
    并行扫描、甚至缓存命中情况，同一个查询两次跑可能给出不同顺序。
    对回填本身这无所谓，但对**测试**影响很大：断言"哪条先被写入"会随机失败。

    这里故意乱序插入（3, 1, 2, 0），断言输出一定是 0, 1, 2, 3；
    如果实现里没有 ORDER BY，这条会不稳定地红。
    """
    document = make_document(db_session)
    for index in (3, 1, 2, 0):
        add_chunk(db_session, document, index, f"text {index}")

    order = [chunk.chunk_index for chunk in repo.list_without_embedding()]

    assert order == [0, 1, 2, 3]


def test_count_all_counts_every_chunk_regardless_of_embedding(db_session, repo) -> None:
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "done", embedding=[0.5] * 384)
    add_chunk(db_session, document, 1, "todo")

    assert repo.count_all() == 2


# --- 基本行为 ----------------------------------------------------------------


def test_index_document_writes_vectors_for_all_chunks(db_session, service, embedder) -> None:
    document = make_document(db_session)
    chunks = [add_chunk(db_session, document, i, f"chunk text {i}") for i in range(3)]

    result = service.index_document(document.id)

    assert result == IndexingResult(embedded_count=3, skipped_count=0)
    for chunk in chunks:
        assert embedding_of(db_session, chunk) == pytest.approx(
            embedder.embed_text(chunk.text), abs=1e-6
        )


def test_index_document_returns_result_with_both_counts(db_session, service) -> None:
    """结果必须能区分"没有待办"和"文档不存在"。

    这是 F-39 教训的另一面：一个只返回 embedded_count 的接口，
    会把"这篇文档是空的"和"这篇文档早就索引过了"显示成同一个 0。
    """
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "already done", embedding=[0.1] * 384)
    add_chunk(db_session, document, 1, "needs work")

    result = service.index_document(document.id)

    assert result.embedded_count == 1
    assert result.skipped_count == 1


def test_index_document_only_embeds_the_given_document(db_session, service) -> None:
    """不能顺手把别的文档也编码了——多做的功和少做的一样是 bug。"""
    first = make_document(db_session, content_hash="a" * 64)
    second = make_document(db_session, content_hash="b" * 64)
    chunk_of_first = add_chunk(db_session, first, 0, "first document")
    chunk_of_second = add_chunk(db_session, second, 0, "second document")

    service.index_document(first.id)

    assert embedding_of(db_session, chunk_of_first) is not None
    assert embedding_of(db_session, chunk_of_second) is None


def test_index_document_raises_for_unknown_document(service) -> None:
    """id 对不上时抛错，而不是"找不到 → 返回 0 → 看起来成功了"。

    静默返回 0 的后果：调用方以为索引跑完了，实际一个向量都没生成，
    而且不会有任何报错——直到发现检索结果莫名其妙地少。
    和 ChunkRepository.update_embedding 是同一条原则。
    """
    with pytest.raises(ValueError):
        service.index_document(uuid.uuid4())


# --- 编码的内容对不对 --------------------------------------------------------


def test_index_document_embeds_the_chunk_text(db_session, service, embedder) -> None:
    """必须编码 chunk 自己的文本，不能张冠李戴（比如错用了文档标题）。"""
    document = make_document(db_session, title="A Very Different Title")
    chunk = add_chunk(db_session, document, 0, "self attention mechanism")

    service.index_document(document.id)

    assert embedding_of(db_session, chunk) == pytest.approx(
        embedder.embed_text("self attention mechanism"), abs=1e-6
    )


def test_index_document_makes_chunks_searchable(db_session, service, repo, embedder) -> None:
    """回填的**唯一意义**：让片段能被检索到。

    单独测"向量写进去了"是不够的——写进去了但检索还查不到（比如漏了归一化、
    或者列的维度不对），系统一样是坏的。所以这里把写路径和读路径串起来验证。
    """
    document = make_document(db_session)
    chunk = add_chunk(db_session, document, 0, "self attention mechanism")
    query = embedder.embed_text("self attention mechanism")

    assert repo.search_similar(query, top_k=5) == []  # 回填之前：查不到

    service.index_document(document.id)

    results = repo.search_similar(query, top_k=5)  # 回填之后：查得到
    assert len(results) == 1
    assert results[0].chunk_id == chunk.id
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


# --- 幂等 --------------------------------------------------------------------


def test_index_document_is_idempotent(db_session, service) -> None:
    """重复调用不会重新编码已经有过向量的片段。"""
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "text")
    add_chunk(db_session, document, 1, "more text")

    first = service.index_document(document.id)
    second = service.index_document(document.id)

    assert first == IndexingResult(embedded_count=2, skipped_count=0)
    # 第二次一个都不该重新编码——不是"重算一遍结果一样"，而是**根本没调模型**
    assert second == IndexingResult(embedded_count=0, skipped_count=2)


def test_index_document_does_not_overwrite_existing_vector(db_session, service, embedder) -> None:
    """已经有向量的片段必须原样保留。

    "跳过"和"重算"的区别在这里才看得出来：如果实现是"把整篇文档重新编码一遍"，
    这条旧向量会被换成新值，测试就会红。为什么这是错的？
    因为已经有向量的片段可能来自**另一个模型**——覆盖它未必是对的，
    而且白白多花一次编码。
    """
    document = make_document(db_session)
    old_vector = embedder.embed_text("banana bread recipe")
    chunk = add_chunk(db_session, document, 0, "self attention", embedding=old_vector)

    service.index_document(document.id)

    assert embedding_of(db_session, chunk) == pytest.approx(old_vector, abs=1e-6)


def test_index_document_with_nothing_pending_does_not_touch_the_client(db_session, service) -> None:
    """没有待办时不应该调用模型——空列表进 encode 是浪费，某些版本还会抛异常。"""

    class ExplodingClient(FakeEmbeddingClient):
        def embed_texts(self, texts):  # type: ignore[override]
            raise AssertionError("没有待办时不该调用 embedding client")

    document = make_document(db_session)
    add_chunk(db_session, document, 0, "done", embedding=[0.1] * 384)

    result = IndexingService(db_session, ExplodingClient()).index_document(document.id)

    assert result == IndexingResult(embedded_count=0, skipped_count=1)


# --- 全库回填 ----------------------------------------------------------------


def test_index_all_pending_covers_every_document(db_session, service) -> None:
    first = make_document(db_session, content_hash="a" * 64)
    second = make_document(db_session, content_hash="b" * 64)
    add_chunk(db_session, first, 0, "alpha")
    add_chunk(db_session, first, 1, "beta")
    add_chunk(db_session, second, 0, "gamma")

    result = service.index_all_pending()

    assert result == IndexingResult(embedded_count=3, skipped_count=0)
    assert count_pending(db_session) == 0


def test_index_all_pending_skips_already_indexed_chunks(db_session, service, embedder) -> None:
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "old", embedding=embedder.embed_text("old"))
    add_chunk(db_session, document, 1, "new")

    result = service.index_all_pending()

    assert result == IndexingResult(embedded_count=1, skipped_count=1)


def test_index_all_pending_on_empty_database_is_a_noop(db_session, service) -> None:
    assert service.index_all_pending() == IndexingResult(embedded_count=0, skipped_count=0)


# --- 客户端返回数量不对时不能静默 --------------------------------------------------


def test_index_document_rejects_wrong_number_of_vectors(db_session) -> None:
    """客户端少返回向量时必须炸掉，不能默默少写几个。

    这条挡的是一个**真实会静默发生**的 bug：实现里用 zip() 配对，
    而 zip 在两边长度不等时会悄悄在短的那边停下——少返回的那部分片段
    永远拿不到向量，而且不会有任何报错。所以实现里显式比了长度。
    """

    class ShortClient(FakeEmbeddingClient):
        def embed_texts(self, texts):  # type: ignore[override]
            return super().embed_texts(texts)[:-1]  # 故意少给一个

    document = make_document(db_session)
    add_chunk(db_session, document, 0, "one")
    add_chunk(db_session, document, 1, "two")

    with pytest.raises(ValueError):
        IndexingService(db_session, ShortClient()).index_document(document.id)


def test_index_all_pending_rejects_wrong_number_of_vectors(db_session) -> None:
    class ShortClient(FakeEmbeddingClient):
        def embed_texts(self, texts):  # type: ignore[override]
            return []

    document = make_document(db_session)
    add_chunk(db_session, document, 0, "one")

    with pytest.raises(ValueError):
        IndexingService(db_session, ShortClient()).index_all_pending()
