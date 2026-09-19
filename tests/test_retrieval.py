"""向量检索的测试（Task 8：只测 repository 层）。

这一层是 RAG 的**取数环节**：给一个查询向量，从库里找出最相似的 top_k 个片段。
它不负责"把问题变成向量"（Task 9 的 RetrievalService 才做那件事）。

关于测试数据的一个关键决定：**用 384 维的真实维度，不用 3 维假向量**。
计划书原本建议"用假 3 维向量"，但 `chunks.embedding` 列的类型写死了 `vector(384)`，
3 维向量根本插不进去。更重要的是——**测试用的维度必须和真实维度一致**，
否则测出来的是"另一个东西能不能跑"，而不是"这个系统行不行"。

向量的来源是 `tests/fakes.py` 的假客户端：它把词哈希到 384 维里的某一维，
所以**共享词汇的文本向量更相似**。这让"排序是否正确"可以被真正验证，
而不是拿一堆互不相关的随机数假装在测检索。
"""

import uuid

import pytest
from sqlalchemy.orm import Session

from app.models.tables import Chunk, Document
from app.repositories.chunks import ChunkRepository, RetrievedChunk
from tests.fakes import FakeEmbeddingClient


@pytest.fixture
def embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
def repo(db_session: Session) -> ChunkRepository:
    return ChunkRepository(db_session)


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


# --- 空库与边界情况 ----------------------------------------------------------


def test_search_on_empty_database_returns_empty_list(repo, embedder) -> None:
    assert repo.search_similar(embedder.embed_text("anything"), top_k=5) == []


def test_search_rejects_top_k_below_one(repo, embedder) -> None:
    """top_k 是 0 或负数时，数据库对 LIMIT -1 会直接报错。
    在这里拦下来，报错信息才有意义。"""
    with pytest.raises(ValueError):
        repo.search_similar(embedder.embed_text("anything"), top_k=0)


def test_search_skips_chunks_without_embedding(db_session, repo, embedder) -> None:
    """还没有生成向量的片段不能出现在结果里。

    这不是假设的情况：Task 5 导入文档时只存文本，向量是稍后回填的，
    中间这段窗口里库里合法地存在大量 embedding 为 NULL 的片段。
    """
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "self attention", embedding=None)
    add_chunk(
        db_session, document, 1, "self attention", embedding=embedder.embed_text("self attention")
    )

    results = repo.search_similar(embedder.embed_text("self attention"), top_k=10)

    assert len(results) == 1
    assert results[0].chunk_index == 1


# --- 排序：这是检索的核心 ----------------------------------------------------


def test_search_returns_most_similar_chunk_first(db_session, repo, embedder) -> None:
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "banana bread recipe", embedder.embed_text("banana bread recipe"))
    add_chunk(db_session, document, 1, "self attention mechanism", embedder.embed_text("self attention mechanism"))
    add_chunk(db_session, document, 2, "gradient descent optimizer", embedder.embed_text("gradient descent optimizer"))

    results = repo.search_similar(embedder.embed_text("self attention layer"), top_k=3)

    assert results[0].text == "self attention mechanism"


def test_search_orders_by_decreasing_similarity(db_session, repo, embedder) -> None:
    """按相似度从高到低排。

    查询用 4 个词，四个片段分别共享 4/3/2/1 个词，
    余弦相似度是 1.0 / 0.866 / 0.707 / 0.5——**四个值互不相同**，
    所以可以断言完整的顺序，而不是只断言"谁排第一"。
    （如果分数有并列，并列项之间的顺序是未定义的，断言顺序就会随机失败。）
    """
    document = make_document(db_session)
    texts = ["alpha beta gamma delta", "alpha beta gamma", "alpha beta", "alpha"]
    for index, text in enumerate(texts):
        add_chunk(db_session, document, index, text, embedder.embed_text(text))

    results = repo.search_similar(embedder.embed_text("alpha beta gamma delta"), top_k=4)

    assert [result.text for result in results] == texts
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_search_score_of_identical_text_is_close_to_one(db_session, repo, embedder) -> None:
    """score 是余弦相似度：文本完全相同时应该是 1.0。

    这个断言语义而不只是数字——如果哪天有人把 `1 - 距离` 写成了 `距离`，
    或者忘了归一化，这里会立刻失败。
    """
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "self attention", embedder.embed_text("self attention"))

    results = repo.search_similar(embedder.embed_text("self attention"), top_k=1)

    assert results[0].score == pytest.approx(1.0, abs=1e-6)


# --- top_k 与结果的形状 ------------------------------------------------------


def test_search_respects_top_k(db_session, repo, embedder) -> None:
    document = make_document(db_session)
    for index in range(5):
        text = f"chunk number {index}"
        add_chunk(db_session, document, index, text, embedder.embed_text(text))

    assert len(repo.search_similar(embedder.embed_text("chunk number 0"), top_k=2)) == 2
    assert len(repo.search_similar(embedder.embed_text("chunk number 0"), top_k=5)) == 5


def test_search_returns_each_chunk_at_most_once(db_session, repo, embedder) -> None:
    """JOIN documents 不能把结果放大。

    这里要防的是经典的 JOIN 扇出：如果一个 chunk 关联出多行 documents，
    同一条 chunk 会在结果里出现多次，top_k 就会悄悄少给几条。
    """
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "self attention", embedder.embed_text("self attention"))

    results = repo.search_similar(embedder.embed_text("self attention"), top_k=10)

    assert len(results) == 1
    assert len({result.chunk_id for result in results}) == len(results)


def test_search_returns_metadata_from_both_tables(db_session, repo, embedder) -> None:
    """结果必须同时带上 chunks 表和 documents 表的字段。

    带上 title 是为了回答问题时能给出处——"这段话出自哪篇文档"。
    没有它就只剩一段孤立的文本，用户没法核实。
    """
    document = make_document(db_session, title="Attention Is All You Need")
    chunk = add_chunk(
        db_session, document, 3, "self attention", embedder.embed_text("self attention")
    )

    result = repo.search_similar(embedder.embed_text("self attention"), top_k=1)[0]

    assert isinstance(result, RetrievedChunk)
    assert result.chunk_id == chunk.id
    assert result.document_id == document.id
    assert result.title == "Attention Is All You Need"
    assert result.chunk_index == 3
    assert result.text == "self attention"


def test_search_spans_multiple_documents(db_session, repo, embedder) -> None:
    """检索是**全局**的，不限于某一篇文档。

    这是 RAG 相对"把整篇论文塞给模型"的关键差别之一：
    问题可以从任意一篇文档里找答案。
    """
    first = make_document(db_session, title="Paper A", content_hash="a" * 64)
    second = make_document(db_session, title="Paper B", content_hash="b" * 64)
    add_chunk(db_session, first, 0, "banana bread recipe", embedder.embed_text("banana bread recipe"))
    add_chunk(db_session, second, 0, "self attention", embedder.embed_text("self attention"))

    results = repo.search_similar(embedder.embed_text("self attention"), top_k=5)

    assert results[0].title == "Paper B"


# --- 回填向量 ----------------------------------------------------------------


def test_update_embedding_writes_the_vector(db_session, repo, embedder) -> None:
    document = make_document(db_session)
    chunk = add_chunk(db_session, document, 0, "self attention", embedding=None)
    vector = embedder.embed_text("self attention")

    repo.update_embedding(chunk.id, vector)
    db_session.flush()

    # 从数据库重新读一遍（而不是看内存里的对象），确认真的写进去了
    stored = db_session.get(Chunk, chunk.id)
    assert stored is not None
    assert stored.embedding is not None
    assert list(stored.embedding) == pytest.approx(vector, abs=1e-6)


def test_update_embedding_makes_chunk_searchable(db_session, repo, embedder) -> None:
    """回填的最终目的是让片段**能被检索到**——这才是这个方法的完整意义。"""
    document = make_document(db_session)
    chunk = add_chunk(db_session, document, 0, "self attention", embedding=None)
    query = embedder.embed_text("self attention")

    assert repo.search_similar(query, top_k=5) == []

    repo.update_embedding(chunk.id, embedder.embed_text("self attention"))

    results = repo.search_similar(query, top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == chunk.id


def test_update_embedding_overwrites_existing_vector(db_session, repo, embedder) -> None:
    """重复回填应该覆盖旧向量，而不是报错或留下两份。"""
    document = make_document(db_session)
    chunk = add_chunk(db_session, document, 0, "text", embedder.embed_text("banana bread recipe"))
    new_vector = embedder.embed_text("self attention")

    repo.update_embedding(chunk.id, new_vector)

    stored = db_session.get(Chunk, chunk.id)
    assert stored is not None
    assert list(stored.embedding) == pytest.approx(new_vector, abs=1e-6)

    results = repo.search_similar(embedder.embed_text("self attention"), top_k=1)
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_update_embedding_raises_for_unknown_chunk(repo, embedder) -> None:
    """id 对不上时抛错而不是静默跳过。

    静默跳过会让"有些片段永远没有向量"这种 bug 一直藏着，
    直到某天发现检索结果莫名其妙地差，而且完全不知道从哪查起。
    """
    with pytest.raises(ValueError):
        repo.update_embedding(uuid.uuid4(), embedder.embed_text("self attention"))
