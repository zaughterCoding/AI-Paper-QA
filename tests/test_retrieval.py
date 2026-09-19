"""向量检索的测试（Task 8 的 repository 层 + Task 9 的 service 层）。

检索在代码里分成两层，失败方式完全不同：

    RetrievalService.retrieve(question)         ← service：问题 → 向量 → 片段
        └── ChunkRepository.search_similar()    ← repository：向量 → 片段

- **repository 层**测的是 *SQL 写对没有*：排序方向、NULL 过滤、JOIN 不扇出。
  写错的表现是**返回错误的片段**，通常能一眼看出来。
- **service 层**测的是 *编排对不对*：校验在不在编码之前、每件事各做几次。
  写错的表现往往是**结果看起来完全正常**，只是多编码了一次问题、
  或者对空问题白白调了一次模型。这类问题不会有任何症状，
  只会在账单和耗时才看得出来——所以只能靠测试钉住。

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
from app.services.retrieval import DEFAULT_TOP_K, MAX_TOP_K, RetrievalService
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


# --- 检索服务（Task 9）-------------------------------------------------------
#
# 这一组测的不是"能不能查出正确的片段"（上面已经测过了），而是**编排**：
# 校验在不在编码之前、每件事各做几次、参数有没有原样传下去。
#
# 为什么值得单独测：这类错误**没有症状**。多编码一次问题，结果一模一样，
# 只是慢了一倍、贵了一倍；漏了校验，空问题也照样能返回一堆片段。
# 靠肉眼和手工验收都发现不了，只能靠把次数钉死。


class _CountingEmbeddingClient:
    """包住假客户端，记录每次编码用的文本。

    为什么不用 mock 库：这里只需要"记下来"这一个动作，
    一个三行的类比 `Mock()` 加一串断言更直白，而且它**保留真实行为**——
    返回值仍然是那个能被检索命中的哈希向量，所以同一个替身既能计数
    又能当正常客户端用。
    """

    def __init__(self) -> None:
        self._inner = FakeEmbeddingClient()
        self.encoded_texts: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.encoded_texts.append(text)
        return self._inner.embed_text(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.encoded_texts.extend(texts)
        return self._inner.embed_texts(texts)


class _SearchSpy:
    """包住 `search_similar`，记录每次调用的参数。

    为什么用"包一层"而不是 monkeypatch 整个类：
    `RetrievalService.__init__` 里自己 new 了一个 `ChunkRepository`
    （和 `IndexingService` 的写法一致），外面拿不到那个实例。
    但 **Python 查找实例属性优先于类属性**，所以给 `service.chunks` 这个实例
    挂一个同名属性就能拦住调用，同时 `_original` 仍然是真正的方法——
    查库逻辑照常执行，只是多记了一笔。
    """

    def __init__(self, repository: ChunkRepository) -> None:
        self._original = repository.search_similar
        self.calls: list[dict] = []

    def __call__(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        self.calls.append({"query_embedding": query_embedding, "top_k": top_k})
        return self._original(query_embedding=query_embedding, top_k=top_k)


@pytest.fixture
def counter() -> _CountingEmbeddingClient:
    return _CountingEmbeddingClient()


@pytest.fixture
def service(db_session: Session, counter: _CountingEmbeddingClient) -> RetrievalService:
    return RetrievalService(db_session, counter)


@pytest.fixture
def search_spy(service: RetrievalService) -> _SearchSpy:
    spy = _SearchSpy(service.chunks)
    service.chunks.search_similar = spy  # type: ignore[method-assign]
    return spy


def seed_chunks(db_session: Session, embedder: FakeEmbeddingClient) -> None:
    """一篇文档、三个主题不同的片段。"""
    document = make_document(db_session)
    for index, text in enumerate(
        ["banana bread recipe", "self attention mechanism", "gradient descent optimizer"]
    ):
        add_chunk(db_session, document, index, text, embedder.embed_text(text))


# --- 校验 --------------------------------------------------------------------


def test_retrieve_rejects_an_empty_question(service) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        service.retrieve("")


def test_retrieve_rejects_a_whitespace_only_question(service) -> None:
    """只输入空格 / 换行的"空问题"同样没有语义。

    直接判 `not question` 会漏掉它们——字符串本身非空，但编码出来的向量
    要么是全零（和任何向量都没有意义明确的相似度），要么只反映标点。
    """
    with pytest.raises(ValueError, match="must not be empty"):
        service.retrieve("   \n\t  ")


@pytest.mark.parametrize("top_k", [0, -1, -100, MAX_TOP_K + 1, 1000])
def test_retrieve_rejects_top_k_out_of_range(service, top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k must be between"):
        service.retrieve("self attention", top_k=top_k)


@pytest.mark.parametrize("top_k", [1, DEFAULT_TOP_K, MAX_TOP_K])
def test_retrieve_accepts_top_k_within_range(db_session, embedder, counter, top_k: int) -> None:
    """边界值本身必须是合法的。

    只测"超界的要报错"是不够的——把上限写成 0 也能让那条测试通过，
    但服务就再也检索不了任何东西了。这是 H 变异（Task 8c）留下的教训：
    **越界的检查必须配一条界内的检查**，否则阈值本身写错也没人发现。
    """
    seed_chunks(db_session, embedder)

    assert len(RetrievalService(db_session, counter).retrieve("self attention", top_k=top_k)) <= top_k


# --- 编排：调用次数与顺序 ----------------------------------------------------


def test_retrieve_embeds_the_question_exactly_once(db_session, embedder, counter) -> None:
    """问题只编码一次。

    多编码一次在结果上**完全看不出来**，只是每次提问都白等一次模型前向、
    多付一次钱。这类"结果正确但成本翻倍"的问题没有任何症状，
    只有把次数钉死才拦得住。
    """
    seed_chunks(db_session, embedder)

    RetrievalService(db_session, counter).retrieve("self attention")

    assert counter.encoded_texts == ["self attention"]


def test_retrieve_searches_the_database_exactly_once(db_session, embedder, service, search_spy) -> None:
    """查库也只查一次——不能"先查一次看看，再查一次取结果"。

    注意这里用的是 `service` / `search_spy` 两个 fixture：**被测对象**和
    **挂 spy 的对象**必须是同一个实例。

    第一版不是这样写的——测试体里又 `RetrievalService(db_session, counter)`
    new 了一个，同时声明了 `search_spy` fixture 却没用它。那一版**恰好还是对的**
    （因为我在新实例上另挂了一个 spy），但被测对象和 spy 从此分成两条线：
    下一个人把多余的 new 删掉、改用 fixture 的 spy 时，
    断言就会挂在一个没人调用的对象上——而写死 `== 1` 时那是**失败**而不是假绿，
    所以它会红一次，然后被人"修"成 `== 0` 或者干脆删掉。

    让两者始终是同一个实例，是这类计数测试唯一稳妥的写法。
    """
    seed_chunks(db_session, embedder)

    service.retrieve("self attention")

    assert len(search_spy.calls) == 1


def test_retrieve_passes_the_question_to_the_embedder_unchanged(db_session, embedder, counter) -> None:
    """送去编码的是调用方给的原字符串，没有被 strip / 截断 / 改写。

    保留原样是为了**可复现**：出问题时能拿同一个字符串重跑，
    不用猜中间那一层动过什么。
    """
    seed_chunks(db_session, embedder)

    RetrievalService(db_session, counter).retrieve("  self attention  ")

    assert counter.encoded_texts == ["  self attention  "]


def test_retrieve_forwards_top_k_to_the_repository(db_session, embedder, counter) -> None:
    """top_k 必须原样传下去，不能被吞掉、也不能被写死成默认值。"""
    seed_chunks(db_session, embedder)
    service = RetrievalService(db_session, counter)
    spy = _SearchSpy(service.chunks)
    service.chunks.search_similar = spy  # type: ignore[method-assign]

    service.retrieve("self attention", top_k=2)

    assert spy.calls[0]["top_k"] == 2


def test_retrieve_uses_the_default_top_k_when_omitted(db_session, embedder, counter) -> None:
    """不传 top_k 时用的是 DEFAULT_TOP_K，而不是"全部返回"。"""
    seed_chunks(db_session, embedder)
    document = make_document(db_session, content_hash="b" * 64)
    for index in range(DEFAULT_TOP_K + 3):
        text = f"self attention variant {index}"
        add_chunk(db_session, document, index, text, embedder.embed_text(text))

    results = RetrievalService(db_session, counter).retrieve("self attention")

    assert len(results) == DEFAULT_TOP_K


def test_retrieve_validates_the_question_before_encoding(service, counter) -> None:
    """空问题的校验必须发生在编码**之前**。

    顺序反了的话，一个没有语义的问题照样会占用一次模型前向——
    而且某些模型对空字符串返回全零向量，检索会返回一堆看似随机的片段，
    把"输入不合法"伪装成"检索质量差"。
    """
    with pytest.raises(ValueError):
        service.retrieve("")

    assert counter.encoded_texts == []


def test_retrieve_validates_top_k_before_encoding(db_session, embedder, counter) -> None:
    """top_k 越界同样要在编码之前拦下——理由同上。"""
    seed_chunks(db_session, embedder)

    with pytest.raises(ValueError):
        RetrievalService(db_session, counter).retrieve("self attention", top_k=0)

    assert counter.encoded_texts == []


def test_retrieve_does_not_commit(db_session, embedder, counter, monkeypatch) -> None:
    """检索是**只读**操作，不该提交事务。

    这条不是吹毛求疵：Task 11 的 Ask API 会在一个请求里同时用这个 session
    做检索（将来还要写 QA 日志）。检索路径里混进一次 commit，
    事务边界就不再是调用方说了算了。
    """

    def fail() -> None:
        raise AssertionError("retrieve() 不该提交事务")

    seed_chunks(db_session, embedder)
    monkeypatch.setattr(db_session, "commit", fail)

    assert RetrievalService(db_session, counter).retrieve("self attention")


# --- 端到端：结果本身 --------------------------------------------------------


def test_retrieve_returns_chunks_ordered_by_similarity(db_session, embedder, counter) -> None:
    """把两层接起来跑一遍：问题经过编码、查库，回到最相关的片段。"""
    seed_chunks(db_session, embedder)

    results = RetrievalService(db_session, counter).retrieve("self attention layer", top_k=3)

    assert [result.text for result in results][0] == "self attention mechanism"
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_spans_multiple_documents(db_session, embedder, counter) -> None:
    """检索是全局的：答案可以在任何一篇文档里。

    这是这一层存在的意义——问题不绑定到某一篇文档，
    用户也不需要知道答案在哪篇里。
    """
    first = make_document(db_session, title="Paper A", content_hash="a" * 64)
    second = make_document(db_session, title="Paper B", content_hash="b" * 64)
    add_chunk(db_session, first, 0, "banana bread recipe", embedder.embed_text("banana bread recipe"))
    add_chunk(db_session, second, 0, "self attention", embedder.embed_text("self attention"))

    results = RetrievalService(db_session, counter).retrieve("self attention", top_k=5)

    assert results[0].title == "Paper B"


def test_retrieve_returns_empty_list_on_an_empty_database(service) -> None:
    """库里什么都没有时返回空列表，而不是抛异常。

    "没找到"和"出错了"是两回事：前者是一个正常的业务结果
    （语料还没导入、问题问的是语料之外的东西），
    后者才是需要调用方处理的情况。混成一种，上层就没法区分
    "该告诉用户没找到"还是"该报警"。
    """
    assert service.retrieve("self attention") == []


def test_retrieve_ignores_chunks_without_vectors(db_session, embedder, counter) -> None:
    """还没回填向量的片段不能被检索到——这层不能绕过 repository 的过滤。"""
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "self attention", embedding=None)

    assert RetrievalService(db_session, counter).retrieve("self attention") == []
