"""文档 API 的测试。

和 test_ingestion.py 不同，这里的测试走完整的 HTTP 栈：

    TestClient → 路由 → 依赖注入 → Service → Repository → PostgreSQL

所以它验证的东西也更高一层：状态码对不对、JSON 字段名对不对、
请求体校验有没有生效。Service 内部的逻辑细节仍然由 test_ingestion.py 负责。
"""

import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.database import get_db_session
from app.main import create_app
from app.models.tables import Chunk, Document
from app.rag.embeddings import get_embedding_client
from tests.fakes import FakeEmbeddingClient

# 400 个词，配合默认 chunk_size=180 / overlap=30 → 3 个片段
CONTENT = " ".join(f"token{i}" for i in range(400))


def _count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


@pytest.fixture
def embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
def client(db_session: Session, embedder: FakeEmbeddingClient) -> Iterator[TestClient]:
    """一个"连着测试数据库、且不加载真模型"的应用实例。

    这里替换了**两个**依赖：

    - `get_db_session` 本来会自己开一条真实连接，那样测试就绕开了 conftest 的
      事务隔离——数据真的落库，测试之间互相污染，跑完还得手工清理。
    - `get_embedding_client` 是进程级单例，**必须**替换掉。否则第一个 POST 请求
      会去加载真实的 sentence-transformers 模型：要下载几百 MB、要几秒，
      而且单例一旦建好，整个测试会话都甩不掉它。

    dependency_overrides 是 FastAPI 提供的依赖替换机制：请求进来时，
    Depends(...) 拿到的是我们塞进去的这个对象。
    """
    app = create_app()
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[get_embedding_client] = lambda: embedder

    # 用 with 而不是直接 TestClient(app)：with 会触发 startup/shutdown 生命周期，
    # 将来加了启动时预热模型之类的逻辑，测试才会走一样的路径。
    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_create_document_returns_id_and_chunk_count(client: TestClient, db_session: Session) -> None:
    response = client.post(
        "/documents",
        json={"title": "Attention", "source": "arxiv:1706.03762", "content": CONTENT},
    )

    assert response.status_code == 201

    body = response.json()
    uuid.UUID(body["document_id"])  # 不是合法 UUID 会直接抛异常
    assert body["chunk_count"] == 3

    # 接口说成功了，库里就必须真的有数据——只信响应不信数据库是假绿
    assert _count(db_session, Document) == 1
    assert _count(db_session, Chunk) == 3


def test_create_document_reports_whether_it_was_new(client: TestClient) -> None:
    payload = {"title": "Attention", "source": "arxiv", "content": CONTENT}

    first = client.post("/documents", json=payload).json()
    second = client.post("/documents", json=payload).json()

    assert first["created"] is True
    assert second["created"] is False
    assert second["document_id"] == first["document_id"]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="all-fields-missing"),
        pytest.param({"title": "", "source": "arxiv", "content": CONTENT}, id="empty-title"),
        pytest.param({"title": "t", "source": "", "content": CONTENT}, id="empty-source"),
        pytest.param({"title": "t", "source": "arxiv", "content": ""}, id="empty-content"),
        pytest.param({"title": "t" * 301, "source": "arxiv", "content": CONTENT}, id="title-too-long"),
        # 下面两个能通过 pydantic（长度不为 0），但过不了 service 的空白校验。
        # 它们验证的是「service 抛的 ValueError 被翻译成了 422」这条路。
        pytest.param({"title": "   ", "source": "arxiv", "content": CONTENT}, id="blank-title"),
        pytest.param({"title": "t", "source": "arxiv", "content": "  \n\t "}, id="blank-content"),
    ],
)
def test_create_document_rejects_invalid_payload(
    client: TestClient, db_session: Session, payload: dict
) -> None:
    response = client.post("/documents", json=payload)

    assert response.status_code == 422
    # 校验失败必须什么都没写进去
    assert _count(db_session, Document) == 0
    assert _count(db_session, Chunk) == 0


# --- 导入之后向量有没有真的生成（F-39 的回归测试）--------------------------------
#
# 这几条是**回归测试**：它们钉住的是那个曾经掉进任务缝里的步骤——
# 设计书写了"生成 embedding / 保存向量"，但没有任何任务实现它，
# 结果 update_embedding 写好了却没人调用，库里全是 NULL 向量。
# 如果哪天有人重构时又把这一步弄丢了，这里会立刻红。


def _count_without_embedding(session: Session) -> int:
    return session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.embedding.is_(None))
    ) or 0


def _payload(content: str = CONTENT) -> dict:
    return {"title": "Attention", "source": "arxiv:1706.03762", "content": content}


def test_create_document_embeds_every_chunk(
    client: TestClient, db_session: Session, embedder: FakeEmbeddingClient
) -> None:
    """POST /documents 之后，每个片段都必须有向量。

    这是 F-39 的直接回归测试：只断言"文档存进去了"是不够的——
    那个 bug 里文档和片段都好好地存进去了，**只是没有向量**，
    而检索永远看不到它们。
    """
    body = client.post("/documents", json=_payload()).json()

    assert body["chunk_count"] == 3
    assert body["embedded_chunk_count"] == 3
    assert _count_without_embedding(db_session) == 0


def test_create_document_stores_vectors_that_match_the_chunk_text(
    client: TestClient, db_session: Session, embedder: FakeEmbeddingClient
) -> None:
    """向量必须由**片段自己的文本**算出来，而不是别的什么东西。"""
    client.post("/documents", json=_payload())

    chunks = list(db_session.scalars(select(Chunk).order_by(Chunk.chunk_index)))
    for chunk in chunks:
        assert list(chunk.embedding) == pytest.approx(
            embedder.embed_text(chunk.text), abs=1e-6
        )


def test_create_document_does_not_reembed_on_duplicate(
    client: TestClient, db_session: Session
) -> None:
    """重复导入不会白干一遍编码。

    注意断言的是 `embedded_chunk_count == 0`（这次一个新向量都没写），
    而不是"向量还在"——后者证明不了"没有重复计算"。
    """
    client.post("/documents", json=_payload())

    second = client.post("/documents", json=_payload()).json()

    assert second["created"] is False
    assert second["chunk_count"] == 3
    assert second["embedded_chunk_count"] == 0


def test_create_document_backfills_vectors_lost_earlier(
    client: TestClient, db_session: Session
) -> None:
    """自愈：向量缺失时，重新导入同一篇内容会把它们补回来。

    这个行为很重要，因为导入和索引是**两个事务**：如果索引那一步失败了
    （进程被 kill、内存不够），文档已经存下来了，但响应是 500。
    客户端重试同样的内容时，ingest 会走"重复导入"分支，
    而索引会再跑一次、把缺的向量补上。**请求失败不等于数据白丢。**
    """
    client.post("/documents", json=_payload())

    # 人为制造"向量丢了"的状态：和索引步骤失败一次的后果等价
    db_session.execute(update(Chunk).values(embedding=None), execution_options={"synchronize_session": False})
    db_session.flush()
    db_session.expire_all()
    assert _count_without_embedding(db_session) == 3

    body = client.post("/documents", json=_payload()).json()

    assert body["created"] is False
    assert body["embedded_chunk_count"] == 3
    assert _count_without_embedding(db_session) == 0


def test_create_document_fails_loudly_when_indexing_fails(db_session: Session) -> None:
    """索引那一步失败时，请求必须**失败**，绝不能吞掉异常返回一个"看起来成功"的 201。

    这条是变异测试逼出来的：把路由改成 `try/except` 吞掉索引异常、
    照样返回 201 并报告 embedded_chunk_count=0，**原来所有的测试都还是绿的**。

    为什么这个行为必须被钉住：吞掉之后，调用方以为一切正常，
    实际文档存进去了、chunk 也数得出来，**只有一个向量都没有**——
    文档能列出来，但检索永远查不到它。
    这正是 F-39 那个缺口能藏那么久的原因：**它没有任何信号**。

    同时验证"请求失败 ≠ 数据白丢"：文档和片段已经在一个独立的事务里提交了，
    所以客户端重试时能走"重复导入"分支把向量补上（见上一条测试）。
    """

    class BrokenClient(FakeEmbeddingClient):
        def embed_texts(self, texts):  # type: ignore[override]
            raise RuntimeError("model exploded")

    app = create_app()
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[get_embedding_client] = lambda: BrokenClient()

    # raise_server_exceptions=False：默认情况下 TestClient 会把服务端异常直接抛给测试，
    # 那样就看不到状态码了。这里要断言的恰恰是"返回了什么状态码"。
    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/documents", json=_payload())

    assert response.status_code == 500
    # 文档留下了（导入和索引是两个事务），但向量一个都没有
    assert _count(db_session, Document) == 1
    assert _count(db_session, Chunk) == 3
    assert _count_without_embedding(db_session) == 3


def test_list_documents_starts_empty(client: TestClient) -> None:
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == []


def test_list_documents_returns_imported_documents(client: TestClient) -> None:
    client.post("/documents", json={"title": "A", "source": "src-a", "content": CONTENT})
    client.post("/documents", json={"title": "B", "source": "src-b", "content": CONTENT + " extra"})

    body = client.get("/documents").json()

    assert {item["title"] for item in body} == {"A", "B"}
    # 字段集合是接口契约的一部分：多返回字段会泄露内部结构，少返回字段会让客户端崩
    assert all(set(item) == {"id", "title", "source", "created_at"} for item in body)


def test_list_documents_returns_newest_first(client: TestClient, db_session: Session) -> None:
    """顺序用**写死的时间戳**来测，而不是靠"先插的应该在前"。

    后者在 Windows 上可能翻车：两次插入如果落在同一个系统时钟刻度里，
    created_at 会完全相同，排序结果就变成不确定的。测试必须不依赖这种运气。
    """
    db_session.add_all(
        [
            Document(
                title="older",
                source="s",
                content_hash="a" * 64,
                created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            ),
            Document(
                title="newer",
                source="s",
                content_hash="b" * 64,
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
        ]
    )
    db_session.flush()

    body = client.get("/documents").json()

    assert [item["title"] for item in body] == ["newer", "older"]
