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
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.database import get_db_session
from app.main import create_app
from app.models.tables import Chunk, Document

# 400 个词，配合默认 chunk_size=180 / overlap=30 → 3 个片段
CONTENT = " ".join(f"token{i}" for i in range(400))


def _count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


@pytest.fixture
def client(db_session: Session) -> Iterator[TestClient]:
    """一个"连着测试数据库"的应用实例。

    get_db_session 本来会自己开一条真实连接，那样测试就绕开了 conftest 的
    事务隔离——数据真的落库，测试之间互相污染，跑完还得手工清理。
    dependency_overrides 是 FastAPI 提供的依赖替换机制：请求进来时，
    Depends(get_db_session) 拿到的是我们塞进去的这个测试 session。
    """
    app = create_app()
    app.dependency_overrides[get_db_session] = lambda: db_session

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
