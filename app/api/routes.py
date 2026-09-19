"""HTTP 路由。

这一层只做三件事，多一件都不做：

1. 解析请求（交给 pydantic schema）
2. 调用 service
3. 把结果/异常翻译成 HTTP（状态码、JSON）

它不写 SQL、不做业务判断、不控制事务——`session.commit()` 在 service 里，
所以路由里看不到 commit，这不是遗漏，是分层的结果。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.schemas import (
    DocumentCreateRequest,
    DocumentCreateResponse,
    DocumentListItem,
)
from app.core.database import get_db_session
from app.repositories.documents import DocumentRepository
from app.services.ingestion import DocumentIngestionService

router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    """存活探针：只回答「服务进程还在不在」，不检查数据库。"""
    return {"status": "ok"}


@router.post("/documents", status_code=status.HTTP_201_CREATED)
def create_document(
    payload: DocumentCreateRequest,
    # Annotated[Session, Depends(...)] 是 FastAPI 现在推荐的写法。
    # 老写法 session: Session = Depends(get_db_session) 也能跑，但它把
    # 一个函数调用放在了默认值的位置——那在 Python 里通常是 bug 的温床
    # （默认值只在定义时求值一次）。Annotated 把这个信息挪回类型位置。
    session: Annotated[Session, Depends(get_db_session)],
) -> DocumentCreateResponse:
    """导入一篇文档。同一篇内容重复提交会返回原文档而不是报错。"""
    service = DocumentIngestionService(session)

    try:
        result = service.ingest(
            title=payload.title, source=payload.source, content=payload.content
        )
    except ValueError as exc:
        # service 抛的是 ValueError（它不知道 HTTP 是什么），
        # 由这一层翻译成状态码——"业务异常 → HTTP 语义"的映射是 API 层的职责。
        # 用 422 而不是 400，是为了和 pydantic 的校验错误保持一致：
        # 客户端不需要区分"这个字段格式不对"和"这个字段内容不合法"，
        # 两者都是"请求能读懂，但内容不接受"。
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    return DocumentCreateResponse(
        document_id=result.document_id,
        chunk_count=result.chunk_count,
        created=result.created,
    )


@router.get("/documents")
def list_documents(
    session: Annotated[Session, Depends(get_db_session)],
) -> list[DocumentListItem]:
    """列出已导入的文档，最新的在前。"""
    documents = DocumentRepository(session).list_documents()
    return [DocumentListItem.model_validate(document) for document in documents]
