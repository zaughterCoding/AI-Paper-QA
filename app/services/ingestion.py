"""文档导入服务：把一段原始文本变成"文档 + 一组片段"存进数据库。"""

import hashlib
import uuid
from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.rag.chunking import TextChunk, TextChunker
from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRepository


@dataclass(frozen=True)
class IngestionResult:
    """导入结果。返回它而不是直接返回 Document 对象，是为了让上层
    （API 层）拿到一个**和数据库无关**的简单结构，不用去碰 ORM 对象。
    """

    document_id: uuid.UUID
    chunk_count: int
    created: bool  # True=新建，False=内容已存在，复用了原来的文档


class DocumentIngestionService:
    def __init__(self, session: Session, chunker: TextChunker | None = None) -> None:
        self.session = session
        # chunker 允许从外面传进来（依赖注入）：测试时可以塞一个返回固定片段的
        # 假切分器，从而只测服务的编排逻辑，不受真实切分算法影响。
        self.chunker = chunker or TextChunker()
        self.documents = DocumentRepository(session)
        self.chunks = ChunkRepository(session)

    def ingest(self, title: str, source: str, content: str) -> IngestionResult:
        """导入一篇文档。同一个内容重复导入不会产生第二份数据。"""
        self._validate(title=title, content=content)

        # 指纹：对内容本身做 sha256。同样的文字永远得到同样的值，
        # 所以它能当"这篇文档是否已经导过"的判据。
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        existing = self.documents.get_by_content_hash(content_hash)
        if existing is not None:
            # 幂等：重复导入返回原文档，而不是报错、也不是再存一份。
            # 注意 chunk_count 是从**库里数出来的**，不是 0——
            # 调用方拿到的信息应该和第一次导入时一样。
            return IngestionResult(
                document_id=existing.id,
                chunk_count=self.chunks.count_by_document(existing.id),
                created=False,
            )

        try:
            document = self.documents.create_document(
                title=title, source=source, content_hash=content_hash
            )
            # 先建文档再切分，是为了让 chunks 有 document_id 可挂。
            # 切分本身是纯计算（不碰数据库），失败也不会留下脏数据。
            text_chunks: list[TextChunk] = self.chunker.chunk(content)
            saved = self.chunks.create_chunks(document_id=document.id, chunks=text_chunks)

            # 所有片段都准备好了才提交——要么全成，要么全不成。
            # 如果在这里之前任何一步抛异常，事务回滚，数据库里不会出现
            # "有文档但没有片段"的残缺状态。
            self.session.commit()
        except IntegrityError:
            # 竞态兜底：上面的"先查再插"在多请求并发时有窗口期——
            # 两个请求可能同时查到"不存在"，然后都去插入。
            # 唯一约束会挡住第二个，抛 IntegrityError。
            # 这里回滚后重查，把它当成"已存在"处理，用户看到的仍然是幂等结果。
            self.session.rollback()
            existing = self.documents.get_by_content_hash(content_hash)
            if existing is None:
                raise  # 不是重复导致的错误，如实往上抛，别吞掉
            return IngestionResult(
                document_id=existing.id,
                chunk_count=self.chunks.count_by_document(existing.id),
                created=False,
            )

        return IngestionResult(
            document_id=document.id,
            chunk_count=len(saved),
            created=True,
        )

    @staticmethod
    def _validate(title: str, content: str) -> None:
        """业务规则校验。

        为什么 API 层已经有 pydantic 校验了，这里还要再查一遍？
        因为 service 不只被 API 调用——脚本、测试、将来的评估程序都会直接调它。
        校验放在最靠近数据的地方，任何入口都绕不过去。
        """
        if not title.strip():
            raise ValueError("title must not be empty")
        if not content.strip():
            raise ValueError("content must not be empty")
