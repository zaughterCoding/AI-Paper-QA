"""documents 表的读写。"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.tables import Document


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        # Repository 持有 session，但不负责开启/提交/关闭它——
        # 事务的边界由 Service 决定（见 app/services/ingestion.py）。
        self.session = session

    def create_document(self, title: str, source: str, content_hash: str) -> Document:
        """插入一篇文档，返回带数据库生成字段的对象。"""
        document = Document(title=title, source=source, content_hash=content_hash)
        self.session.add(document)

        # flush 是必须的：id 那一列的 default=uuid.uuid4 是在**发 SQL 的时候**
        # 才应用的，不 flush 的话 document.id 还是 None，后面没法给 chunk 当外键。
        # 注意 flush ≠ commit：SQL 发出去了，但事务没提交，还能回滚。
        self.session.flush()
        return document

    def get_by_content_hash(self, content_hash: str) -> Document | None:
        """按内容指纹查文档，用于避免重复导入。

        命中唯一索引，不是全表扫描。找不到时返回 None——
        这是 Python 里表达"可能没有"的标准方式（而不是返回 -1 或抛异常）。
        """
        return self.session.scalar(
            select(Document).where(Document.content_hash == content_hash)
        )
