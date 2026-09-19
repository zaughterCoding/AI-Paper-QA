"""documents 表的读写。"""

import uuid

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

    def list_documents(self) -> list[Document]:
        """列出全部文档，最新的在前。

        scalars() 返回的是迭代器，外面套一层 list() 让它变成真正的列表：
        迭代器只能遍历一次，返回给上层一个"用完就空"的对象是个隐藏陷阱。
        """
        return list(
            self.session.scalars(select(Document).order_by(Document.created_at.desc()))
        )

    def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        """按主键查文档。

        用的是 `session.get()` 而不是 `select().where(id == ...)`：
        `get()` 会先查 session 的 identity map（本次会话里已经加载过的对象），
        命中就不发 SQL。对"按主键取一个对象"这种最常见的需求，
        `get()` 是 SQLAlchemy 推荐的标准写法。
        """
        return self.session.get(Document, document_id)

    def get_by_content_hash(self, content_hash: str) -> Document | None:
        """按内容指纹查文档，用于避免重复导入。

        命中唯一索引，不是全表扫描。找不到时返回 None——
        这是 Python 里表达"可能没有"的标准方式（而不是返回 -1 或抛异常）。
        """
        return self.session.scalar(
            select(Document).where(Document.content_hash == content_hash)
        )
