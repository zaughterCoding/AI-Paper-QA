"""导入这个包就等于"注册了所有表"。

Alembic 生成迁移时靠 Base.metadata 找到全部模型；如果某个模型模块从没被导入过，
它就不在 metadata 里，迁移会静默地漏掉那张表。所以在 __init__ 里统一导出，
让 `import app.models` 一定加载全部表定义。
"""

from app.models.tables import Chunk, Document, QALog

__all__ = ["Chunk", "Document", "QALog"]
