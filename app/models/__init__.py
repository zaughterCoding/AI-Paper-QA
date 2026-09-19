"""Importing this package registers every table.

Alembic builds migrations from ``Base.metadata``, which only knows about models that
have actually been imported. Re-exporting them here means ``import app.models`` always
loads the full schema instead of silently missing a table.
"""

from app.models.tables import Chunk, Document, QALog

__all__ = ["Chunk", "Document", "QALog"]
