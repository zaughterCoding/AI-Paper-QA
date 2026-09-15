"""create documents chunks qa_logs

Revision ID: a5f439213056
Revises:
Create Date: 2026-09-14 20:27:38.155239

这是项目的第一个迁移：一次性建出三张表。

迁移文件是**数据库结构的历史记录**。它的价值在于：任何一台机器上，
只要按顺序执行这些文件，就能得到和别人完全一样的表结构。
所以迁移文件一旦提交就不要再改内容——要改结构就新写一个迁移。

注意：本文件由 `alembic revision --autogenerate` 生成后手工修订过两处，
见下面 upgrade() 里的注释。
"""

from typing import Sequence, Union

import pgvector.sqlalchemy  # 手工补的：下面用到了 VECTOR 类型，autogenerate 不会自动加这行
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a5f439213056"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """建表。"""
    # 手工补的：pgvector 的 vector 类型不是 PostgreSQL 自带的，必须先装扩展。
    # 没有这一句，全新的数据库执行到建 chunks 表时会报
    # "type vector does not exist"。IF NOT EXISTS 保证重复执行不报错。
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("source", sa.String(length=500), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("content_hash"),
    )
    op.create_table(
        "qa_logs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("retrieved_chunk_ids", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        # 384 维向量列，对应 app/core/config.py 里的 EMBEDDING_DIM
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=384), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        # 数据库层面的级联删除：删文档时自动删掉它的所有 chunk
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_chunk_index"),
    )
    op.create_index(op.f("ix_chunks_document_id"), "chunks", ["document_id"], unique=False)


def downgrade() -> None:
    """回滚：删表。

    注意这里**没有** DROP EXTENSION vector。扩展是数据库级别的共享资源，
    可能还有别的表在用；而且重建它的代价远小于误删的风险。
    迁移的 downgrade 应该只撤销自己创建的东西，不越界。
    """
    op.drop_index(op.f("ix_chunks_document_id"), table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("qa_logs")
    op.drop_table("documents")
