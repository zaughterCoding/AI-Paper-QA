"""Alembic 运行环境。

这里做两件事：
1. 告诉 Alembic 「数据库在哪」——直接读 app 的 Settings，不从 alembic.ini 读。
   这样数据库地址全项目只有一个来源（.env），不会出现"改了 .env，
   但 alembic.ini 还是旧地址"这种两处不一致的经典故障。
2. 告诉 Alembic 「表长什么样」——把 Base.metadata 交给它，才能自动比对差异。
"""

from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context

# 必须 import app.models，让三张表全部注册进 Base.metadata。
# 少导入一个模型，autogenerate 就会认为"这张表不该存在"，进而生成删表语句。
from app.core.config import get_settings
from app.core.database import Base
from app.models import tables  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# autogenerate（自动生成迁移）靠它比对"代码里的表"和"数据库里的表"的差异
target_metadata = Base.metadata


def get_url() -> str:
    return get_settings().database_url


def run_migrations_offline() -> None:
    """离线模式：不连数据库，只把 SQL 打印出来。

    用途是"生成 SQL 交给 DBA 审阅后再手工执行"这类场景。
    """
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # 让 autogenerate 也能察觉「列类型改了」（比如 vector(384) → vector(768)），
        # 默认是关的，很多"改类型没生成迁移"的坑都来自这里
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：真的连上数据库执行迁移，这是日常用的模式。"""
    connectable = create_engine(get_url(), poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
