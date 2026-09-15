"""pytest 共享夹具（fixture）。

核心思路：测试跑在一个**独立的测试库**上，且每个测试都包在一个事务里，
测完回滚。这样测试可以随意增删数据，彼此之间零污染，也不用写清理代码。

    开发库 paperqa        ← 你的真实数据，测试绝不碰
    测试库 paperqa_test   ← 每次跑测试重建表结构，跑完数据全部回滚
"""

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import Base
from app.models import tables  # noqa: F401  必须导入，否则 Base.metadata 里没有表

TEST_DB_NAME = "paperqa_test"


def replace_database(url: str, database: str) -> str:
    """把连接串末尾的库名换掉，其余部分（用户、密码、主机、端口）保持不变。

    "postgresql+psycopg://postgres:postgres@localhost:5432/paperqa"
        → "postgresql+psycopg://postgres:postgres@localhost:5432/paperqa_test"
    """
    return url.rsplit("/", 1)[0] + "/" + database


@pytest.fixture(scope="session")
def test_engine() -> Iterator[Engine]:
    """整个测试会话共用一个引擎：建库、建表结构、跑完释放。

    scope="session" 表示只执行一次，不是每个测试都重建一遍——建表是慢操作。
    """
    settings = get_settings()

    # 1) 连到系统的 postgres 库去建测试库（不能连一个不存在的库去建它自己）
    #    AUTOCOMMIT 是必须的：CREATE DATABASE 不能在事务里执行
    admin_engine = create_engine(
        replace_database(settings.database_url, "postgres"), isolation_level="AUTOCOMMIT"
    )
    with admin_engine.connect() as conn:
        exists = conn.execute(
            text("select 1 from pg_database where datname = :name"), {"name": TEST_DB_NAME}
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    admin_engine.dispose()

    # 2) 连测试库，装 pgvector 扩展并建表
    engine = create_engine(replace_database(settings.database_url, TEST_DB_NAME))
    with engine.begin() as conn:
        # vector 扩展不能自动迁移生成，必须手动建
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    yield engine
    engine.dispose()


@pytest.fixture
def db_session(test_engine: Engine) -> Iterator[Session]:
    """每个测试拿到的会话，测试结束自动回滚。

    这里的 join_transaction_mode="create_savepoint" 是关键：
    服务层（Task 5 之后）会调用 session.commit()。如果不管它，commit 会真的
    把数据写进测试库，回滚就失效了、测试之间开始互相污染。
    create_savepoint 让 session.commit() 只释放一个 SAVEPOINT，
    外层那个大事务还在我们手里，最后依然能整体回滚。
    """
    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
