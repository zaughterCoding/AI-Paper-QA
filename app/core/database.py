from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。Alembic 靠它拿到全部表的定义。"""


# 注意：create_engine 此刻并不会真的去连数据库，它是"惰性"的，
# 第一次执行 SQL 时才建立连接。所以这里 import 时不会因为数据库没启动而报错。
engine = create_engine(
    get_settings().database_url,
    # 连接池里的连接可能已经被数据库单方面关掉了（超时、重启）。
    # pool_pre_ping 在借出连接前先做一次轻量探测，失效就换一条，
    # 避免服务运行几小时后突然开始报 "server closed the connection"。
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI 依赖：每个请求一个数据库会话，请求结束自动关闭。

    用 yield 而不是 return，才能保证无论请求成功还是抛异常，finally 都会执行。
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
