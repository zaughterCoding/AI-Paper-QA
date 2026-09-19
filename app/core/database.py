from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Base class for every ORM model. Alembic reads the table definitions from it."""


# create_engine is lazy: it does not connect here, only on the first query. Importing
# this module therefore succeeds even while the database is down.
engine = create_engine(
    get_settings().database_url,
    # A pooled connection may have been closed server-side (restart, idle timeout).
    # pool_pre_ping probes it before handing it out and replaces it if it is dead.
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency: one session per request.

    Yielded rather than returned so the session is closed whether the request
    succeeds or raises.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
