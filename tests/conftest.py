"""Shared pytest fixtures.

Tests run against a separate test database, and every test is wrapped in a transaction
that is rolled back afterwards, so tests cannot pollute each other.
"""

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import Base
from app.models import tables  # noqa: F401  required so Base.metadata knows the tables

TEST_DB_NAME = "paperqa_test"


def replace_database(url: str, database: str) -> str:
    """Swap the database name at the end of a connection URL, keeping the rest."""
    return url.rsplit("/", 1)[0] + "/" + database


@pytest.fixture(scope="session")
def test_engine() -> Iterator[Engine]:
    """One engine for the whole session: create the database and its schema once.

    Creating tables is slow, so it happens once rather than per test.
    """
    settings = get_settings()

    # Connect to the system `postgres` database to create the test database; a
    # database cannot be created from a connection to itself.
    # AUTOCOMMIT is required: CREATE DATABASE cannot run inside a transaction.
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

    engine = create_engine(replace_database(settings.database_url, TEST_DB_NAME))
    with engine.begin() as conn:
        # The vector extension is not created by create_all.
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    yield engine
    engine.dispose()


@pytest.fixture
def db_session(test_engine: Engine) -> Iterator[Session]:
    """A session per test, rolled back when the test ends.

    `join_transaction_mode="create_savepoint"` keeps that true even though the service
    layer calls `session.commit()`: the commit then only releases a SAVEPOINT nested in
    the outer transaction, which this fixture still rolls back.
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
