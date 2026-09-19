"""Alembic runtime environment.

Two things happen here. Alembic is told where the database is, read from the application's
own Settings rather than from alembic.ini, so the URL has a single source of truth and
cannot drift between the two. And it is told what the schema looks like, by handing it
Base.metadata -- that is what lets autogenerate diff code against database.
"""

from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context

# Importing app.models registers all three tables in Base.metadata. Missing one would make
# autogenerate conclude that table should not exist and emit a DROP for it.
from app.core.config import get_settings
from app.core.database import Base
from app.models import tables  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# autogenerate diffs "tables in code" against "tables in the database" using this.
target_metadata = Base.metadata


def get_url() -> str:
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Offline mode: emit the SQL without connecting, for review or manual execution."""
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Without this, changing a column's type (vector(384) -> vector(768)) produces no
        # migration at all. It is off by default.
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Online mode: connect and apply the migrations. This is the usual one."""
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
