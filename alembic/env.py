import sys
from logging.config import fileConfig

import psycopg2
from alembic import context

from repo_knowledge.config import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def get_db_url() -> str:
    # Just in case we need a URL string (e.g. for offline mode)
    return f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine.
    """
    url = get_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Here we create a raw psycopg2 DBAPI connection and pass it to Alembic,
    avoiding SQLAlchemy completely.
    """
    # Create DBAPI connection
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    )

    context.configure(
        connection=conn,
        target_metadata=target_metadata,
        compare_type=True,
    )

    try:
        with context.begin_transaction():
            context.run_migrations()
    finally:
        conn.close()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
