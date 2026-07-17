from logging.config import fileConfig

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

    Here we configure Alembic to use a URL directly.
    Alembic requires an Engine or a URL to work in online mode because it extracts
    the dialect from the Engine. Passing a raw DBAPI connection directly to connection=
    is not supported as it expects a SQLAlchemy connection. We will pass a URL instead.
    """
    import os

    # We read env variables because tests might override POSTGRES_HOST / PORT
    host = os.getenv("POSTGRES_HOST", POSTGRES_HOST)
    port = int(os.getenv("POSTGRES_PORT", POSTGRES_PORT))

    url = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{host}:{port}/{POSTGRES_DB}"
    )

    from sqlalchemy import create_engine

    connectable = create_engine(url)

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
