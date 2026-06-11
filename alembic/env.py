"""Окружение Alembic: миграции схемы StockLens.

target_metadata берётся из пакета stocklens-core — единственного источника истины
по схеме данных. URL подключения разрешается без прямого чтения os.environ:
сначала программно заданный sqlalchemy.url (путь интеграционных тестов),
иначе — DATABASE_URL через pydantic-settings.
"""

from logging.config import fileConfig

from alembic import context
from pydantic import PostgresDsn
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine, pool
from stocklens_core.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


class _MigrationSettings(BaseSettings):
    """Миграциям нужен только синхронный DSN — полный CoreSettings (с REDIS_URL
    и async-DSN) здесь избыточен и потребовал бы лишних переменных окружения."""

    database_url: PostgresDsn


def _resolve_database_url() -> str:
    """Вернуть синхронный DSN: программный override либо DATABASE_URL из окружения.

    model_validate({}) читает env-источники pydantic-settings; прямой вызов
    конструктора mypy strict трактует как пропуск обязательного аргумента.
    """
    configured_url = config.get_main_option("sqlalchemy.url")
    if configured_url:
        return configured_url
    return str(_MigrationSettings.model_validate({}).database_url)


def run_migrations_offline() -> None:
    """Сгенерировать SQL миграций без подключения к БД (offline-режим)."""
    context.configure(
        url=_resolve_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Применить миграции через синхронный движок (online-режим)."""
    connectable = create_engine(_resolve_database_url(), poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
