"""Integration tests: Alembic migrations against real PostgreSQL 16 (testcontainers).

Каждый тест получает свежий контейнер — состояние не разделяется между тестами.
URL прокидывается программно через `sqlalchemy.url` (env.py отдаёт ему приоритет),
поэтому CoreSettings и полный набор переменных окружения здесь не нужны.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect
from stocklens_core.models import Base
from testcontainers.postgres import PostgresContainer

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_TABLES = set(Base.metadata.tables.keys())
ALEMBIC_VERSION_TABLE = "alembic_version"


@pytest.fixture
def pg_url() -> Iterator[str]:
    """Свежий PostgreSQL 16 на каждый тест; DSN — под psycopg v3."""
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as container:
        yield container.get_connection_url()


def _alembic_config(database_url: str) -> Config:
    """Конфиг Alembic с абсолютными путями (тесты не зависят от cwd) и явным URL."""
    config = Config(str(_REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_REPO_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_upgrade_head_matches_core_metadata(pg_url: str) -> None:
    """После `upgrade head` схема БД эквивалентна метаданным core.

    Сравниваются структура (таблицы, колонки, типы, констрейнты) И server_default:
    на PostgreSQL `compare_server_default=True` сравнивает дефолты на стороне СУБД
    и не даёт ложных диффов (проверено эмпирически на этой схеме: jsonb/now()/bool).
    """
    command.upgrade(_alembic_config(pg_url), "head")

    engine = create_engine(pg_url)
    with engine.connect() as connection:
        migration_context = MigrationContext.configure(
            connection,
            opts={"compare_type": True, "compare_server_default": True},
        )
        diffs = compare_metadata(migration_context, Base.metadata)

    assert diffs == [], f"Schema differs from metadata: {diffs}"


def test_upgrade_head_creates_all_tables_and_version_stamp(pg_url: str) -> None:
    """`upgrade head` создаёт все таблицы спеки §6 плюс служебную alembic_version."""
    command.upgrade(_alembic_config(pg_url), "head")

    engine = create_engine(pg_url)
    inspector = inspect(engine)
    actual_tables = set(inspector.get_table_names())

    assert actual_tables == EXPECTED_TABLES | {ALEMBIC_VERSION_TABLE}


def test_downgrade_base_reverts_schema_and_upgrade_reapplies(pg_url: str) -> None:
    """Цикл `upgrade head → downgrade base → upgrade head` проходит без ошибок.

    После downgrade в БД не остаётся доменных таблиц (только alembic_version),
    повторный upgrade воспроизводит схему — миграция обратима и идемпотентна.
    """
    config = _alembic_config(pg_url)
    command.upgrade(config, "head")
    command.downgrade(config, "base")

    engine = create_engine(pg_url)
    inspector = inspect(engine)
    tables_after_downgrade = set(inspector.get_table_names())
    assert tables_after_downgrade <= {ALEMBIC_VERSION_TABLE}, (
        f"Domain tables left after downgrade: {tables_after_downgrade}"
    )

    command.upgrade(config, "head")
    inspector = inspect(engine)
    assert set(inspector.get_table_names()) == EXPECTED_TABLES | {ALEMBIC_VERSION_TABLE}
