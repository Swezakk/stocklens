"""Tests for stocklens_core.settings — CoreSettings field validation and env loading."""

import pytest
from pydantic import ValidationError
from stocklens_core.settings import CoreSettings


def _load_settings() -> CoreSettings:
    """Загрузить CoreSettings из переменных окружения через model_validate.

    model_validate({}) читает env-источники pydantic-settings без обращения
    к сгенерированному pydantic-__init__, поэтому mypy не требует явной передачи полей.
    """
    return CoreSettings.model_validate({})


def test_loads_valid_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """CoreSettings загружается корректно при наличии всех обязательных переменных окружения."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/stocklens")
    monkeypatch.setenv(
        "DATABASE_URL_ASYNC", "postgresql+asyncpg://user:pass@localhost:5432/stocklens"
    )
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    settings = _load_settings()

    assert str(settings.database_url).startswith("postgresql+psycopg://")
    assert str(settings.database_url_async).startswith("postgresql+asyncpg://")
    assert str(settings.redis_url).startswith("redis://")


def test_missing_database_url_raises_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Отсутствие DATABASE_URL должно вызвать ValidationError при создании CoreSettings."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL_ASYNC", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    with pytest.raises(ValidationError):
        _load_settings()


def test_default_tickers_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    """tickers_universe по умолчанию равно 'IMOEX', если переменная не задана."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/stocklens")
    monkeypatch.setenv(
        "DATABASE_URL_ASYNC", "postgresql+asyncpg://user:pass@localhost:5432/stocklens"
    )
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.delenv("TICKERS_UNIVERSE", raising=False)

    settings = _load_settings()

    assert settings.tickers_universe == "IMOEX"


def test_custom_tickers_universe_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """TICKERS_UNIVERSE из окружения переопределяет дефолт."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/stocklens")
    monkeypatch.setenv(
        "DATABASE_URL_ASYNC", "postgresql+asyncpg://user:pass@localhost:5432/stocklens"
    )
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("TICKERS_UNIVERSE", "SBER,GAZP,LKOH")

    settings = _load_settings()

    assert settings.tickers_universe == "SBER,GAZP,LKOH"


def test_psycopg_scheme_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Схема postgresql+psycopg принимается как валидный PostgresDsn."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@db:5432/stocklens")
    monkeypatch.setenv("DATABASE_URL_ASYNC", "postgresql+asyncpg://user:pass@db:5432/stocklens")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")

    settings = _load_settings()

    assert "psycopg" in str(settings.database_url)


def test_asyncpg_scheme_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Схема postgresql+asyncpg принимается как валидный PostgresDsn."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@db:5432/stocklens")
    monkeypatch.setenv("DATABASE_URL_ASYNC", "postgresql+asyncpg://user:pass@db:5432/stocklens")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")

    settings = _load_settings()

    assert "asyncpg" in str(settings.database_url_async)


def test_settings_isolated_from_real_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Тест явно очищает переменные, чтобы не зависеть от реального окружения."""
    for var in ("DATABASE_URL", "DATABASE_URL_ASYNC", "REDIS_URL", "TICKERS_UNIVERSE"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ValidationError):
        _load_settings()
