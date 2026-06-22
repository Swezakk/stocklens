"""Конфигурация дашборда через pydantic-settings (DESIGN.md §6, §7, §8, §9).

Все параметры читаются из переменных окружения, инжектируемых Docker Compose.
Прямое обращение к os.environ запрещено (инвариант проекта).

Часть полей читается из env-имён API-контракта через validation_alias:
- API_URL              — базовый URL API (общий с другими сервисами).
- AUTH_OWNER_USERNAME  — имя владельца; пароль вводит пользователь на гейте (DESIGN §7).
"""

import functools

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardSettings(BaseSettings):
    """Параметры сервиса stocklens-dashboard."""

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    api_base_url: str = Field(
        default="http://api:8000",
        validation_alias=AliasChoices("API_URL", "api_base_url"),
        description="Базовый URL StockLens API (без /api/v1).",
    )
    api_prefix: str = Field(
        default="/api/v1",
        description="Префикс версии API; все маршруты живут под ним.",
    )
    auth_username: str = Field(
        default="admin",
        validation_alias=AliasChoices("AUTH_OWNER_USERNAME", "auth_username"),
        description="Имя владельца для логина в API (DESIGN §7).",
    )
    request_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description="Таймаут HTTP-запроса к API в секундах.",
    )
    cache_ttl_seconds: int = Field(
        default=600,
        ge=0,
        description="TTL кэша st.cache_data: дневные данные обновляются нечасто (DESIGN §8).",
    )
    token_refresh_margin_seconds: int = Field(
        default=90,
        ge=60,
        description="Запас до истечения токена для проактивного refresh; ≥ leeway API 60s.",
    )
    news_corpus_period_days: int = Field(
        default=30,
        ge=1,
        description="Глубина корпуса новостей для агрегатов тональности в днях (DESIGN §9).",
    )
    news_corpus_max_articles: int = Field(
        default=1000,
        ge=1,
        description="Жёсткий потолок статей при добивании корпуса пагинацией (DESIGN §9).",
    )


@functools.lru_cache(maxsize=1)
def get_settings() -> DashboardSettings:
    """Вернуть кэшированный экземпляр DashboardSettings (читается из env один раз)."""
    return DashboardSettings()
