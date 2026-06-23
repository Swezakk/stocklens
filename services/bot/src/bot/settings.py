"""Конфигурация Telegram-бота через pydantic-settings (DESIGN.md §11, §13).

Все параметры читаются из переменных окружения, инжектируемых Docker Compose.
Прямое обращение к os.environ запрещено (инвариант проекта).

Бот — доверенный backend-потребитель API: в отличие от дашборда (где пароль вводит
пользователь на гейте), бот держит учётные данные владельца в env и сам получает JWT.
Секреты (TELEGRAM_BOT_TOKEN, AUTH_OWNER_PASSWORD) — SecretStr, чтобы не утечь в логи.
"""

import functools
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    """Параметры сервиса stocklens-bot.

    Имена env берутся из общего контракта проекта через validation_alias (TELEGRAM_BOT_TOKEN,
    API_URL, AUTH_OWNER_USERNAME, AUTH_OWNER_PASSWORD) — те же переменные, что у api/dashboard.
    """

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    telegram_bot_token: SecretStr = Field(
        validation_alias=AliasChoices("TELEGRAM_BOT_TOKEN", "telegram_bot_token"),
        description="Токен бота от @BotFather.",
    )
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
    auth_password: SecretStr = Field(
        validation_alias=AliasChoices("AUTH_OWNER_PASSWORD", "auth_password"),
        description="Пароль владельца для получения JWT у API (бот — backend, хранит в env).",
    )
    request_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description="Таймаут HTTP-запроса к API в секундах.",
    )
    token_refresh_margin_seconds: int = Field(
        default=90,
        ge=60,
        description="Запас до истечения токена для проактивного refresh; ≥ leeway API 60s.",
    )
    digest_chat_id: int = Field(
        validation_alias=AliasChoices("DIGEST_CHAT_ID", "digest_chat_id"),
        description="Telegram chat_id владельца для ежедневного дайджеста.",
    )
    alert_poll_interval_minutes: int = Field(
        default=30,
        gt=0,
        validation_alias=AliasChoices("ALERT_POLL_INTERVAL_MINUTES", "alert_poll_interval_minutes"),
        description="Интервал опроса сработавших алертов в минутах.",
    )
    digest_hour_msk: int = Field(
        default=8,
        ge=0,
        le=23,
        validation_alias=AliasChoices("DIGEST_HOUR_MSK", "digest_hour_msk"),
        description="Час отправки дайджеста по московскому времени (0-23).",
    )
    digest_minute_msk: int = Field(
        default=30,
        ge=0,
        le=59,
        validation_alias=AliasChoices("DIGEST_MINUTE_MSK", "digest_minute_msk"),
        description="Минута отправки дайджеста по московскому времени (0-59).",
    )
    log_pretty: bool = Field(
        default=False,
        description="True → ConsoleRenderer (dev), False → JSONRenderer (прод).",
    )
    heartbeat_path: Path = Field(
        default=Path("/tmp/bot-heartbeat"),
        description="Файл-сигнал живости для healthcheck контейнера.",
    )


@functools.lru_cache(maxsize=1)
def get_settings() -> BotSettings:
    """Вернуть кэшированный экземпляр BotSettings (читается из env один раз)."""
    return BotSettings.model_validate({})
