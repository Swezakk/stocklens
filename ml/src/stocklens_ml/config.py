"""Конфигурация оффлайн ML-проекта через pydantic-settings (ml-spec §2.1).

Все параметры — из окружения; прямое чтение os.environ запрещено. `DATABASE_URL` —
sync-DSN (postgresql+psycopg://...), модели читаются из прод-БД на чтение при обучении.
"""

from datetime import date

from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Горизонт прогноза в торговых днях (ml-spec §4; D8).
HORIZON_DAYS: int = 5

#: Начало обучающего окна — пост-2022 (структурный разрыв 28.02–24.03.2022; D8).
TRAIN_START_DEFAULT: date = date(2022, 4, 1)


class MlSettings(BaseSettings):
    """Параметры обучения и регистрации ML-моделей StockLens.

    `protected_namespaces=()` — поле `model_alias` начинается с `model_`, что иначе
    конфликтует с защищённым неймспейсом Pydantic v2 (ml-spec §8.6).
    """

    database_url: PostgresDsn
    mlflow_tracking_uri: str = "http://localhost:5000"
    volatility_model_name: str = "stocklens-volatility"
    trend_model_name: str = "stocklens-trend"
    model_alias: str = "production"
    horizon_days: int = HORIZON_DAYS
    train_start: date = TRAIN_START_DEFAULT

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )
