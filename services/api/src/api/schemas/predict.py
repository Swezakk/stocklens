"""DTO ML-прогнозов (ml-spec §8.3). Строго отделены от ORM-моделей."""

from datetime import date

from pydantic import BaseModel, Field, field_validator


class VolatilityPredictionIn(BaseModel):
    """Запрос прогноза волатильности по тикеру."""

    ticker: str = Field(min_length=1, max_length=16)

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        """Тикеры MOEX — в верхнем регистре; нормализуем, чтобы 'sber' находил 'SBER'."""
        return value.strip().upper()


class VolatilityMetrics(BaseModel):
    """Метрики модели-победителя против naive baseline (walk-forward QLIKE/RMSE)."""

    qlike: float
    qlike_baseline: float
    rmse: float


class VolatilityPredictionOut(BaseModel):
    """Ответ прогноза 5-дневной волатильности.

    ``protected_namespaces=()`` — поля ``model``/``model_version`` иначе конфликтуют с
    защищённым неймспейсом Pydantic v2 (``model_*``).
    """

    model_config = {"protected_namespaces": ()}

    ticker: str
    predicted_for: date
    horizon_days: int
    volatility: float
    model: str
    model_version: str
    metrics_vs_baseline: VolatilityMetrics
