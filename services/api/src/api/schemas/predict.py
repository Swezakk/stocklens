"""DTO ML-прогнозов (ml-spec §8.3, §10). Строго отделены от ORM-моделей."""

from datetime import date

from pydantic import BaseModel, Field, field_validator
from stocklens_core.enums import TrendDirection


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


class VolatilityForecastPoint(BaseModel):
    """Одна точка ряда «прогноз vs реализованная волатильность» (ml-spec §10)."""

    date: date
    forecast: float | None = None
    realized: float | None = None


class VolatilityRegime(BaseModel):
    """Режим волатильности: прогноз vs исторический квантиль (ml-spec §9).

    ``protected_namespaces=()`` — нет полей ``model_*``, но конфигурация единообразна с
    остальными DTO этого модуля.
    """

    model_config = {"protected_namespaces": ()}

    ticker: str
    predicted_for: date
    volatility: float
    threshold: float
    is_elevated: bool
    quantile: float
    lookback: int


class VolatilityForecastHistoryOut(BaseModel):
    """История прогнозов волатильности с реализованными значениями для графика (ml-spec §10).

    ``protected_namespaces=()`` — поле ``model`` конфликтует с защищённым неймспейсом Pydantic v2.

    ``live_metrics`` — скользящий QLIKE по созревшим парам (forecast + realized оба присутствуют).
    Сопоставим с офлайновым ``metrics_vs_baseline`` (walk-forward); None если пар < 10.
    ``live_sample_size`` — количество пар после joint-маски (конечные положительные h/RV/b).
    ``forward`` — режим волатильности по последнему сохранённому прогнозу (ml-spec §9);
    None, если нет сохранённых прогнозов или недостаточно истории rv_target для квантиля.
    """

    model_config = {"protected_namespaces": ()}

    ticker: str
    model: str | None
    model_version: str | None
    metrics_vs_baseline: VolatilityMetrics | None
    points: list[VolatilityForecastPoint]
    live_metrics: VolatilityMetrics | None = None
    live_sample_size: int = 0
    forward: VolatilityRegime | None = None


class ForecastRefreshSummary(BaseModel):
    """Итог пакетной генерации прогнозов (возвращается сервисным методом, не HTTP-ответом)."""

    generated: int
    skipped: int
    failed: int
    total: int


class ForecastRefreshOut(BaseModel):
    """Ответ эндпоинта POST /bot/forecasts/refresh."""

    accepted: bool
    reason: str | None = None


class ShapContribution(BaseModel):
    """Вклад одной фичи в предсказание тренда (ml-spec §8.3).

    SHAP передаётся списком пар «фича → вклад», а не словарём ``dict[str, float]`` из §8.3:
    список сохраняет порядок фич и моделирует каждую запись явной DTO-моделью.
    """

    feature: str
    value: float


class TrendPredictionIn(BaseModel):
    """Запрос прогноза направления тренда по тикеру."""

    ticker: str = Field(min_length=1, max_length=16)

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        """Тикеры MOEX — в верхнем регистре; нормализуем, чтобы 'sber' находил 'SBER'."""
        return value.strip().upper()


class TrendPredictionOut(BaseModel):
    """Ответ прогноза направления тренда (ml-spec §8.3).

    ``protected_namespaces=()`` — поле ``model_version`` иначе конфликтует с защищённым
    неймспейсом Pydantic v2 (``model_*``).

    Вероятностная оценка, а не торговый сигнал: ``direction`` выводится из ``prob_up``.
    ``shap`` — список вкладов фич (см. :class:`ShapContribution`), ``base_value`` —
    ожидаемое значение модели до учёта вкладов.
    """

    model_config = {"protected_namespaces": ()}

    ticker: str
    predicted_for: date
    horizon_days: int
    prob_up: float = Field(ge=0, le=1)
    direction: TrendDirection
    shap: list[ShapContribution]
    base_value: float
    model_version: str
