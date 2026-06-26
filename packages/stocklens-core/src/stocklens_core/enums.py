"""Доменные перечисления StockLens — единственный источник истины для статусов, меток и типов.

Все перечисления наследуют str, что гарантирует JSON-сериализацию без приведения типов
и совместимость с SQLAlchemy Enum(native_enum=False).
"""

from enum import StrEnum


class CollectorRunStatus(StrEnum):
    """Статус запуска сборщика данных."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class SentimentLabel(StrEnum):
    """Тональность новостной статьи по результатам NLP-классификации."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class PredictionKind(StrEnum):
    """Вид ML-прогноза."""

    VOLATILITY = "volatility"
    TREND = "trend"


class TrendDirection(StrEnum):
    """Направление прогноза тренда: классификация вверх/вниз (ml-spec §8.3).

    Вероятностная оценка, а не торговый сигнал: ``UP`` соответствует ``prob_up >= 0.5``.
    """

    UP = "up"
    DOWN = "down"


class Currency(StrEnum):
    """Поддерживаемые валюты для котировок ЦБ и дивидендных выплат."""

    RUB = "RUB"
    USD = "USD"
    EUR = "EUR"
    CNY = "CNY"


class AlertKind(StrEnum):
    """Вид Telegram-алерта для подписок пользователя."""

    SENTIMENT_SPIKE = "sentiment_spike"
    VOLATILITY_REGIME = "volatility_regime"
    DIVIDEND_UPCOMING = "dividend_upcoming"
    PRICE_LEVEL = "price_level"
