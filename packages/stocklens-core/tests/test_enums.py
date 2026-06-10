"""Tests for stocklens_core.enums — values, str-subclass invariants."""

from stocklens_core.enums import (
    AlertKind,
    CollectorRunStatus,
    Currency,
    PredictionKind,
    SentimentLabel,
)


def test_collector_run_status_values() -> None:
    """Каждый член должен возвращать задокументированное строковое значение через .value."""
    assert CollectorRunStatus.SUCCESS.value == "success"
    assert CollectorRunStatus.PARTIAL.value == "partial"
    assert CollectorRunStatus.FAILED.value == "failed"


def test_collector_run_status_is_str() -> None:
    """CollectorRunStatus должен быть подтипом str для совместимости с ORM и JSON."""
    assert isinstance(CollectorRunStatus.SUCCESS, str)


def test_sentiment_label_values() -> None:
    assert SentimentLabel.POSITIVE.value == "positive"
    assert SentimentLabel.NEUTRAL.value == "neutral"
    assert SentimentLabel.NEGATIVE.value == "negative"


def test_sentiment_label_is_str() -> None:
    assert isinstance(SentimentLabel.POSITIVE, str)


def test_prediction_kind_values() -> None:
    assert PredictionKind.VOLATILITY.value == "volatility"
    assert PredictionKind.TREND.value == "trend"


def test_prediction_kind_is_str() -> None:
    assert isinstance(PredictionKind.VOLATILITY, str)


def test_currency_values() -> None:
    assert Currency.RUB.value == "RUB"
    assert Currency.USD.value == "USD"
    assert Currency.EUR.value == "EUR"
    assert Currency.CNY.value == "CNY"


def test_currency_is_str() -> None:
    assert isinstance(Currency.RUB, str)


def test_alert_kind_values() -> None:
    assert AlertKind.SENTIMENT_SPIKE.value == "sentiment_spike"
    assert AlertKind.VOLATILITY_REGIME.value == "volatility_regime"
    assert AlertKind.DIVIDEND_UPCOMING.value == "dividend_upcoming"
    assert AlertKind.PRICE_LEVEL.value == "price_level"


def test_alert_kind_is_str() -> None:
    assert isinstance(AlertKind.SENTIMENT_SPIKE, str)


def test_all_enums_have_no_extra_members() -> None:
    """Guard against accidental additions — member counts must match the spec."""
    assert len(CollectorRunStatus) == 3
    assert len(SentimentLabel) == 3
    assert len(PredictionKind) == 2
    assert len(Currency) == 4
    assert len(AlertKind) == 4
