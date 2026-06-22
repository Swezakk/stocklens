"""Тесты чистых presentation-трансформов дашборда (DESIGN.md §5, §9).

Покрывают format_delta (рост/падение/без изменения, ноль-делитель, отрицательные),
group_sentiment_by_date (пусто, один день, порядок дней, среднее, None-тональность,
переход московской полуночи) и word_frequencies (стоп-слова, пунктуация, top_n, пусто).
"""

from datetime import UTC, date, datetime

import pytest
from dashboard.api_client.dto import NewsOut, SentimentOut
from dashboard.components.transforms import (
    DeltaDirection,
    SentimentDayPoint,
    format_delta,
    group_sentiment_by_date,
    word_frequencies,
)
from stocklens_core.enums import SentimentLabel

_MINUS_SIGN = "−"


def _news(published_at: datetime, label: SentimentLabel | None, score: float) -> NewsOut:
    """Собрать NewsOut с заданным временем публикации и (опциональной) тональностью."""
    sentiment = (
        None
        if label is None
        else SentimentOut(label=label, score=score, model_version="rubert-tiny2-v1")
    )
    return NewsOut(
        id=1,
        source="Интерфакс",
        url="https://example.com/news",
        title="Заголовок",
        summary=None,
        published_at=published_at,
        sentiment=sentiment,
        tickers=["SBER"],
    )


def test_format_delta_up_uses_plus_sign_and_up_glyph() -> None:
    text, direction = format_delta(current=110.0, previous=100.0)
    assert direction is DeltaDirection.UP
    assert text == "▲ +10.00%"


def test_format_delta_down_uses_typographic_minus_and_down_glyph() -> None:
    text, direction = format_delta(current=90.0, previous=100.0)
    assert direction is DeltaDirection.DOWN
    assert text == f"▼ {_MINUS_SIGN}10.00%"


def test_format_delta_equal_values_are_flat() -> None:
    text, direction = format_delta(current=100.0, previous=100.0)
    assert direction is DeltaDirection.FLAT
    assert text == "→ 0.00%"


def test_format_delta_zero_previous_is_flat_without_division_error() -> None:
    text, direction = format_delta(current=42.0, previous=0.0)
    assert direction is DeltaDirection.FLAT
    assert text == "→ 0.00%"


def test_format_delta_negative_base_recovering_is_up() -> None:
    # previous < 0, current > previous → рост; знаменатель |previous|, знак из направления.
    text, direction = format_delta(current=-50.0, previous=-100.0)
    assert direction is DeltaDirection.UP
    assert text == "▲ +50.00%"


def test_format_delta_negative_base_falling_is_down() -> None:
    text, direction = format_delta(current=-150.0, previous=-100.0)
    assert direction is DeltaDirection.DOWN
    assert text == f"▼ {_MINUS_SIGN}50.00%"


def test_group_sentiment_by_date_empty_returns_empty() -> None:
    assert group_sentiment_by_date([]) == []


def test_group_sentiment_by_date_single_day_means_scores() -> None:
    moment = datetime(2026, 6, 20, 9, 0, tzinfo=UTC)
    later = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    series = group_sentiment_by_date(
        [
            _news(moment, SentimentLabel.POSITIVE, 0.80),
            _news(later, SentimentLabel.NEGATIVE, 0.40),
        ]
    )
    assert len(series) == 1
    point = series[0]
    assert point.day == date(2026, 6, 20)
    assert point.mean_score == pytest.approx(0.60)
    assert point.counts == {SentimentLabel.POSITIVE: 1, SentimentLabel.NEGATIVE: 1}


def test_group_sentiment_by_date_orders_days_ascending() -> None:
    earlier = datetime(2026, 6, 18, 9, 0, tzinfo=UTC)
    later = datetime(2026, 6, 20, 9, 0, tzinfo=UTC)
    series = group_sentiment_by_date(
        [
            _news(later, SentimentLabel.NEUTRAL, 0.5),
            _news(earlier, SentimentLabel.POSITIVE, 0.9),
        ]
    )
    assert [point.day for point in series] == [date(2026, 6, 18), date(2026, 6, 20)]


def test_group_sentiment_by_date_excludes_none_sentiment_from_mean_and_counts() -> None:
    moment = datetime(2026, 6, 20, 9, 0, tzinfo=UTC)
    series = group_sentiment_by_date(
        [
            _news(moment, SentimentLabel.POSITIVE, 1.0),
            _news(moment, None, 0.0),
        ]
    )
    assert series == [
        SentimentDayPoint(
            day=date(2026, 6, 20),
            mean_score=1.0,
            counts={SentimentLabel.POSITIVE: 1},
        )
    ]


def test_group_sentiment_by_date_all_none_returns_empty() -> None:
    moment = datetime(2026, 6, 20, 9, 0, tzinfo=UTC)
    assert group_sentiment_by_date([_news(moment, None, 0.0)]) == []


def test_group_sentiment_by_date_buckets_by_moscow_calendar_day() -> None:
    # 22:30 UTC = 01:30 МСК следующих суток: московский день должен быть 21-е, не 20-е.
    utc_late = datetime(2026, 6, 20, 22, 30, tzinfo=UTC)
    utc_early = datetime(2026, 6, 20, 5, 0, tzinfo=UTC)
    series = group_sentiment_by_date(
        [
            _news(utc_late, SentimentLabel.POSITIVE, 0.5),
            _news(utc_early, SentimentLabel.NEUTRAL, 0.5),
        ]
    )
    assert [point.day for point in series] == [date(2026, 6, 20), date(2026, 6, 21)]


def test_word_frequencies_removes_stopwords_and_lowercases() -> None:
    titles = ["Сбербанк и Газпром", "Сбербанк на бирже"]
    result = word_frequencies(titles, top_n=10)
    words = dict(result)
    assert words["сбербанк"] == 2
    assert "и" not in words
    assert "на" not in words


def test_word_frequencies_strips_punctuation_and_digits() -> None:
    result = word_frequencies(["Прибыль: +12%, рекорд!"], top_n=10)
    words = {word for word, _ in result}
    assert words == {"прибыль", "рекорд"}


def test_word_frequencies_respects_top_n_and_deterministic_tiebreak() -> None:
    titles = ["акция актив банк банк актив акция", "вклад"]
    # частоты: акция=2, актив=2, банк=2, вклад=1; при равенстве — по алфавиту.
    result = word_frequencies(titles, top_n=2)
    assert result == [("актив", 2), ("акция", 2)]


def test_word_frequencies_empty_titles_returns_empty() -> None:
    assert word_frequencies([], top_n=5) == []


def test_word_frequencies_non_positive_top_n_returns_empty() -> None:
    assert word_frequencies(["Сбербанк растёт"], top_n=0) == []
