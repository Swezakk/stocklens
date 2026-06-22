"""Тесты чистых хелперов рыночной сводки сайдбара (DESIGN.md §4).

UI-раскладка не тестируется (DESIGN §10) — покрываются только pure-хелперы выбора и
форматирования значений сводки: последние два close индекса, свежий close, свежая ставка,
форматирование значения индекса.
"""

from datetime import date
from decimal import Decimal

from dashboard.api_client.dto import IndexValueOut, KeyRateOut
from dashboard.components.sidebar import (
    _format_index,
    _latest_close,
    _latest_rate,
    _latest_two_closes,
)


def _index_value(trade_date: date, close: str) -> IndexValueOut:
    """Собрать IndexValueOut с заданными торговой датой и close (Decimal-строка)."""
    return IndexValueOut(trade_date=trade_date, close=Decimal(close))


def _key_rate(rate_date: date, rate: str) -> KeyRateOut:
    """Собрать KeyRateOut с заданными датой и ставкой."""
    return KeyRateOut(rate_date=rate_date, rate=Decimal(rate))


def test_latest_two_closes_returns_last_then_previous_by_date() -> None:
    values = [
        _index_value(date(2026, 6, 18), "3180.00"),
        _index_value(date(2026, 6, 20), "3210.55"),
        _index_value(date(2026, 6, 19), "3200.10"),
    ]
    assert _latest_two_closes(values) == (3210.55, 3200.10)


def test_latest_two_closes_single_point_returns_none() -> None:
    assert _latest_two_closes([_index_value(date(2026, 6, 20), "3210.55")]) is None


def test_latest_two_closes_empty_returns_none() -> None:
    assert _latest_two_closes([]) is None


def test_latest_close_picks_most_recent_date() -> None:
    values = [
        _index_value(date(2026, 6, 18), "3180.00"),
        _index_value(date(2026, 6, 20), "3210.55"),
    ]
    assert _latest_close(values) == 3210.55


def test_latest_close_empty_returns_none() -> None:
    assert _latest_close([]) is None


def test_latest_rate_picks_most_recent_date() -> None:
    rates = [
        _key_rate(date(2026, 6, 1), "16.00"),
        _key_rate(date(2026, 6, 15), "14.25"),
    ]
    assert _latest_rate(rates) == 14.25


def test_latest_rate_empty_returns_none() -> None:
    assert _latest_rate([]) is None


def test_format_index_uses_space_thousands_separator() -> None:
    assert _format_index(2385.53) == "2 385.53"
