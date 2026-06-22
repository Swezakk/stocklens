"""Тесты чистых хелперов страницы «Акции» (DESIGN.md §10.2).

UI-раскладка не тестируется (DESIGN §10); тестируются чистые преобразования данных,
которые страница вынесла из `render()`: границы периода, опции тикеров, сортировка
свечей по возрастанию, дельта последнего close (вход DESC) и фильтр дивидендных
отсечек по окну графика. Свечи/дивиденды собираются напрямую (как в test_charts),
без поднятия клиента или Streamlit-runtime.
"""

from datetime import date
from decimal import Decimal

from dashboard.api_client.dto import CandleOut, DividendOut, SecurityOut
from dashboard.components.charts import build_comparison_chart
from dashboard.pages.stocks import (
    _dividends_in_window,
    _latest_close_delta,
    _period_bounds,
    _sort_candles_ascending,
    _ticker_options,
)
from stocklens_core.enums import Currency


def _candle(trade_date: date, close: str) -> CandleOut:
    """Собрать CandleOut с заданными датой и close (прочие OHLCV — производные от close)."""
    close_dec = Decimal(close)
    return CandleOut(
        id=1,
        security_id=1,
        trade_date=trade_date,
        open=close_dec,
        high=close_dec,
        low=close_dec,
        close=close_dec,
        volume=1000,
        value=Decimal("1000000.00"),
        is_weekend_session=False,
    )


def _security(ticker: str) -> SecurityOut:
    """Собрать SecurityOut с заданным тикером (остальные поля — наполнители)."""
    return SecurityOut(
        id=1,
        ticker=ticker,
        name=ticker,
        board="TQBR",
        aliases=[ticker],
        is_active=True,
    )


def _securities(tickers: list[str]) -> list[SecurityOut]:
    """Собрать список SecurityOut из тикеров (вход хелпера _ticker_options)."""
    return [_security(ticker) for ticker in tickers]


def _dividend(ex_date: date) -> DividendOut:
    """Собрать DividendOut с заданной ex-датой (валюта/сумма — наполнители)."""
    return DividendOut(
        id=1,
        security_id=1,
        ex_date=ex_date,
        value=Decimal("12.50"),
        currency=Currency.RUB,
    )


def test_period_bounds_returns_iso_window_from_reference() -> None:
    date_from, date_to = _period_bounds(period_days=90, reference=date(2026, 6, 22))
    assert date_to == "2026-06-22"
    assert date_from == "2026-03-24"


def test_period_bounds_one_month_window() -> None:
    date_from, date_to = _period_bounds(period_days=30, reference=date(2026, 6, 22))
    assert date_from == "2026-05-23"
    assert date_to == "2026-06-22"


def test_ticker_options_sorted_alphabetically() -> None:
    securities = _securities(["GAZP", "SBER", "AFLT"])
    assert _ticker_options(securities) == ["AFLT", "GAZP", "SBER"]


def test_ticker_options_empty_list_returns_empty() -> None:
    assert _ticker_options([]) == []


def test_sort_candles_ascending_orders_by_trade_date() -> None:
    descending = [
        _candle(date(2026, 6, 20), "120.00"),
        _candle(date(2026, 6, 18), "100.00"),
        _candle(date(2026, 6, 19), "110.00"),
    ]
    ordered = _sort_candles_ascending(descending)
    assert [candle.trade_date for candle in ordered] == [
        date(2026, 6, 18),
        date(2026, 6, 19),
        date(2026, 6, 20),
    ]


def test_sort_candles_ascending_does_not_mutate_input() -> None:
    original = [
        _candle(date(2026, 6, 20), "120.00"),
        _candle(date(2026, 6, 18), "100.00"),
    ]
    _sort_candles_ascending(original)
    assert [candle.trade_date for candle in original] == [
        date(2026, 6, 20),
        date(2026, 6, 18),
    ]


def test_sort_candles_ascending_empty_returns_empty() -> None:
    assert _sort_candles_ascending([]) == []


def test_latest_close_delta_reads_latest_then_previous_from_desc_input() -> None:
    # Вход DESC (новейшая первая): последний close = 120, предыдущий = 110.
    desc = [_candle(date(2026, 6, 20), "120.00"), _candle(date(2026, 6, 19), "110.00")]
    pair = _latest_close_delta(desc)
    assert pair == (120.0, 110.0)


def test_latest_close_delta_single_candle_returns_none() -> None:
    assert _latest_close_delta([_candle(date(2026, 6, 20), "120.00")]) is None


def test_latest_close_delta_empty_returns_none() -> None:
    assert _latest_close_delta([]) is None


def test_dividends_in_window_keeps_only_in_range_ex_dates() -> None:
    dividends = [
        _dividend(date(2026, 3, 1)),  # до окна
        _dividend(date(2026, 5, 10)),  # внутри окна
        _dividend(date(2026, 7, 1)),  # после окна
    ]
    kept = _dividends_in_window(dividends, "2026-04-01", "2026-06-22")
    assert [dividend.ex_date for dividend in kept] == [date(2026, 5, 10)]


def test_dividends_in_window_includes_boundary_dates() -> None:
    dividends = [_dividend(date(2026, 4, 1)), _dividend(date(2026, 6, 22))]
    kept = _dividends_in_window(dividends, "2026-04-01", "2026-06-22")
    assert [dividend.ex_date for dividend in kept] == [
        date(2026, 4, 1),
        date(2026, 6, 22),
    ]


def test_dividends_in_window_empty_input_returns_empty() -> None:
    assert _dividends_in_window([], "2026-04-01", "2026-06-22") == []


def test_comparison_single_point_series_rebases_to_100() -> None:
    # Сравнение режима через публичный билдер (он владеет rebase-математикой):
    # одиночная точка серии нормируется к базе 100 (edge: первая = последняя точка).
    fig = build_comparison_chart({"SBER": [_candle(date(2026, 6, 18), "200.00")]})
    assert list(fig.data[0].y) == [100.0]


def test_comparison_empty_series_produces_no_trace() -> None:
    # Пустая серия не даёт трейса (нечего нормировать) — edge сравнения через билдер.
    fig = build_comparison_chart({"EMPTY": []})
    assert len(fig.data) == 0
