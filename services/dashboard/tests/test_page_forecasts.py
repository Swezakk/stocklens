"""Unit-тесты чистых хелперов страницы «Прогнозы» (без поднятия Streamlit, DESIGN §10.4).

UI-оркестрация (`render`) не тестируется (тонкая); покрываются типизированные хелперы:
сортировка тикеров, клэмп окна под границы эндпоинта, форматирование метрики.
"""

from dashboard.api_client.dto import SecurityOut
from dashboard.pages.forecasts import _clamp_lookback, _format_metric, _ticker_options


def _security(ticker: str) -> SecurityOut:
    return SecurityOut(id=1, ticker=ticker, name=ticker, board="TQBR", aliases=[], is_active=True)


def test_ticker_options_sorts_alphabetically() -> None:
    options = _ticker_options([_security("LKOH"), _security("GAZP"), _security("SBER")])
    assert options == ["GAZP", "LKOH", "SBER"]


def test_clamp_lookback_keeps_value_inside_window() -> None:
    assert _clamp_lookback(90) == 90


def test_clamp_lookback_caps_above_max_to_365() -> None:
    assert _clamp_lookback(1000) == 365


def test_clamp_lookback_raises_below_min_to_5() -> None:
    assert _clamp_lookback(1) == 5


def test_format_metric_keeps_three_decimals() -> None:
    assert _format_metric(0.8442) == "0.844"
