"""Тесты чистых хелперов страницы «Портфель» (DESIGN.md §5, §10.5).

UI-layout страницы не тестируется (DESIGN §10) — render() тонкий. Покрываются чистые
типизированные хелперы форматирования и разложения данных: денежная сумма и P&L со
знаком (типографский минус, заглушка отсутствующей оценки), знак P&L, строка таблицы
позиций, DataFrame со стабильным порядком колонок, разложение equity-кривой и опции
тикеров для удаления.
"""

from datetime import UTC, date, datetime
from decimal import Decimal

import pandas as pd
from dashboard.api_client.dto import EquityPointOut, PositionOut
from dashboard.api_client.errors import ApiServerError, ApiUnavailableError, AuthError
from dashboard.components.transforms import DeltaDirection
from dashboard.pages.portfolio import (
    _COL_AVG_PRICE,
    _COL_CURRENT_PRICE,
    _COL_CURRENT_VALUE,
    _COL_PNL,
    _COL_QUANTITY,
    _COL_TICKER,
    _POSITION_COLUMNS,
    _equity_series,
    _format_money,
    _format_pnl,
    _is_empty_state_error,
    _pnl_direction,
    _position_row,
    _positions_dataframe,
    _ticker_options,
    _vs_imoex_delta,
)

_MINUS_SIGN = "−"


def _position(
    ticker: str = "SBER",
    *,
    quantity: int = 10,
    avg_price: str = "280.00",
    current_price: str | None = "310.75",
    current_value: str | None = "3107.50",
    unrealized_pnl: str | None = "307.50",
) -> PositionOut:
    """Собрать PositionOut с дефолтной рыночной оценкой (поля можно переопределить)."""
    return PositionOut(
        ticker=ticker,
        quantity=quantity,
        avg_price=Decimal(avg_price),
        opened_at=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
        current_price=None if current_price is None else Decimal(current_price),
        current_value=None if current_value is None else Decimal(current_value),
        unrealized_pnl=None if unrealized_pnl is None else Decimal(unrealized_pnl),
    )


def test_format_money_positive_keeps_two_decimals() -> None:
    assert _format_money(Decimal("3107.5")) == "3107.50"


def test_format_money_negative_uses_typographic_minus() -> None:
    assert _format_money(Decimal("-150.25")) == f"{_MINUS_SIGN}150.25"


def test_format_money_none_renders_placeholder() -> None:
    assert _format_money(None) == "—"


def test_format_money_quantizes_long_decimal() -> None:
    assert _format_money(Decimal("310.754")) == "310.75"


def test_pnl_direction_classifies_sign() -> None:
    assert _pnl_direction(Decimal("1")) is DeltaDirection.UP
    assert _pnl_direction(Decimal("-1")) is DeltaDirection.DOWN
    assert _pnl_direction(Decimal("0")) is DeltaDirection.FLAT


def test_format_pnl_positive_prepends_plus_sign() -> None:
    assert _format_pnl(Decimal("307.50")) == "+307.50"


def test_format_pnl_negative_uses_typographic_minus() -> None:
    assert _format_pnl(Decimal("-307.50")) == f"{_MINUS_SIGN}307.50"


def test_format_pnl_zero_has_no_sign() -> None:
    assert _format_pnl(Decimal("0")) == "0.00"


def test_format_pnl_none_renders_placeholder() -> None:
    assert _format_pnl(None) == "—"


def test_position_row_formats_all_columns() -> None:
    row = _position_row(_position())
    assert row[_COL_TICKER] == "SBER"
    assert row[_COL_QUANTITY] == 10
    assert row[_COL_AVG_PRICE] == "280.00"
    assert row[_COL_CURRENT_PRICE] == "310.75"
    assert row[_COL_CURRENT_VALUE] == "3107.50"
    assert row[_COL_PNL] == "+307.50"


def test_position_row_missing_valuation_uses_placeholder() -> None:
    row = _position_row(_position(current_price=None, current_value=None, unrealized_pnl=None))
    assert row[_COL_CURRENT_PRICE] == "—"
    assert row[_COL_CURRENT_VALUE] == "—"
    assert row[_COL_PNL] == "—"


def test_positions_dataframe_keeps_column_order() -> None:
    frame = _positions_dataframe([_position("SBER"), _position("GAZP")])
    assert list(frame.columns) == list(_POSITION_COLUMNS)
    assert len(frame) == 2


def test_positions_dataframe_empty_keeps_columns() -> None:
    frame = _positions_dataframe([])
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == list(_POSITION_COLUMNS)
    assert frame.empty


def test_equity_series_splits_parallel_columns() -> None:
    curve = [
        EquityPointOut(date=date(2026, 1, 1), portfolio=100.0, imoex=100.0),
        EquityPointOut(date=date(2026, 2, 1), portfolio=110.0, imoex=104.0),
    ]
    dates, portfolio, imoex = _equity_series(curve)
    assert dates == [date(2026, 1, 1), date(2026, 2, 1)]
    assert portfolio == [100.0, 110.0]
    assert imoex == [100.0, 104.0]


def test_equity_series_empty_curve_returns_empty_lists() -> None:
    dates, portfolio, imoex = _equity_series([])
    assert dates == []
    assert portfolio == []
    assert imoex == []


def test_ticker_options_sorted_and_deduplicated_by_position() -> None:
    options = _ticker_options([_position("GAZP"), _position("SBER"), _position("LKOH")])
    assert options == ["GAZP", "LKOH", "SBER"]


def test_ticker_options_empty_positions() -> None:
    assert _ticker_options([]) == []


def test_vs_imoex_delta_portfolio_ahead_uses_percentage_points() -> None:
    direction, text = _vs_imoex_delta(portfolio_return_pct=10.98, imoex_return_pct=4.20)
    assert direction is DeltaDirection.UP
    assert text == "▲ +6.78 п.п."


def test_vs_imoex_delta_portfolio_behind_uses_typographic_minus() -> None:
    direction, text = _vs_imoex_delta(portfolio_return_pct=2.0, imoex_return_pct=5.5)
    assert direction is DeltaDirection.DOWN
    assert text == f"▼ {_MINUS_SIGN}3.50 п.п."


def test_vs_imoex_delta_equal_returns_is_flat() -> None:
    direction, text = _vs_imoex_delta(portfolio_return_pct=4.2, imoex_return_pct=4.2)
    assert direction is DeltaDirection.FLAT
    assert text == "→ 0.00 п.п."


def test_is_empty_state_error_true_for_422_server_error() -> None:
    assert _is_empty_state_error(ApiServerError(status=422, detail="портфель пуст")) is True


def test_is_empty_state_error_false_for_other_server_status() -> None:
    assert _is_empty_state_error(ApiServerError(status=500, detail="сбой")) is False


def test_is_empty_state_error_false_for_unavailable_and_auth() -> None:
    assert _is_empty_state_error(ApiUnavailableError()) is False
    assert _is_empty_state_error(AuthError()) is False
