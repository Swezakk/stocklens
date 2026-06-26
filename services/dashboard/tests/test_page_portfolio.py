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
import pytest
import streamlit as st
from dashboard.api_client.client import _FALLBACK_DETAIL, ApiClient
from dashboard.api_client.dto import (
    EquityPointOut,
    FrontierPoint,
    OptimizationStrategy,
    OptimizeResult,
    PositionOut,
)
from dashboard.api_client.errors import ApiServerError, ApiUnavailableError, AuthError
from dashboard.components import charts, feedback
from dashboard.components.transforms import DeltaDirection
from dashboard.pages import portfolio
from dashboard.pages.portfolio import (
    _COL_AVG_PRICE,
    _COL_CURRENT_PRICE,
    _COL_CURRENT_VALUE,
    _COL_PNL,
    _COL_QUANTITY,
    _COL_TICKER,
    _EMPTY_EQUITY,
    _EMPTY_FRONTIER,
    _EMPTY_FRONTIER_DEGENERATE,
    _EMPTY_POSITIONS,
    _POSITION_COLUMNS,
    _equity_series,
    _format_money,
    _format_pnl,
    _is_empty_state_error,
    _pnl_color,
    _pnl_direction,
    _position_row,
    _positions_dataframe,
    _render_frontier_section,
    _render_load_failure,
    _style_positions,
    _ticker_options,
    _vs_imoex_delta,
)

from dashboard import theme

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


def test_pnl_color_positive_is_up_color() -> None:
    assert _pnl_color("+307.50") == f"color: {theme.UP}"


def test_pnl_color_negative_typographic_minus_is_down_color() -> None:
    """Падение читается с типографского минуса U+2212, а не ASCII «-» (формат _format_pnl)."""
    assert _pnl_color(f"{_MINUS_SIGN}150.25") == f"color: {theme.DOWN}"


def test_pnl_color_ascii_minus_is_not_colored() -> None:
    """ASCII «-» не цвет падения: _format_pnl никогда его не выдаёт, страхуемся от ложного down."""
    assert _pnl_color("-150.25") == ""


def test_pnl_color_zero_and_placeholder_have_no_rule() -> None:
    assert _pnl_color("0.00") == ""
    assert _pnl_color("—") == ""


def test_style_positions_colors_only_pnl_column() -> None:
    """Styler красит P&L по знаку и не трогает прочие колонки (рендерится st.dataframe)."""
    frame = _positions_dataframe(
        [_position("SBER", unrealized_pnl="307.50"), _position("GAZP", unrealized_pnl="-50.00")]
    )

    rendered = _style_positions(frame).to_html()

    assert f"color: {theme.UP}" in rendered
    assert f"color: {theme.DOWN}" in rendered


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


def _fake_client() -> ApiClient:
    """Реальный ApiClient (конструктор лишь создаёт httpx.Client, сети нет).

    Нужен для типизированного аргумента render-функций: ``_load_optimize`` в тестах
    подменяется monkeypatch-ем, поэтому сетевые вызовы клиента не выполняются.
    """
    return ApiClient(
        base_url="http://api:8000",
        api_prefix="/api/v1",
        timeout=1.0,
        token_provider=lambda: "token",
        on_unauthorized=lambda: None,
    )


def _optimize_result(
    *,
    fallback_reason: str | None = None,
    strategy: OptimizationStrategy = OptimizationStrategy.MAX_SHARPE,
    requested_strategy: OptimizationStrategy = OptimizationStrategy.MAX_SHARPE,
    frontier: list[FrontierPoint] | None = None,
) -> OptimizeResult:
    """Собрать OptimizeResult со всеми обязательными полями (фронтир/фолбэк настраиваются)."""
    return OptimizeResult(
        strategy=strategy,
        requested_strategy=requested_strategy,
        weights={"SBER": 0.6, "GAZP": 0.4},
        expected_return=0.12,
        volatility=0.2,
        sharpe=0.6,
        frontier=[FrontierPoint(volatility=0.2, expected_return=0.12)]
        if frontier is None
        else frontier,
        equal_weight_sharpe=0.5,
        imoex_sharpe=0.4,
        fallback_reason=fallback_reason,
    )


@pytest.fixture
def _silence_subheader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Заглушить st.subheader внутри секции фронтира (UI-побочка, не предмет теста)."""
    monkeypatch.setattr(st, "subheader", lambda *args, **kwargs: None)


def test_render_frontier_section_shows_fallback_banner_with_non_empty_frontier(
    monkeypatch: pytest.MonkeyPatch,
    _silence_subheader: None,
) -> None:
    """Авто-фолбэк max-Sharpe → min-vol: баннер причины + график строятся вместе."""
    reason = "Максимизация Шарпа невозможна: применена минимизация риска."
    result = _optimize_result(
        fallback_reason=reason,
        strategy=OptimizationStrategy.MIN_VOLATILITY,
        frontier=[
            FrontierPoint(volatility=0.18, expected_return=0.10),
            FrontierPoint(volatility=0.22, expected_return=0.14),
        ],
    )
    infos: list[str] = []
    empties: list[str] = []
    charts_built: list[object] = []
    monkeypatch.setattr(portfolio, "_load_optimize", lambda _client: result)
    monkeypatch.setattr(feedback, "render_info", infos.append)
    monkeypatch.setattr(feedback, "render_empty", empties.append)
    monkeypatch.setattr(charts, "build_efficient_frontier_chart", lambda **kwargs: kwargs)
    monkeypatch.setattr(charts, "render_chart", charts_built.append)

    _render_frontier_section(_fake_client())

    assert infos == [reason]
    assert empties == []
    assert len(charts_built) == 1


def test_render_frontier_section_shows_fallback_banner_with_empty_frontier(
    monkeypatch: pytest.MonkeyPatch,
    _silence_subheader: None,
) -> None:
    """Вырожденный фолбэк (доходности неразличимы): баннер причины + точное пустое сообщение.

    Регресс c695d3fb: пустой фронтир-200 НЕ должен показывать «нужно ≥ 2 бумаги».
    """
    reason = "Граница вырождена: применена минимизация риска."
    result = _optimize_result(
        fallback_reason=reason,
        strategy=OptimizationStrategy.MIN_VOLATILITY,
        frontier=[],
    )
    infos: list[str] = []
    empties: list[str] = []
    monkeypatch.setattr(portfolio, "_load_optimize", lambda _client: result)
    monkeypatch.setattr(feedback, "render_info", infos.append)
    monkeypatch.setattr(feedback, "render_empty", empties.append)

    _render_frontier_section(_fake_client())

    assert infos == [reason]
    assert empties == [_EMPTY_FRONTIER_DEGENERATE]
    assert _EMPTY_FRONTIER_DEGENERATE != _EMPTY_FRONTIER
    assert "≥ 2 бумаги" not in _EMPTY_FRONTIER_DEGENERATE


def test_render_frontier_section_no_banner_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    _silence_subheader: None,
) -> None:
    """Без авто-фолбэка (fallback_reason=None) баннер render_info не показывается."""
    result = _optimize_result(fallback_reason=None)
    infos: list[str] = []
    monkeypatch.setattr(portfolio, "_load_optimize", lambda _client: result)
    monkeypatch.setattr(feedback, "render_info", infos.append)
    monkeypatch.setattr(charts, "build_efficient_frontier_chart", lambda **kwargs: kwargs)
    monkeypatch.setattr(charts, "render_chart", lambda *a, **k: None)

    _render_frontier_section(_fake_client())

    assert infos == []


def test_render_load_failure_prefers_server_detail_on_empty_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """422 с реальным server-detail показывает причину API, а не хардкод секции."""
    detail = "Недостаточно истории котировок для построения границы за выбранный период."
    error = ApiServerError(status=422, detail=detail)
    empties: list[str] = []
    monkeypatch.setattr(feedback, "render_empty", empties.append)

    _render_load_failure(error, _EMPTY_FRONTIER)

    assert empties == [detail]


def test_render_load_failure_falls_back_to_section_message_on_generic_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """422 с дженерик-фолбэком клиента (тело без detail) → хардкод секции."""
    error = ApiServerError(status=422, detail=_FALLBACK_DETAIL)
    empties: list[str] = []
    monkeypatch.setattr(feedback, "render_empty", empties.append)

    _render_load_failure(error, _EMPTY_FRONTIER)

    assert empties == [_EMPTY_FRONTIER]


def test_render_load_failure_unavailable_error_uses_error_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ApiUnavailableError (не 422) → ветка ошибки сервиса, не пустое состояние."""
    empties: list[str] = []
    errors: list[str] = []
    monkeypatch.setattr(feedback, "render_empty", empties.append)
    monkeypatch.setattr(feedback, "render_error", errors.append)

    _render_load_failure(ApiUnavailableError(), _EMPTY_POSITIONS)

    assert empties == []
    assert len(errors) == 1


def test_render_load_failure_service_error_branch_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Не-422 (5xx) по-прежнему идёт в render_error, не в пустое состояние."""
    error = ApiServerError(status=500, detail="сбой")
    empties: list[str] = []
    errors: list[str] = []
    monkeypatch.setattr(feedback, "render_empty", empties.append)
    monkeypatch.setattr(feedback, "render_error", errors.append)

    _render_load_failure(error, _EMPTY_EQUITY)

    assert empties == []
    assert errors == [error.user_message]
