"""Тесты чистых хелперов страницы «Обзор» (DESIGN.md §10.1).

UI-раскладка не тестируется (DESIGN §10) — покрываются только pure-хелперы
преобразования данных: упорядочивание точек индекса, выбор последних двух close,
свежий курс валюты и ключевой ставки, направление/бейдж мувера, выбор времени
последнего успешного сбора и подпись футера.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from dashboard.api_client.dto import (
    CollectorRunOut,
    CurrencyRateOut,
    IndexValueOut,
    KeyRateOut,
    MoverOut,
)
from dashboard.components.transforms import DeltaDirection
from dashboard.pages.overview import (
    _INDEX_KPI_LIMIT,
    _INDEX_SPARKLINE_LIMIT,
    _format_update_footer,
    _index_points,
    _latest_currency_rate,
    _latest_key_rate,
    _latest_two_closes,
    _latest_update_time,
    _mover_badge,
    _mover_close_text,
    _mover_direction,
    _mover_row_markdown,
    _period_bounds,
)
from stocklens_core.enums import CollectorRunStatus, Currency

_MINUS_SIGN = "−"
_MOSCOW_TZ = ZoneInfo("Europe/Moscow")


def _index_value(trade_date: date, close: str) -> IndexValueOut:
    """Собрать IndexValueOut с заданными торговой датой и close (Decimal-строка)."""
    return IndexValueOut(trade_date=trade_date, close=Decimal(close))


def _currency_rate(rate_date: date, rate: str) -> CurrencyRateOut:
    """Собрать CurrencyRateOut USD/RUB с заданными датой и курсом."""
    return CurrencyRateOut(currency=Currency.USD, rate_date=rate_date, rate=Decimal(rate))


def _key_rate(rate_date: date, rate: str) -> KeyRateOut:
    """Собрать KeyRateOut с заданными датой и ставкой."""
    return KeyRateOut(rate_date=rate_date, rate=Decimal(rate))


def _mover(ticker: str, name: str, close: str, prev_close: str, change_pct: float) -> MoverOut:
    """Собрать MoverOut с заданными ценами закрытия и процентом изменения."""
    return MoverOut(
        ticker=ticker,
        name=name,
        close=Decimal(close),
        prev_close=Decimal(prev_close),
        change_pct=change_pct,
    )


def _run(
    finished_at: datetime | None,
    status: CollectorRunStatus,
    *,
    run_id: int = 1,
) -> CollectorRunOut:
    """Собрать CollectorRunOut с заданными временем завершения и статусом."""
    return CollectorRunOut(
        id=run_id,
        source="moex",
        started_at=datetime(2026, 6, 20, 8, 0, tzinfo=UTC),
        finished_at=finished_at,
        status=status,
        records_added=10,
        error_message=None,
    )


def test_index_points_orders_by_trade_date_ascending() -> None:
    values = [
        _index_value(date(2026, 6, 20), "3210.55"),
        _index_value(date(2026, 6, 18), "3180.00"),
        _index_value(date(2026, 6, 19), "3200.10"),
    ]
    assert _index_points(values) == [
        (date(2026, 6, 18), 3180.00),
        (date(2026, 6, 19), 3200.10),
        (date(2026, 6, 20), 3210.55),
    ]


def test_index_points_empty_returns_empty() -> None:
    assert _index_points([]) == []


def test_period_bounds_returns_iso_window_from_reference() -> None:
    date_from, date_to = _period_bounds(period_days=365, reference=date(2026, 6, 22))
    assert date_to == "2026-06-22"
    assert date_from == "2025-06-22"


def test_period_bounds_one_month_window() -> None:
    date_from, date_to = _period_bounds(period_days=30, reference=date(2026, 6, 22))
    assert date_from == "2026-05-23"
    assert date_to == "2026-06-22"


def test_latest_two_closes_returns_last_then_previous() -> None:
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


def test_latest_currency_rate_picks_most_recent_date() -> None:
    rates = [
        _currency_rate(date(2026, 6, 18), "78.50"),
        _currency_rate(date(2026, 6, 20), "79.10"),
        _currency_rate(date(2026, 6, 19), "78.90"),
    ]
    assert _latest_currency_rate(rates) == Decimal("79.10")


def test_latest_currency_rate_empty_returns_none() -> None:
    assert _latest_currency_rate([]) is None


def test_latest_key_rate_picks_most_recent_date() -> None:
    rates = [
        _key_rate(date(2026, 6, 1), "16.00"),
        _key_rate(date(2026, 6, 15), "15.50"),
    ]
    assert _latest_key_rate(rates) == Decimal("15.50")


def test_latest_key_rate_empty_returns_none() -> None:
    assert _latest_key_rate([]) is None


def test_mover_direction_positive_is_up() -> None:
    assert _mover_direction(2.5) is DeltaDirection.UP


def test_mover_direction_negative_is_down() -> None:
    assert _mover_direction(-1.2) is DeltaDirection.DOWN


def test_mover_direction_zero_is_flat() -> None:
    assert _mover_direction(0.0) is DeltaDirection.FLAT


def test_mover_badge_up_uses_plus_sign_glyph_and_class() -> None:
    badge = _mover_badge(2.50)
    assert "▲ +2.50%" in badge
    assert "delta-badge--up" in badge


def test_mover_badge_down_uses_typographic_minus_and_down_class() -> None:
    badge = _mover_badge(-1.20)
    assert f"▼ {_MINUS_SIGN}1.20%" in badge
    assert "delta-badge--down" in badge


def test_mover_badge_flat_uses_arrow_and_flat_class() -> None:
    badge = _mover_badge(0.0)
    assert "→ 0.00%" in badge
    assert "delta-badge--flat" in badge


def test_mover_close_text_formats_close_two_decimals() -> None:
    mover = _mover("SBER", "Сбербанк", "310.5", "305.00", 1.82)
    assert _mover_close_text(mover) == "310.50"


def test_mover_row_markdown_renders_ticker_name_close_and_badge_in_grid() -> None:
    """Плотный грид: тикер/имя/close в колонках + бейдж (фикс асимметрии space-between)."""
    mover = _mover("SBER", "Сбербанк", "310.55", "305.00", 1.82)

    markdown = _mover_row_markdown(mover)

    assert 'class="mover-row"' in markdown
    assert '<span class="mover-row__ticker">SBER</span>' in markdown
    assert '<span class="mover-row__name">Сбербанк</span>' in markdown
    assert '<span class="mover-row__close">310.55</span>' in markdown
    assert "▲ +1.82%" in markdown
    assert "delta-badge--up" in markdown


def test_mover_row_markdown_escapes_name_html() -> None:
    """Имя бумаги MOEX с «&»/«<» экранируется, как в kpi.py (защита от поломки разметки)."""
    mover = _mover("X", "A & B <co>", "10.00", "9.00", 1.0)

    markdown = _mover_row_markdown(mover)

    assert "&amp;" in markdown
    assert "&lt;co&gt;" in markdown
    assert "<co>" not in markdown


def test_index_sparkline_limit_exceeds_delta_minimum() -> None:
    """KPI IMOEX тянет полную серию под спарклайн, не 2 точки под дельту дня.

    Регресс: при выборке ровно ``_INDEX_KPI_LIMIT`` точек спарклайн вырождался в прямую
    диагональ из двух точек. Окно спарклайна должно быть заметно больше минимума дельты.
    """
    assert _INDEX_SPARKLINE_LIMIT > _INDEX_KPI_LIMIT
    assert _INDEX_SPARKLINE_LIMIT >= 20


def test_latest_update_time_picks_max_finished_success_in_moscow_tz() -> None:
    runs = [
        _run(datetime(2026, 6, 20, 6, 0, tzinfo=UTC), CollectorRunStatus.SUCCESS, run_id=1),
        _run(datetime(2026, 6, 20, 9, 30, tzinfo=UTC), CollectorRunStatus.SUCCESS, run_id=2),
    ]
    moment = _latest_update_time(runs)
    assert moment == datetime(2026, 6, 20, 9, 30, tzinfo=UTC).astimezone(_MOSCOW_TZ)
    assert moment is not None
    assert moment.tzinfo == _MOSCOW_TZ


def test_latest_update_time_ignores_non_success_runs() -> None:
    runs = [
        _run(datetime(2026, 6, 20, 12, 0, tzinfo=UTC), CollectorRunStatus.FAILED, run_id=1),
        _run(datetime(2026, 6, 20, 9, 0, tzinfo=UTC), CollectorRunStatus.SUCCESS, run_id=2),
    ]
    moment = _latest_update_time(runs)
    assert moment == datetime(2026, 6, 20, 9, 0, tzinfo=UTC).astimezone(_MOSCOW_TZ)


def test_latest_update_time_ignores_unfinished_success_runs() -> None:
    runs = [
        _run(None, CollectorRunStatus.SUCCESS, run_id=1),
        _run(datetime(2026, 6, 20, 7, 0, tzinfo=UTC), CollectorRunStatus.SUCCESS, run_id=2),
    ]
    moment = _latest_update_time(runs)
    assert moment == datetime(2026, 6, 20, 7, 0, tzinfo=UTC).astimezone(_MOSCOW_TZ)


def test_latest_update_time_no_success_returns_none() -> None:
    runs = [_run(datetime(2026, 6, 20, 12, 0, tzinfo=UTC), CollectorRunStatus.FAILED)]
    assert _latest_update_time(runs) is None


def test_latest_update_time_empty_returns_none() -> None:
    assert _latest_update_time([]) is None


def test_format_update_footer_known_time_uses_moscow_label() -> None:
    moment = datetime(2026, 6, 20, 9, 30, tzinfo=UTC).astimezone(_MOSCOW_TZ)
    footer = _format_update_footer(moment)
    # 09:30 UTC = 12:30 МСК (летом смещение +3).
    assert footer == "Данные обновлены: 20.06.2026 12:30 МСК"


def test_format_update_footer_none_is_explicit_unavailable_copy() -> None:
    footer = _format_update_footer(None)
    assert "недоступно" in footer
