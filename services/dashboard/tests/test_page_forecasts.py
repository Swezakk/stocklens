"""Unit-тесты чистых хелперов страницы «Прогнозы» (без поднятия Streamlit, DESIGN §10.4).

UI-оркестрация (`render`) не тестируется (тонкая); покрываются типизированные хелперы:
сортировка тикеров, клэмп окна под границы эндпоинта, форматирование метрики,
построение блоков live-метрик и форварда.
"""

from datetime import date

from dashboard.api_client.dto import SecurityOut, VolatilityMetricsOut, VolatilityRegimeOut
from dashboard.pages.forecasts import (
    _build_forward_block,
    _build_live_block,
    _clamp_lookback,
    _format_metric,
    _ForwardBlock,
    _LiveBlock,
    _ticker_options,
)


def _security(ticker: str) -> SecurityOut:
    return SecurityOut(id=1, ticker=ticker, name=ticker, board="TQBR", aliases=[], is_active=True)


def _metrics(
    qlike: float = 0.512,
    qlike_baseline: float = 0.731,
    rmse: float = 0.042,
) -> VolatilityMetricsOut:
    return VolatilityMetricsOut(qlike=qlike, qlike_baseline=qlike_baseline, rmse=rmse)


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


def test_build_live_block_with_metrics_shows_model_and_baseline() -> None:
    """Блок с live_metrics: модель/baseline отформатированы, аннотация содержит N."""
    block = _build_live_block(
        live_metrics=_metrics(qlike=0.521, qlike_baseline=0.743),
        live_sample_size=42,
    )
    assert isinstance(block, _LiveBlock)
    assert block.model_qlike == "0.521"
    assert block.baseline_qlike == "0.743"
    assert "42" in block.annotation
    assert block.is_accumulating is False


def test_build_live_block_none_metrics_with_positive_n_shows_accumulating() -> None:
    """Нет live_metrics, но есть созревшие пары: режим «накапливается»."""
    block = _build_live_block(live_metrics=None, live_sample_size=7)
    assert isinstance(block, _LiveBlock)
    assert block.model_qlike is None
    assert block.baseline_qlike is None
    assert "7" in block.annotation
    assert block.is_accumulating is True


def test_build_live_block_none_metrics_zero_n_shows_accumulating_with_zero() -> None:
    """Нет live_metrics и 0 созревших пар: режим «накапливается», N=0."""
    block = _build_live_block(live_metrics=None, live_sample_size=0)
    assert block.is_accumulating is True
    assert "0" in block.annotation
    assert block.model_qlike is None


def _regime(
    *,
    volatility: float = 0.041,
    is_elevated: bool = False,
    quantile: float = 0.90,
    lookback: int = 252,
    predicted_for: date = date(2026, 6, 20),
) -> VolatilityRegimeOut:
    """Вспомогательная фабрика VolatilityRegimeOut с разумными дефолтами."""
    return VolatilityRegimeOut(
        ticker="SBER",
        predicted_for=predicted_for,
        volatility=volatility,
        threshold=0.035,
        is_elevated=is_elevated,
        quantile=quantile,
        lookback=lookback,
    )


def test_build_forward_block_present_returns_formatted_vol_pct() -> None:
    """forward присутствует: vol_pct содержит значение в процентах с одним знаком."""
    block = _build_forward_block(_regime(volatility=0.041))
    assert isinstance(block, _ForwardBlock)
    assert block.vol_pct == "4.1%"


def test_build_forward_block_present_returns_formatted_date() -> None:
    """forward присутствует: date_label содержит дату в формате DD.MM.YYYY."""
    block = _build_forward_block(_regime(predicted_for=date(2026, 6, 20)))
    assert isinstance(block, _ForwardBlock)
    assert block.date_label == "20.06.2026"


def test_build_forward_block_normal_regime_label() -> None:
    """is_elevated=False → режим «Нормальный»."""
    block = _build_forward_block(_regime(is_elevated=False))
    assert block.regime_label == "Нормальный"
    assert block.is_elevated is False


def test_build_forward_block_elevated_regime_label() -> None:
    """is_elevated=True → режим «Повышенный»."""
    block = _build_forward_block(_regime(is_elevated=True))
    assert block.regime_label == "Повышенный"
    assert block.is_elevated is True


def test_build_forward_block_caption_contains_quantile_and_lookback() -> None:
    """Подпись содержит перцентиль и число торговых дней lookback."""
    block = _build_forward_block(_regime(quantile=0.90, lookback=252))
    # Перцентиль отображается как целое: 0.90 → «90»
    assert "90" in block.caption
    assert "252" in block.caption


def test_build_forward_block_none_returns_unavailable() -> None:
    """forward=None: is_available=False, приложение не падает."""
    block = _build_forward_block(None)
    assert isinstance(block, _ForwardBlock)
    assert block.is_available is False
    # Прочие поля не должны вызываться — проверяем только is_available.
