"""Тесты DTO прогнозов тренда (api.schemas.predict, ml-spec §8.3).

Проверяют нормализацию тикера, границы ``prob_up`` и принятие enum-направления.
БД и Docker не требуются — чистая валидация Pydantic.
"""

from datetime import date

import pytest
from api.schemas.predict import (
    ShapContribution,
    TrendPredictionIn,
    TrendPredictionOut,
)
from pydantic import ValidationError
from stocklens_core.enums import TrendDirection


def _valid_trend_out(prob_up: float) -> TrendPredictionOut:
    """Полностью заполненный ответ тренда — варьируется только ``prob_up`` для проверки границ."""
    return TrendPredictionOut(
        ticker="SBER",
        predicted_for=date(2024, 6, 20),
        horizon_days=5,
        prob_up=prob_up,
        direction=TrendDirection.UP,
        shap=[ShapContribution(feature="rsi_14", value=0.12)],
        base_value=0.5,
        model_version="trend@champion",
    )


def test_trend_prediction_in_upcases_ticker() -> None:
    """Тикер приводится к верхнему регистру и обрезается, как у VolatilityPredictionIn."""
    assert TrendPredictionIn(ticker="  sber ").ticker == "SBER"


def test_trend_prediction_in_rejects_empty_ticker() -> None:
    """Пустой тикер нарушает min_length=1 и отклоняется."""
    with pytest.raises(ValidationError):
        TrendPredictionIn(ticker="")


def test_trend_prediction_out_accepts_direction_enum() -> None:
    """direction принимает член TrendDirection и сохраняет его значение."""
    out = _valid_trend_out(prob_up=0.7)
    assert out.direction is TrendDirection.UP
    assert out.direction.value == "up"


def test_trend_prediction_out_accepts_prob_up_bounds() -> None:
    """Граничные значения 0.0 и 1.0 допустимы (ge=0, le=1)."""
    assert _valid_trend_out(prob_up=0.0).prob_up == 0.0
    assert _valid_trend_out(prob_up=1.0).prob_up == 1.0


def test_trend_prediction_out_rejects_prob_up_above_one() -> None:
    """prob_up вне [0,1] сверху отклоняется как ValidationError."""
    with pytest.raises(ValidationError):
        _valid_trend_out(prob_up=1.5)


def test_trend_prediction_out_rejects_prob_up_below_zero() -> None:
    """prob_up вне [0,1] снизу отклоняется как ValidationError."""
    with pytest.raises(ValidationError):
        _valid_trend_out(prob_up=-0.1)


def test_shap_contribution_is_typed_pair() -> None:
    """ShapContribution — типизированная пара feature/value, не нетипизированный dict."""
    contribution = ShapContribution(feature="macd", value=-0.05)
    assert contribution.feature == "macd"
    assert contribution.value == -0.05
