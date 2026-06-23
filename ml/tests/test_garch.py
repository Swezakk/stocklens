"""Unit-tests for the GARCH(1,1) volatility forecast (ml-spec §5.1)."""

import numpy as np
import pandas as pd
import pytest
from stocklens_ml.models import garch


def test_forecast_variance_is_positive_finite_and_correct_scale() -> None:
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.0, 0.02, 300))  # ~2% дневная волатильность

    forecast = garch.forecast_variance(returns, horizon=5)

    assert np.isfinite(forecast)
    assert forecast > 0.0
    # 5-дневная дисперсия ≈ 5·(0.02)² = 0.002 (доли²). Диапазон ловит ошибку масштаба
    # ×100/÷100²: без деления на 100² результат был бы порядка 20, а не 0.002.
    assert 0.0002 < forecast < 0.02


def test_forecast_variance_returns_decimal_not_percent_scale() -> None:
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0, 0.01, 300))  # 1% дневная

    forecast = garch.forecast_variance(returns, horizon=5)

    # 5·(0.01)² = 0.0005 доли². В процентах² это было бы ~5 — проверяем, что вернули доли².
    assert forecast < 0.01


def test_forecast_variance_raises_on_insufficient_data() -> None:
    with pytest.raises(ValueError, match="GARCH"):
        garch.forecast_variance(pd.Series([0.01] * 10))
