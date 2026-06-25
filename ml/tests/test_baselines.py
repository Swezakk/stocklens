"""Unit-tests for the RW-RV volatility baseline (ml-spec §5.3)."""

import math

import numpy as np
import pandas as pd
import pytest
from stocklens_ml.models import baselines


def test_rw_rv_is_trailing_realized_variance() -> None:
    returns = pd.Series([np.nan, 0.1, 0.2, -0.1, 0.05])

    forecast = baselines.rw_rv_forecast(returns, horizon=2)

    assert math.isnan(forecast.iloc[1])  # < 2 наблюдений
    assert forecast.iloc[2] == pytest.approx(0.1**2 + 0.2**2)
    assert forecast.iloc[3] == pytest.approx(0.2**2 + (-0.1) ** 2)
    assert forecast.iloc[4] == pytest.approx((-0.1) ** 2 + 0.05**2)


def test_always_up_forecast_is_constant_one_aligned_to_index() -> None:
    index = pd.Index([10, 20, 30], name="row")

    forecast = baselines.always_up_forecast(index)

    assert forecast.name == "always_up_forecast"
    assert (forecast == 1.0).all()
    assert forecast.index.equals(index)
