"""Unit-tests for volatility features and target (ml-spec §4.1–4.3), including no-leakage."""

import math

import numpy as np
import pandas as pd
from stocklens_ml.features import volatility


def test_parkinson_variance_matches_formula() -> None:
    candles = pd.DataFrame({"high": [110.0, 105.0], "low": [100.0, 100.0]})

    result = volatility.parkinson_variance(candles)

    coef = 1.0 / (4.0 * math.log(2.0))
    assert result.iloc[0] == coef * math.log(110.0 / 100.0) ** 2
    assert result.iloc[1] == coef * math.log(105.0 / 100.0) ** 2


def test_parkinson_variance_is_split_invariant() -> None:
    # ln(H/L) не зависит от общего множителя — сплит-коррекция не меняет Паркинсон.
    raw = pd.DataFrame({"high": [110.0], "low": [100.0]})
    adjusted = pd.DataFrame({"high": [11.0], "low": [10.0]})

    assert (
        volatility.parkinson_variance(raw).iloc[0]
        == volatility.parkinson_variance(adjusted).iloc[0]
    )


def test_har_regressors_are_trailing_means() -> None:
    parkinson = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    har = volatility.har_regressors(parkinson)

    assert har["rv_d"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert har["rv_w"].iloc[4] == 3.0  # mean(1..5)
    assert har["rv_w"].iloc[5] == 4.0  # mean(2..6)
    assert math.isnan(har["rv_w"].iloc[3])  # < 5 наблюдений — warm-up
    assert har["rv_m"].isna().all()  # < 22 наблюдений


def test_har_regressors_do_not_use_future() -> None:
    base = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    perturbed = base.copy()
    perturbed.iloc[6] = 999.0  # изменение будущего значения (индекс 6)

    har_base = volatility.har_regressors(base)
    har_perturbed = volatility.har_regressors(perturbed)

    # Трейлинговые окна: строки до индекса 6 не зависят от будущего.
    pd.testing.assert_frame_equal(har_base.iloc[:6], har_perturbed.iloc[:6])


def test_realized_variance_target_is_forward_sum() -> None:
    returns = pd.Series([np.nan, 0.1, 0.2, -0.1, 0.05])

    target = volatility.realized_variance_target(returns, horizon=2)

    # RV_target_t = r²_{t+1} + r²_{t+2}; хвост H → NaN.
    assert target.iloc[0] == 0.1**2 + 0.2**2
    assert target.iloc[1] == 0.2**2 + (-0.1) ** 2
    assert target.iloc[2] == (-0.1) ** 2 + 0.05**2
    assert math.isnan(target.iloc[3])
    assert math.isnan(target.iloc[4])
