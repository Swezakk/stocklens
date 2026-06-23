"""Unit-tests for technical indicators (ml-spec §4.4), including no-leakage."""

import math

import pandas as pd
from stocklens_ml.features import technical


def test_return_lags_shift_backward() -> None:
    returns = pd.Series([1.0, 2.0, 3.0])

    lags = technical.return_lags(returns, n_lags=2)

    assert lags["r_lag_0"].tolist() == [1.0, 2.0, 3.0]
    assert math.isnan(lags["r_lag_1"].iloc[0])
    assert lags["r_lag_1"].iloc[1] == 1.0
    assert lags["r_lag_1"].iloc[2] == 2.0


def test_rsi_is_100_for_monotonic_up_and_0_for_monotonic_down() -> None:
    up = pd.Series([float(i) for i in range(20)])
    down = pd.Series([float(20 - i) for i in range(20)])

    assert technical.rsi(up, period=14).iloc[-1] == 100.0
    assert technical.rsi(down, period=14).iloc[-1] == 0.0


def test_rsi_stays_within_bounds() -> None:
    close = pd.Series([10, 11, 10.5, 12, 11.8, 13, 12.5, 14, 13, 15, 14, 16, 15, 17, 16, 18.0])

    rsi = technical.rsi(close, period=14).dropna()

    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


def test_macd_is_zero_for_constant_series() -> None:
    close = pd.Series([100.0] * 40)

    macd = technical.macd(close)

    assert macd["macd"].abs().max() == 0.0
    assert macd["macd_signal"].abs().max() == 0.0
    assert macd["macd_hist"].abs().max() == 0.0


def test_volume_zscore_matches_formula() -> None:
    volume = pd.Series([1.0, 2.0, 3.0])

    z = technical.volume_zscore(volume, window=3)

    # mean([1,2,3])=2, std(ddof=1)=1 → (3-2)/1 = 1.0
    assert z.iloc[2] == 1.0
    assert math.isnan(z.iloc[1])  # < window наблюдений


def test_realized_vol_is_sqrt_of_rolling_sum() -> None:
    parkinson = pd.Series([1.0, 1.0, 1.0, 1.0])

    rv = technical.realized_vol(parkinson, window=3)

    assert rv.iloc[2] == math.sqrt(3.0)
    assert math.isnan(rv.iloc[1])


def test_indicators_do_not_use_future() -> None:
    close = pd.Series([float(i) + (i % 3) for i in range(40)])
    volume = pd.Series([float(100 + (i % 5) * 10) for i in range(40)])
    parkinson = pd.Series([0.01 + (i % 4) * 0.001 for i in range(40)])
    perturbed_close = close.copy()
    perturbed_close.iloc[-1] = 999.0

    rsi_base = technical.rsi(close).iloc[:-1]
    rsi_pert = technical.rsi(perturbed_close).iloc[:-1]
    macd_base = technical.macd(close).iloc[:-1]
    macd_pert = technical.macd(perturbed_close).iloc[:-1]

    pd.testing.assert_series_equal(rsi_base, rsi_pert)
    pd.testing.assert_frame_equal(macd_base, macd_pert)
    # z-объём и RV — трейлинговые: убедимся, что определены на хвосте (sanity).
    assert technical.volume_zscore(volume).iloc[-1] == technical.volume_zscore(volume).iloc[-1]
    assert technical.realized_vol(parkinson).iloc[-1] == technical.realized_vol(parkinson).iloc[-1]
