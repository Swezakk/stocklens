"""Tests for the volatility feature-frame assembly (ml-spec §4.6) — master no-leakage check."""

from datetime import date, timedelta

import pandas as pd
from stocklens_ml.features import assemble


def _candles(n: int, start: date = date(2022, 6, 1)) -> pd.DataFrame:
    rows = []
    for i in range(n):
        trade_date = start + timedelta(days=i)
        base = 100.0 + i + (i % 3)
        rows.append((trade_date, base, base + 2.0, base - 1.5, base + 0.5, 10 + i, False))
    return pd.DataFrame(
        rows,
        columns=["trade_date", "open", "high", "low", "close", "volume", "is_weekend_session"],
    )


def _empty_dividends() -> pd.DataFrame:
    return pd.DataFrame(columns=["ex_date", "value", "currency"])


def _empty_splits() -> pd.DataFrame:
    return pd.DataFrame(columns=["split_date", "before", "after"])


def test_frame_has_expected_columns_and_train_start_cut() -> None:
    candles = _candles(30)

    frame = assemble.build_volatility_frame(
        candles, _empty_dividends(), _empty_splits(), train_start=date(2022, 6, 10)
    )

    assert list(frame.columns) == ["trade_date", "r", "rv_d", "rv_w", "rv_m", "rv_target"]
    assert frame["trade_date"].min() >= date(2022, 6, 10)


def test_feature_columns_do_not_use_future() -> None:
    candles = _candles(30)
    perturbed = candles.copy()
    perturbed.loc[perturbed.index[-1], ["high", "low", "close"]] = [9999.0, 9000.0, 9500.0]

    frame = assemble.build_volatility_frame(
        candles, _empty_dividends(), _empty_splits(), train_start=date(2022, 6, 1)
    )
    frame_perturbed = assemble.build_volatility_frame(
        perturbed, _empty_dividends(), _empty_splits(), train_start=date(2022, 6, 1)
    )

    # Фичи (доходность, HAR-регрессоры) до последней строки не зависят от будущего.
    # rv_target — forward (таргет), его расхождение на ранних строках ожидаемо и не проверяется.
    feature_view = ["trade_date", "r", "rv_d", "rv_w", "rv_m"]
    pd.testing.assert_frame_equal(
        frame[feature_view].iloc[:-1], frame_perturbed[feature_view].iloc[:-1]
    )


def test_latest_rows_have_nan_target() -> None:
    candles = _candles(30)

    frame = assemble.build_volatility_frame(
        candles, _empty_dividends(), _empty_splits(), train_start=date(2022, 6, 1)
    )

    # Последние HORIZON строк не имеют полного будущего окна → таргет NaN.
    assert pd.isna(frame["rv_target"].iloc[-1])


def test_har_regressors_warm_up_is_nan_then_defined() -> None:
    candles = _candles(30)

    frame = assemble.build_volatility_frame(
        candles, _empty_dividends(), _empty_splits(), train_start=date(2022, 6, 1)
    )

    # rv_m требует 22 наблюдений → ранние строки NaN, поздние определены.
    assert pd.isna(frame["rv_m"].iloc[0])
    assert pd.notna(frame["rv_m"].iloc[-1])
