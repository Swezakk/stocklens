"""Unit-tests for volatility evaluation metrics (ml-spec §6.2): QLIKE, RMSE."""

import math

import numpy as np
import pytest
from stocklens_ml.eval import metrics


def test_qlike_is_zero_for_perfect_forecast() -> None:
    variances = np.array([0.01, 0.02, 0.04])

    assert metrics.qlike(variances, variances) == pytest.approx(0.0)


def test_qlike_is_positive_for_imperfect_forecast() -> None:
    realized = np.array([0.02, 0.02])
    forecast = np.array([0.01, 0.04])

    assert metrics.qlike(realized, forecast) > 0.0


def test_qlike_penalizes_under_prediction_more_than_over() -> None:
    realized = np.array([1.0])
    # Недопрогноз дисперсии (h вдвое меньше) штрафуется сильнее, чем перепрогноз (h вдвое больше).
    under = metrics.qlike(realized, np.array([0.5]))
    over = metrics.qlike(realized, np.array([2.0]))

    assert under > over


def test_qlike_operates_on_variances_known_value() -> None:
    # rv=1, h=0.5 → ratio=2 → 2 - ln2 - 1 = 1 - ln2.
    result = metrics.qlike(np.array([1.0]), np.array([0.5]))

    assert result == pytest.approx(1.0 - math.log(2.0))


def test_qlike_ignores_nan_pairs() -> None:
    realized = np.array([1.0, np.nan, 1.0])
    forecast = np.array([1.0, 1.0, 1.0])

    assert metrics.qlike(realized, forecast) == pytest.approx(0.0)


def test_qlike_raises_when_no_valid_pairs() -> None:
    with pytest.raises(ValueError, match="QLIKE"):
        metrics.qlike(np.array([np.nan]), np.array([0.0]))


def test_rmse_matches_formula() -> None:
    actual = np.array([0.0, 0.0])
    predicted = np.array([2.0, 0.0])

    assert metrics.rmse(actual, predicted) == pytest.approx(math.sqrt(2.0))


def test_rmse_ignores_nan_pairs() -> None:
    actual = np.array([1.0, np.nan])
    predicted = np.array([1.0, 5.0])

    assert metrics.rmse(actual, predicted) == pytest.approx(0.0)
