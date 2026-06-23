"""Tests for the walk-forward harness (ml-spec §6.1): aggregation, gap, forecasters."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from stocklens_ml.eval import walk_forward


def _frame(n: int) -> pd.DataFrame:
    idx = np.arange(n)
    return pd.DataFrame(
        {
            "r": 0.01 * np.cos(idx) + 0.002,
            "rv_d": 0.001 + 0.0002 * (idx % 5),
            "rv_w": 0.001 + 0.0002 * (idx % 7),
            "rv_m": 0.001 + 0.0002 * (idx % 11),
            "rv_target": 0.002 + 0.0003 * (idx % 4),
        }
    )


def _perfect(
    frame: pd.DataFrame, train_idx: npt.NDArray[np.intp], test_idx: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    return np.asarray(frame["rv_target"].to_numpy(dtype=float)[test_idx], dtype=np.float64)


def _constant(
    frame: pd.DataFrame, train_idx: npt.NDArray[np.intp], test_idx: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    return np.full(len(test_idx), 0.001)


def test_evaluate_perfect_forecaster_has_zero_qlike_and_beats_constant() -> None:
    frame = _frame(60)

    result = walk_forward.evaluate(
        frame, {"perfect": _perfect, "const": _constant}, n_splits=3, gap=5
    )

    assert result["perfect"]["qlike"] == pytest.approx(0.0, abs=1e-9)
    assert result["perfect"]["qlike"] < result["const"]["qlike"]


def test_evaluate_applies_gap_between_train_and_test() -> None:
    captured: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]] = []

    def recorder(
        frame: pd.DataFrame, train_idx: npt.NDArray[np.intp], test_idx: npt.NDArray[np.intp]
    ) -> npt.NDArray[np.float64]:
        captured.append((train_idx, test_idx))
        return np.asarray(frame["rv_target"].to_numpy(dtype=float)[test_idx], dtype=np.float64)

    walk_forward.evaluate(_frame(60), {"rec": recorder}, n_splits=3, gap=5)

    for train_idx, test_idx in captured:
        # gap=5: между последней train-строкой и первой test-строкой ровно 5 пропущенных.
        assert int(test_idx.min()) - int(train_idx.max()) - 1 == 5


def test_rw_rv_and_har_forecasters_produce_finite_metrics() -> None:
    frame = _frame(60)

    result = walk_forward.evaluate(
        frame,
        {"rw": walk_forward.rw_rv_forecaster, "har": walk_forward.har_forecaster},
        n_splits=3,
        gap=5,
    )

    assert np.isfinite(result["rw"]["qlike"])
    assert np.isfinite(result["har"]["qlike"])


def test_garch_forecaster_produces_finite_positive_forecast() -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"r": rng.normal(0.0, 0.02, 150)})

    forecast = walk_forward.garch_forecaster(
        frame, np.arange(149, dtype=np.intp), np.array([149], dtype=np.intp)
    )

    assert forecast.shape == (1,)
    assert np.isfinite(forecast[0])
    assert forecast[0] > 0.0
