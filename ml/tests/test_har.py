"""Unit-tests for the HAR-RV OLS volatility model (ml-spec §5.2)."""

import numpy as np
import pandas as pd
import pytest
from stocklens_ml.models.har import HarRvModel


def _regressors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rv_d": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            "rv_w": [0.08, 0.06, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01],
            "rv_m": [0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04],
        }
    )


def _linear_target(regressors: pd.DataFrame) -> pd.Series:
    return (
        1.5 * regressors["rv_d"] + 0.5 * regressors["rv_w"] - 0.2 * regressors["rv_m"] + 0.01
    ).rename("rv_target")


def test_har_recovers_exact_linear_relationship() -> None:
    regressors = _regressors()
    target = _linear_target(regressors)

    model = HarRvModel().fit(regressors, target)
    prediction = model.predict(regressors)

    # OLS точно восстанавливает линейную зависимость (это линейная регрессия, не ML).
    assert np.allclose(prediction, target.to_numpy())


def test_har_fit_ignores_nan_rows() -> None:
    regressors = _regressors()
    target = _linear_target(regressors)
    regressors.loc[2, "rv_w"] = np.nan  # строка с NaN-регрессором отбрасывается при обучении

    model = HarRvModel().fit(regressors, target)
    clean = regressors.drop(index=2)
    prediction = model.predict(clean)

    assert np.allclose(prediction, _linear_target(_regressors()).drop(index=2).to_numpy())


def test_har_raises_when_all_rows_invalid() -> None:
    regressors = pd.DataFrame({"rv_d": [np.nan], "rv_w": [np.nan], "rv_m": [np.nan]})
    target = pd.Series([np.nan])

    with pytest.raises(ValueError, match="HAR-RV"):
        HarRvModel().fit(regressors, target)


def test_har_coefficients_reproduce_prediction_as_plain_linear() -> None:
    """Коэффициенты в порядке HAR_REGRESSORS: X @ coef + intercept == predict (для serving)."""
    regressors = _regressors()
    target = _linear_target(regressors)
    model = HarRvModel().fit(regressors, target)

    coef, intercept = model.coefficients()
    manual = regressors[["rv_d", "rv_w", "rv_m"]].to_numpy() @ coef + intercept

    assert np.allclose(manual, model.predict(regressors))
