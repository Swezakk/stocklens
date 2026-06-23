"""Tests for the volatility serving wrapper (ml-spec §2.1, §7.1): self-contained pyfunc.

Гарантируем три свойства: (1) GARCH-ветка совпадает с эталоном models.garch (защита от
дрейфа инлайн-формулы), (2) HAR-ветка применяет линейные коэффициенты, (3) артефакт
самодостаточен — модуль не импортирует stocklens_ml (грузится в API-контейнере без него),
а round-trip через MLflow восстанавливает состояние версии и предсказывает.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stocklens_ml.models.garch import forecast_variance
from stocklens_ml.models.har import HAR_REGRESSORS
from stocklens_ml.registry import pyfunc_volatility
from stocklens_ml.registry.pyfunc_volatility import VolatilityModel, log_volatility_model

import mlflow


def _returns_frame(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.02, n)
    return pd.DataFrame(
        {
            "r": returns,
            "rv_d": np.abs(returns),
            "rv_w": np.abs(returns) * 0.9,
            "rv_m": np.abs(returns) * 0.8,
        }
    )


def test_garch_branch_matches_reference_forecast() -> None:
    frame = _returns_frame()

    forecast = VolatilityModel(method="garch").forecast(frame)
    reference = forecast_variance(frame["r"])

    assert forecast.shape == (1,)
    assert forecast[0] == pytest.approx(reference, rel=1e-6)


def test_har_branch_applies_linear_coefficients() -> None:
    coef = [0.5, 0.3, 0.2]
    intercept = 0.001
    model = VolatilityModel(method="har_rv", har_coef=coef, har_intercept=intercept)
    frame = pd.DataFrame({"r": [0.0], "rv_d": [0.002], "rv_w": [0.0015], "rv_m": [0.001]})

    forecast = model.forecast(frame)

    expected = 0.002 * 0.5 + 0.0015 * 0.3 + 0.001 * 0.2 + intercept
    assert forecast[0] == pytest.approx(expected)


def test_har_regressor_order_matches_model_module() -> None:
    """Порядок HAR-регрессоров в serving-обёртке == в модели: иначе пулинговые коэффициенты
    (фитятся по HAR_REGRESSORS) применятся к не тем колонкам в _forecast_har."""
    assert pyfunc_volatility._HAR_REGRESSORS == HAR_REGRESSORS


def test_module_has_no_stocklens_ml_import() -> None:
    """Артефакт грузится в API без stocklens_ml — модуль не должен его импортировать."""
    source = Path(pyfunc_volatility.__file__).read_text(encoding="utf-8")

    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "stocklens_ml" not in stripped, f"запрещённый импорт: {stripped}"


def test_log_and_load_round_trip_restores_state_and_predicts(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    frame = _returns_frame()
    metrics = {"qlike": 0.844, "qlike_baseline": 2.203, "rmse": 0.0025}

    with mlflow.start_run():
        info = log_volatility_model(method="garch", metrics=metrics, horizon=5, input_example=frame)

    loaded = mlflow.pyfunc.load_model(info.model_uri)
    forecast = loaded.predict(frame)
    underlying = loaded.unwrap_python_model()

    assert float(np.asarray(forecast)[0]) > 0.0
    assert underlying.method == "garch"
    assert underlying.metrics["qlike"] == pytest.approx(0.844)
    assert underlying.horizon == 5


def test_log_registers_model_version_when_name_given(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    frame = _returns_frame()

    with mlflow.start_run():
        info = log_volatility_model(
            method="garch",
            metrics={"qlike": 0.84, "qlike_baseline": 2.2, "rmse": 0.0025},
            horizon=5,
            input_example=frame,
            registered_model_name="stocklens-volatility",
        )

    # §7.2: predictions.model_version = str(mv.version); ModelInfo отдаёт int — кастует вызывающий.
    assert str(info.registered_model_version) == "1"
