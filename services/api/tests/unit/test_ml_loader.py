"""Unit-тест loader: маппинг pyfunc → LoadedVolatilityModel и degraded-путь (без сети).

Пинит контракт между loader и pyfunc-обёрткой (5g): имена атрибутов unwrap'нутой модели
(method/metrics/horizon) и форму version из реестра. Реальный round-trip к реестру — §11.3
(тикеты a1c4f7e2/b9d3e5a8). Маппинг проверяется прямым вызовом ``_load_volatility`` (один
вызов, без retry-петли); degraded — патчем самого ``_load_volatility`` (mlflow не дёргается).
"""

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from api.core.settings import ApiSettings
from api.ml import loader
from api.ml.bundle import LoadedVolatilityModel
from api.ml.trend import CatBoostTrendPredictor


class _FakeVolatilityModel:
    method = "garch"
    metrics: ClassVar[dict[str, float]] = {"qlike": 0.844, "qlike_baseline": 2.203, "rmse": 0.0025}
    horizon = 5

    def forecast(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.array([0.0009], dtype=np.float64)


def _settings(**overrides: object) -> ApiSettings:
    return ApiSettings.model_validate(
        {
            "database_url_async": "postgresql+asyncpg://u:p@h:5432/d",
            "redis_url": "redis://h:6379/0",
            **overrides,
        }
    )


def test_load_volatility_maps_pyfunc_fields() -> None:
    fake_pyfunc = MagicMock()
    fake_pyfunc.unwrap_python_model.return_value = _FakeVolatilityModel()
    fake_client = MagicMock()
    fake_client.get_model_version_by_alias.return_value = SimpleNamespace(version=7)

    with (
        patch("mlflow.pyfunc.load_model", return_value=fake_pyfunc),
        patch("api.ml.loader.MlflowClient", return_value=fake_client),
    ):
        model = loader._load_volatility(_settings())

    assert model.model_version == "7"  # int версии реестра → str (§7.2)
    assert model.method == "garch"
    assert model.horizon_days == 5
    assert model.metrics["qlike"] == pytest.approx(0.844)


def test_load_trend_maps_native_catboost_fields() -> None:
    fake_model = MagicMock()
    fake_client = MagicMock()
    fake_client.get_model_version_by_alias.return_value = SimpleNamespace(version=3)

    with (
        patch("mlflow.catboost.load_model", return_value=fake_model) as load_model,
        patch("api.ml.loader.MlflowClient", return_value=fake_client),
    ):
        model = loader._load_trend(_settings())

    load_model.assert_called_once_with("models:/stocklens-trend@production")
    assert model.model_version == "3"  # int версии реестра → str (§7.3)
    assert model.horizon_days == 5  # из stocklens_ml.config.HORIZON_DAYS (нативный артефакт)
    assert isinstance(model.predictor, CatBoostTrendPredictor)
    assert model.predictor._model is fake_model  # нативный артефакт обёрнут адаптером


def test_load_bundle_returns_empty_volatility_when_volatility_load_fails() -> None:
    # Тренд тоже патчится: после рестракта load_bundle грузит обе модели — без патча тренд
    # ушёл бы в сеть (mlflow:5000) внутри unit-прогона. Это сохранение изоляции, не ослабление.
    with (
        patch("api.ml.loader._load_volatility", side_effect=RuntimeError("registry down")),
        patch("api.ml.loader._load_trend", side_effect=RuntimeError("registry down")),
    ):
        bundle = loader.load_bundle(_settings(ml_load_attempts=1, ml_load_interval_seconds=0.0))

    assert bundle.volatility is None  # readiness репортит degraded, процесс не падает (§8.2)
    assert bundle.trend is None


def test_load_bundle_keeps_volatility_when_trend_load_fails() -> None:
    # Инвариант «ошибка одного источника не валит остальные»: промах тренда (независимый
    # try/except) оставляет волатильность загруженной, а не обнуляет bundle.
    loaded_volatility = LoadedVolatilityModel(
        predictor=_FakeVolatilityModel(),
        model_version="7",
        method="garch",
        metrics={"qlike": 0.844},
        horizon_days=5,
    )
    with (
        patch("api.ml.loader._load_volatility", return_value=loaded_volatility),
        patch("api.ml.loader._load_trend", side_effect=RuntimeError("trend registry down")),
    ):
        bundle = loader.load_bundle(_settings(ml_load_attempts=1, ml_load_interval_seconds=0.0))

    assert bundle.volatility is loaded_volatility  # волатильность не затронута промахом тренда
    assert bundle.trend is None
    assert bundle.ready() is True  # тренд не блокирует readiness
