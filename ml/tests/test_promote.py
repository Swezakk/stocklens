"""Tests for registry promotion via aliases (ml-spec §7.1, §12): champion / production.

Stages устарели — продвижение только алиасами. Проверяем: champion указывает на нужную
версию, переназначение переносит алиас на новую версию, production ставится независимо.
"""

from pathlib import Path

import pandas as pd
from stocklens_ml.registry import promote
from stocklens_ml.registry.pyfunc_volatility import log_volatility_model

import mlflow
from mlflow import MlflowClient

_MODEL = "stocklens-volatility"
_HAR_COEF = [0.5, 0.3, 0.2]
_HAR_INTERCEPT = 0.001
_METRICS = {"qlike": 0.58, "qlike_baseline": 1.47, "rmse": 0.0018}


def _register_har_version() -> str:
    frame = pd.DataFrame({"r": [0.0], "rv_d": [0.002], "rv_w": [0.0015], "rv_m": [0.001]})
    with mlflow.start_run():
        info = log_volatility_model(
            method="har_rv",
            metrics=_METRICS,
            horizon=5,
            input_example=frame,
            har_coef=_HAR_COEF,
            har_intercept=_HAR_INTERCEPT,
            registered_model_name=_MODEL,
        )
    return str(info.registered_model_version)


def test_mark_champion_assigns_alias_to_version(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    version = _register_har_version()
    client = MlflowClient()

    promote.mark_champion(client, _MODEL, version)

    champion = client.get_model_version_by_alias(_MODEL, promote.CHAMPION_ALIAS)
    assert str(champion.version) == version


def test_mark_champion_moves_alias_to_newer_version(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    client = MlflowClient()
    first = _register_har_version()
    promote.mark_champion(client, _MODEL, first)
    second = _register_har_version()

    promote.mark_champion(client, _MODEL, second)

    champion = client.get_model_version_by_alias(_MODEL, promote.CHAMPION_ALIAS)
    assert second != first
    assert str(champion.version) == second


def test_promote_to_production_sets_independent_alias(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    version = _register_har_version()
    client = MlflowClient()

    promote.mark_champion(client, _MODEL, version)
    promote.promote_to_production(client, _MODEL, version)

    production = client.get_model_version_by_alias(_MODEL, promote.PRODUCTION_ALIAS)
    assert str(production.version) == version
    assert promote.PRODUCTION_ALIAS != promote.CHAMPION_ALIAS
