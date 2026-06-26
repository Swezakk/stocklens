"""Tests for trend training orchestration (ml-spec §11, §12): winner selection via the
ROC-AUC baseline gate and native-CatBoost champion registration."""

from pathlib import Path

import mlflow.catboost
import numpy as np
import pandas as pd
import pytest
from stocklens_ml.config import MlSettings
from stocklens_ml.features.assemble import (
    TREND_FEATURE_COLUMNS,
    TREND_TARGET_COLUMN,
)
from stocklens_ml.models.trend import TrendHyperparams
from stocklens_ml.training import train_trend

import mlflow
from mlflow import MlflowClient

_TINY_ITERATIONS = 40
_SEED = 7


def _settings() -> MlSettings:
    return MlSettings.model_validate(
        {"database_url": "postgresql+psycopg://user:pass@localhost:5432/db"}
    )


def _planted_frame(n: int = 360, nan_tail: int = 5, seed: int = _SEED) -> pd.DataFrame:
    """Фрейм тренда с обучаемым направленным сигналом и NaN-хвостом таргета.

    Таргет строго определяется знаком первого лага доходности → CatBoost обязан разделить
    классы (ROC-AUC > 0.5). Последние ``nan_tail`` строк имеют NaN-таргет (как реальный
    forward-таргет из build_trend_frame) — проверяет, что регистрация их отбрасывает, а не
    падает на NaN-метках в eval_set.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame()
    driver = rng.normal(size=n)
    frame["r_lag_0"] = driver
    for lag in range(1, 5):
        frame[f"r_lag_{lag}"] = rng.normal(size=n)
    frame["rsi"] = rng.normal(size=n)
    frame["macd"] = rng.normal(size=n)
    frame["macd_signal"] = rng.normal(size=n)
    frame["macd_hist"] = rng.normal(size=n)
    frame["volume_zscore"] = rng.normal(size=n)
    frame["realized_vol"] = np.abs(rng.normal(size=n))
    target = (driver > 0.0).astype(float)
    target[-nan_tail:] = np.nan
    frame[TREND_TARGET_COLUMN] = target
    assert list(frame.columns) == [*TREND_FEATURE_COLUMNS, TREND_TARGET_COLUMN]
    return frame


def test_default_hyperparams_match_spec_starting_values() -> None:
    # Дефолты TrendHyperparams = стартовые значения спеки §5.4 (600/4/0.03/6.0).
    hp = TrendHyperparams()
    assert hp.iterations == 600
    assert hp.depth == 4
    assert hp.learning_rate == pytest.approx(0.03)
    assert hp.l2_leaf_reg == pytest.approx(6.0)


def _labelled_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Фрейм тренда без NaN-хвоста таргета (метки обязательны для fit/eval_set)."""
    frame = train_trend._drop_unlabelled(_planted_frame())
    return frame[TREND_FEATURE_COLUMNS], frame[TREND_TARGET_COLUMN]


def test_fit_with_purged_validation_honors_hyperparams() -> None:
    # Переданный TrendHyperparams строит TrendModel с теми же глубиной/L2 (анти-хардкод).
    x, y = _labelled_xy()
    train_idx = np.arange(len(x), dtype=np.intp)
    hp = TrendHyperparams(iterations=_TINY_ITERATIONS, depth=2, l2_leaf_reg=12.0)

    model = train_trend._fit_with_purged_validation(
        x, y, train_idx=train_idx, horizon=5, hyperparams=hp
    )

    params = model._model.get_params()
    assert params["depth"] == 2
    assert params["l2_leaf_reg"] == pytest.approx(12.0)
    assert params["iterations"] == _TINY_ITERATIONS


def test_fit_with_purged_validation_iterations_override_wins() -> None:
    # Явный iterations переопределяет iterations внутри hyperparams (контракт тестов unchanged).
    x, y = _labelled_xy()
    train_idx = np.arange(len(x), dtype=np.intp)
    hp = TrendHyperparams(iterations=600, depth=3)

    model = train_trend._fit_with_purged_validation(
        x, y, train_idx=train_idx, horizon=5, iterations=_TINY_ITERATIONS, hyperparams=hp
    )

    params = model._model.get_params()
    assert params["iterations"] == _TINY_ITERATIONS
    assert params["depth"] == 3


def test_fit_with_purged_validation_default_uses_spec_starting_values() -> None:
    # Дефолтный путь (без hyperparams) строит модель со стартовыми значениями спеки §5.4.
    x, y = _labelled_xy()
    train_idx = np.arange(len(x), dtype=np.intp)

    model = train_trend._fit_with_purged_validation(
        x, y, train_idx=train_idx, horizon=5, iterations=_TINY_ITERATIONS
    )

    params = model._model.get_params()
    assert params["depth"] == 4
    assert params["l2_leaf_reg"] == pytest.approx(6.0)
    assert params["learning_rate"] == pytest.approx(0.03)


def test_select_winner_returns_model_exceeding_baseline_roc_auc() -> None:
    results = {
        "SBER": {
            "baseline_always_up": {"accuracy": 0.55, "f1": 0.70, "roc_auc": 0.50},
            "trend_catboost": {"accuracy": 0.62, "f1": 0.64, "roc_auc": 0.61},
        }
    }

    winner = train_trend.select_winner(results)

    assert winner is not None
    assert winner.method == "trend_catboost"
    assert winner.metrics["roc_auc"] == pytest.approx(0.61)
    assert winner.metrics["roc_auc_baseline"] == pytest.approx(0.50)


def test_select_winner_returns_none_when_no_model_exceeds_baseline() -> None:
    # Один кандидат ровно на baseline (0.5), другой ниже → строгий гейт отвергает обоих.
    results = {
        "SBER": {
            "baseline_always_up": {"accuracy": 0.55, "f1": 0.70, "roc_auc": 0.50},
            "trend_catboost": {"accuracy": 0.55, "f1": 0.60, "roc_auc": 0.50},
            "weak_model": {"accuracy": 0.50, "f1": 0.50, "roc_auc": 0.47},
        }
    }

    assert train_trend.select_winner(results) is None


def test_select_winner_picks_highest_mean_roc_auc() -> None:
    results = {
        "SBER": {
            "baseline_always_up": {"accuracy": 0.55, "f1": 0.70, "roc_auc": 0.50},
            "trend_catboost": {"accuracy": 0.60, "f1": 0.62, "roc_auc": 0.58},
            "other": {"accuracy": 0.61, "f1": 0.63, "roc_auc": 0.66},
        }
    }

    winner = train_trend.select_winner(results)

    assert winner is not None
    assert winner.method == "other"


def test_log_run_records_classification_metrics_and_gain(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    metrics = {
        "baseline_always_up": {"accuracy": 0.55, "f1": 0.70, "roc_auc": 0.50},
        "trend_catboost": {"accuracy": 0.62, "f1": 0.64, "roc_auc": 0.61},
    }

    train_trend.log_run("TEST", metrics, _settings(), n_splits=5)

    runs = mlflow.search_runs(experiment_names=[train_trend._EXPERIMENT])
    assert not isinstance(runs, list)  # output_format по умолчанию — pandas DataFrame
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["params.ticker"] == "TEST"
    assert row["metrics.roc_auc_trend_catboost"] == pytest.approx(0.61)
    assert row["metrics.accuracy_trend_catboost"] == pytest.approx(0.62)
    assert row["metrics.f1_trend_catboost"] == pytest.approx(0.64)
    # roc_auc_gain = value − baseline (0.50): 0.61 − 0.50 = 0.11.
    assert row["metrics.roc_auc_gain_trend_catboost"] == pytest.approx(0.11)


def test_register_champion_registers_model_and_marks_alias(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    winner = train_trend.WinnerSelection(
        "trend_catboost", {"roc_auc": 0.61, "roc_auc_baseline": 0.50, "accuracy": 0.62, "f1": 0.64}
    )

    info = train_trend.register_champion(
        _settings(),
        winner,
        [_planted_frame()],
        iterations=_TINY_ITERATIONS,
    )

    client = MlflowClient()
    champion = client.get_model_version_by_alias(_settings().trend_model_name, "champion")
    assert str(champion.version) == str(info.registered_model_version)
    assert champion.version is not None


def test_register_champion_fits_final_model_with_passed_hyperparams(tmp_path: Path) -> None:
    # Финализированный sweep'ом конфиг (depth=2) применяется к финальному фиту champion.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    winner = train_trend.WinnerSelection(
        "trend_catboost", {"roc_auc": 0.61, "roc_auc_baseline": 0.50, "accuracy": 0.62, "f1": 0.64}
    )
    hp = TrendHyperparams(iterations=_TINY_ITERATIONS, depth=2, l2_leaf_reg=12.0)

    info = train_trend.register_champion(_settings(), winner, [_planted_frame()], hyperparams=hp)

    loaded = mlflow.catboost.load_model(info.model_uri)
    params = loaded.get_params()
    assert params["depth"] == 2
    assert params["l2_leaf_reg"] == pytest.approx(12.0)
