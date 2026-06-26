"""Tests for the trend hyperparameter sweep (ml-spec §5.4): a small conservative grid,
per-config MLflow logging under "trend-tuning" without registration, per-ticker isolation,
and a search-only run that never registers a model."""

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from stocklens_ml.config import MlSettings
from stocklens_ml.features.assemble import TREND_FEATURE_COLUMNS, TREND_TARGET_COLUMN
from stocklens_ml.models.trend import TrendHyperparams
from stocklens_ml.training import tune_trend

import mlflow
from mlflow import MlflowClient

_TINY_ITERATIONS = 30
_SEED = 13
_MAX_GRID_SIZE = 9
_MIN_L2 = 6.0
_MAX_DEPTH = 4


def _settings() -> MlSettings:
    return MlSettings.model_validate(
        {"database_url": "postgresql+psycopg://user:pass@localhost:5432/db"}
    )


def _learnable_frame(n: int = 360, seed: int = _SEED) -> pd.DataFrame:
    """Фрейм тренда с обучаемым сигналом: таргет = знак первого лага → ROC-AUC > 0.5.

    Оба класса присутствуют по всему окну (signal детерминирован знаком драйвера),
    поэтому ни train-, ни test-фолды не вырождаются в один класс.
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
    frame[TREND_TARGET_COLUMN] = (driver > 0.0).astype(float)
    assert list(frame.columns) == [*TREND_FEATURE_COLUMNS, TREND_TARGET_COLUMN]
    return frame


def _single_class_test_tail_frame(n: int = 360, seed: int = _SEED) -> pd.DataFrame:
    """Фрейм, где объединённая test-выборка walk-forward одноклассовая → ROC-AUC ValueError.

    Оба класса лежат в ранней (всегда train) части окна — CatBoost.fit видит оба класса и не
    падает. Хвост (все test-фолды expanding-сплиттера) — единственный класс 1, поэтому
    конкатенированный realized одноклассовый и ``classification_metrics.roc_auc`` бросает
    ValueError. Это и проверяет изоляцию: тикер пропускается, не валит весь sweep.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame()
    for lag in range(5):
        frame[f"r_lag_{lag}"] = rng.normal(size=n)
    frame["rsi"] = rng.normal(size=n)
    frame["macd"] = rng.normal(size=n)
    frame["macd_signal"] = rng.normal(size=n)
    frame["macd_hist"] = rng.normal(size=n)
    frame["volume_zscore"] = rng.normal(size=n)
    frame["realized_vol"] = np.abs(rng.normal(size=n))
    target = np.ones(n)
    target[: n // 4] = (rng.normal(size=n // 4) > 0.0).astype(float)
    frame[TREND_TARGET_COLUMN] = target
    return frame


def _single_class_train_fold_frame(n: int = 360, seed: int = _SEED) -> pd.DataFrame:
    """Фрейм, где первый expanding train-фолд одноклассовый → CatBoost.fit бросает CatBoostError.

    Ранняя треть окна — единственный класс 0 (первый train-фолд walk-forward не видит второго
    класса, CatBoost отказывается обучаться); дальше оба класса. CatBoostError наследует Exception
    напрямую (не ValueError) — этот фрейм проверяет именно ветку CatBoost изоляции, а не roc_auc.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame()
    for lag in range(5):
        frame[f"r_lag_{lag}"] = rng.normal(size=n)
    frame["rsi"] = rng.normal(size=n)
    frame["macd"] = rng.normal(size=n)
    frame["macd_signal"] = rng.normal(size=n)
    frame["macd_hist"] = rng.normal(size=n)
    frame["volume_zscore"] = rng.normal(size=n)
    frame["realized_vol"] = np.abs(rng.normal(size=n))
    target = np.ones(n)
    target[: n // 3] = 0.0
    target[n // 3 :] = (rng.normal(size=n - n // 3) > 0.0).astype(float)
    frame[TREND_TARGET_COLUMN] = target
    return frame


def _tiny_grid() -> list[TrendHyperparams]:
    """Двухконфигурный грид с крошечными iterations — для быстрых тестов sweep."""
    return [
        TrendHyperparams(iterations=_TINY_ITERATIONS, depth=2, l2_leaf_reg=12.0),
        TrendHyperparams(iterations=_TINY_ITERATIONS, depth=3, l2_leaf_reg=6.0),
    ]


def test_hyperparameter_grid_is_small_and_conservative() -> None:
    grid = tune_trend._hyperparameter_grid()

    assert len(grid) <= _MAX_GRID_SIZE
    assert grid, "грид не должен быть пустым"
    for config in grid:
        assert config.depth <= _MAX_DEPTH, f"глубина {config.depth} > {_MAX_DEPTH} (переобучение)"
        assert config.l2_leaf_reg >= _MIN_L2, f"L2 {config.l2_leaf_reg} < {_MIN_L2} (слабая регул.)"


def test_evaluate_grid_logs_one_run_per_config_without_registering(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    grid = _tiny_grid()
    frames = {"AAA": _learnable_frame(seed=1), "BBB": _learnable_frame(seed=2)}

    with structlog.testing.capture_logs() as logs:
        result = tune_trend.evaluate_grid(frames, _settings(), n_splits=3, grid=grid)

    runs = mlflow.search_runs(experiment_names=[tune_trend._EXPERIMENT])
    assert not isinstance(runs, list)
    assert len(runs) == len(grid)
    assert result.best is not None
    assert MlflowClient().search_registered_models() == []
    complete = [entry for entry in logs if entry["event"] == "tuning_complete"]
    assert len(complete) == 1
    assert "beats_baseline" in complete[0]
    assert "mean_roc_auc" in complete[0]


def test_evaluate_grid_skips_ticker_that_raises_and_completes(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    grid = _tiny_grid()
    frames = {
        "GOOD1": _learnable_frame(seed=1),
        "BAD": _single_class_test_tail_frame(seed=2),
        "GOOD2": _learnable_frame(seed=3),
    }

    with structlog.testing.capture_logs() as logs:
        result = tune_trend.evaluate_grid(frames, _settings(), n_splits=3, grid=grid)

    assert result.best is not None
    skipped = [entry for entry in logs if entry["event"] == "ticker_skipped"]
    assert any(entry["ticker"] == "BAD" for entry in skipped)
    runs = mlflow.search_runs(experiment_names=[tune_trend._EXPERIMENT])
    assert len(runs) == len(grid)
    assert MlflowClient().search_registered_models() == []


def test_evaluate_grid_skips_ticker_whose_catboost_fit_raises(tmp_path: Path) -> None:
    # CatBoostError (одноклассовый train-фолд) наследует Exception, не ValueError — sweep не должен
    # падать из-за такого тикера, иначе изоляция дырявая.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    grid = _tiny_grid()
    frames = {
        "GOOD1": _learnable_frame(seed=1),
        "DEGENERATE": _single_class_train_fold_frame(seed=2),
        "GOOD2": _learnable_frame(seed=3),
    }

    with structlog.testing.capture_logs() as logs:
        result = tune_trend.evaluate_grid(frames, _settings(), n_splits=3, grid=grid)

    assert result.best is not None
    skipped = [entry for entry in logs if entry["event"] == "ticker_skipped"]
    assert any(entry["ticker"] == "DEGENERATE" for entry in skipped)
    runs = mlflow.search_runs(experiment_names=[tune_trend._EXPERIMENT])
    assert len(runs) == len(grid)


def test_evaluate_grid_best_reports_whether_it_beats_baseline(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    grid = _tiny_grid()
    frames = {"AAA": _learnable_frame(seed=1), "BBB": _learnable_frame(seed=2)}

    result = tune_trend.evaluate_grid(frames, _settings(), n_splits=3, grid=grid)

    assert result.best is not None
    # Сигнал детерминирован (таргет = знак лага) → лучший конфиг обязан бить baseline 0.5.
    assert result.best_mean_roc_auc > tune_trend._BASELINE_ROC_AUC
    assert result.beats_baseline is True
