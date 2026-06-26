"""Tests for the trend model-family comparison (sanity check, ml-spec §5.4, §6.2): a
one-shot confirmatory experiment that scores OTHER model families (logreg, random forest)
on the SAME features/walk-forward as the reference CatBoost. Covers per-fold scaler leakage
control (scaler fit on TRAIN only), forecaster output contract (p_up in [0,1]), warm-up
NaN-feature dropping, a three-family sweep that logs one MLflow run per family under
"trend-model-comparison" without registering any model, and per-ticker isolation."""

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from stocklens_ml.config import MlSettings
from stocklens_ml.features.assemble import TREND_FEATURE_COLUMNS, TREND_TARGET_COLUMN
from stocklens_ml.training import compare_trend_models

import mlflow
from mlflow import MlflowClient

_SEED = 17
_RF_TINY_ESTIMATORS = 20
_EXPECTED_FAMILIES = 3


def _settings() -> MlSettings:
    return MlSettings.model_validate(
        {"database_url": "postgresql+psycopg://user:pass@localhost:5432/db"}
    )


def _learnable_frame(n: int = 360, seed: int = _SEED) -> pd.DataFrame:
    """Фрейм тренда с обучаемым сигналом: таргет = знак первого лага → ROC-AUC > 0.5.

    Оба класса присутствуют по всему окну (signal детерминирован знаком драйвера),
    поэтому ни train-, ни test-фолды не вырождаются в один класс. Фичи без NaN —
    _drop_unfeatured ничего не отбрасывает.
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


def _single_class_train_fold_frame(n: int = 360, seed: int = _SEED) -> pd.DataFrame:
    """Фрейм, где первый expanding train-фолд одноклассовый → каждая семья пропускает тикер.

    Ранняя треть окна — единственный класс 0; дальше оба класса. Первый train-фолд
    walk-forward видит только класс 0: logreg бросает ValueError, CatBoost — CatBoostError,
    RandomForest без явного guard'а фитится на одном классе и предсказал бы proba формы
    (n, 1) (→ IndexError на [:, 1]) — форкастер RF обязан явно поднять ValueError, чтобы
    изоляция отработала для ВСЕХ трёх семей.
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


def _mean_shifted_frame(n: int = 120, seed: int = _SEED) -> pd.DataFrame:
    """Фрейм, где средние фич по test-региону СДВИНУТЫ относительно train-региона.

    Цель — сделать тест на утечку дискриминирующим: средние по всему фрейму ≠ средние по
    train-окну. Если бы скейлер фитился на всём фрейме (утечка), его ``mean_`` совпал бы с
    полным средним, а не с train-средним. Сдвиг в хвосте гарантирует, что эти средние
    различимы.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame()
    for col in TREND_FEATURE_COLUMNS:
        frame[col] = rng.normal(size=n)
    # Сдвинуть хвост (test-регион) на крупную константу — полное среднее уедет от train-среднего.
    tail = n // 3
    for col in TREND_FEATURE_COLUMNS:
        frame.loc[frame.index[-tail:], col] = frame[col].iloc[-tail:] + 100.0
    frame[TREND_TARGET_COLUMN] = (rng.normal(size=n) > 0.0).astype(float)
    return frame


def test_drop_unfeatured_removes_warmup_nan_rows_and_resets_index() -> None:
    frame = _learnable_frame(n=10)
    # Первые две строки — warm-up: NaN хотя бы в одной фиче (как реальный warm-up окна).
    frame.loc[0, "rsi"] = np.nan
    frame.loc[1, "macd"] = np.nan

    cleaned = compare_trend_models._drop_unfeatured(frame)

    assert len(cleaned) == len(frame) - 2
    assert not cleaned[TREND_FEATURE_COLUMNS].isna().to_numpy().any()
    # Индекс сброшен (позиционная согласованность с .iloc[train_idx] из TimeSeriesSplit).
    assert list(cleaned.index) == list(range(len(cleaned)))


def test_fit_train_scaler_is_fit_on_train_rows_only_not_whole_frame() -> None:
    # Утечка препроцессинга — главный риск проекта: скейлер ОБЯЗАН фититься на train-строках.
    frame = _mean_shifted_frame()
    x = frame[TREND_FEATURE_COLUMNS]
    n = len(frame)
    train_idx = np.arange(n // 2, dtype=np.intp)  # ранняя половина — без сдвинутого хвоста

    scaler = compare_trend_models._fit_train_scaler(x, train_idx)

    train_mean = x.iloc[train_idx].to_numpy().mean(axis=0)
    whole_mean = x.to_numpy().mean(axis=0)
    # Скейлер совпадает с train-средним, НЕ с полным средним (иначе была бы утечка).
    np.testing.assert_allclose(scaler.mean_, train_mean)
    assert not np.allclose(scaler.mean_, whole_mean), "скейлер не должен видеть полный фрейм"


def test_each_forecaster_returns_p_up_in_unit_range() -> None:
    frame = _learnable_frame(n=120)
    n = len(frame)
    train_idx = np.arange(0, 80, dtype=np.intp)
    test_idx = np.arange(80, n, dtype=np.intp)
    forecasters = {
        compare_trend_models.FAMILY_CATBOOST: compare_trend_models.build_forecaster(
            compare_trend_models.FAMILY_CATBOOST, horizon=5
        ),
        compare_trend_models.FAMILY_LOGREG: compare_trend_models.build_forecaster(
            compare_trend_models.FAMILY_LOGREG, horizon=5
        ),
        compare_trend_models.FAMILY_RANDOM_FOREST: compare_trend_models.build_forecaster(
            compare_trend_models.FAMILY_RANDOM_FOREST,
            horizon=5,
            rf_n_estimators=_RF_TINY_ESTIMATORS,
        ),
    }

    for family, forecaster in forecasters.items():
        p_up = forecaster(frame, train_idx, test_idx)
        assert p_up.shape == (len(test_idx),), family
        assert p_up.dtype == np.float64, family
        assert np.all((p_up >= 0.0) & (p_up <= 1.0)), family


def test_compare_families_logs_one_run_per_family_without_registering(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    frames = {
        "AAA": _learnable_frame(seed=1),
        "BBB": _learnable_frame(seed=2),
        "CCC": _learnable_frame(seed=3),
    }

    with structlog.testing.capture_logs() as logs:
        summary = compare_trend_models.compare_families(
            frames, _settings(), n_splits=3, rf_n_estimators=_RF_TINY_ESTIMATORS
        )

    runs = mlflow.search_runs(experiment_names=[compare_trend_models._EXPERIMENT])
    assert not isinstance(runs, list)
    # Один прогон на семью — три семьи.
    assert len(runs) == _EXPECTED_FAMILIES
    # Регистрации НЕТ — это confirmatory эксперимент, не train.
    assert MlflowClient().search_registered_models() == []
    assert len(summary) == _EXPECTED_FAMILIES
    complete = [entry for entry in logs if entry["event"] == "comparison_complete"]
    assert len(complete) == 1
    families = complete[0]["families"]
    assert len(families) == _EXPECTED_FAMILIES
    assert {item["family"] for item in families} == {
        compare_trend_models.FAMILY_CATBOOST,
        compare_trend_models.FAMILY_LOGREG,
        compare_trend_models.FAMILY_RANDOM_FOREST,
    }
    for item in families:
        assert "mean_roc_auc" in item
        assert "beats_baseline" in item


def test_compare_families_skips_single_class_ticker_for_all_families(tmp_path: Path) -> None:
    # Одноклассовый train-фолд: logreg→ValueError, CatBoost→CatBoostError, RF→guard ValueError.
    # Все три семьи должны пропустить плохой тикер и оценить остальные (изоляция держит).
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    frames = {
        "GOOD1": _learnable_frame(seed=1),
        "DEGENERATE": _single_class_train_fold_frame(seed=2),
        "GOOD2": _learnable_frame(seed=3),
    }

    with structlog.testing.capture_logs() as logs:
        summary = compare_trend_models.compare_families(
            frames, _settings(), n_splits=3, rf_n_estimators=_RF_TINY_ESTIMATORS
        )

    assert len(summary) == _EXPECTED_FAMILIES
    skipped = [entry for entry in logs if entry["event"] == "ticker_skipped"]
    skipped_families = {(entry["family"], entry["ticker"]) for entry in skipped}
    # Плохой тикер пропущен КАЖДОЙ из трёх семей — иначе RF упал бы на IndexError.
    assert (compare_trend_models.FAMILY_CATBOOST, "DEGENERATE") in skipped_families
    assert (compare_trend_models.FAMILY_LOGREG, "DEGENERATE") in skipped_families
    assert (compare_trend_models.FAMILY_RANDOM_FOREST, "DEGENERATE") in skipped_families
    # Каждая семья оценена на n_tickers=2 (два хороших тикера).
    for item in summary:
        assert item.n_tickers == 2
