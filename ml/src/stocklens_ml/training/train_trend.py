"""Оценка/обучение модели тренда по walk-forward (ml-spec §5.4, §6.2, §11, §12).

Поток: по каждому тикеру — котировки → фрейм фич тренда → walk-forward (accuracy/F1/ROC-AUC
для CatBoost-классификатора и always-up baseline), лог прогона в MLflow. Затем выбирается
агрегатный победитель (метод с лучшим средним ROC-AUC по тикерам, строго бьющий baseline; D6)
и регистрируется в реестр под алиасом ``champion``.

**Гейт — по ROC-AUC, не по accuracy.** Always-up baseline даёт accuracy ≈ base rate (53–57 %)
и ROC-AUC = 0.5. Accuracy базовой ставки отвергла бы любую честную модель, поэтому гейт берёт
ROC-AUC (инвариант к балансу классов): модель засчитывается, только если её средний ROC-AUC
строго превышает 0.5. Если ни один метод не бьёт baseline — регистрации нет (лог + отказ).

Артефакт — **нативный** ``mlflow.catboost`` (а не pyfunc-обёртка волатильности): CatBoost
самодостаточен, переносимого ручного состояния не несёт. Промоушен ``champion``→``production``
ручной (§12). Запуск: ``python -m stocklens_ml.training.train_trend --tickers SBER GAZP``.
"""

import argparse
import math
from dataclasses import dataclass

import mlflow.catboost
import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo
from sqlalchemy.orm import Session

import mlflow
from mlflow import MlflowClient
from stocklens_ml.config import MlSettings
from stocklens_ml.data import loader
from stocklens_ml.eval import walk_forward
from stocklens_ml.features import assemble
from stocklens_ml.features.assemble import TREND_FEATURE_COLUMNS, TREND_TARGET_COLUMN
from stocklens_ml.models.baselines import always_up_forecast
from stocklens_ml.models.trend import TrendModel
from stocklens_ml.registry import promote

_log = structlog.get_logger(__name__)

_EXPERIMENT = "trend"
#: Метод-baseline тренда (always-up, P(up)=1) — точка отсчёта гейта.
_BASELINE = "baseline_always_up"
#: Имя CatBoost-метода тренда (единственный обучаемый кандидат).
_METHOD_CATBOOST = "trend_catboost"

#: Имена классификационных метрик (контракт с :mod:`eval.classification_metrics`).
_METRIC_ACCURACY = "accuracy"
_METRIC_F1 = "f1"
_METRIC_ROC_AUC = "roc_auc"
#: Метрика гейта: ROC-AUC (инвариант к балансу классов; см. модульный docstring).
_GATE_METRIC = _METRIC_ROC_AUC
#: ROC-AUC always-up baseline по определению равен 0.5 (нет ранжирования) — отсчёт для gain.
_BASELINE_ROC_AUC = 0.5
#: Размер input_example для signature serving-артефакта.
_INPUT_EXAMPLE_ROWS = 100

Metrics = dict[str, dict[str, float]]


@dataclass(frozen=True)
class WinnerSelection:
    """Агрегатный победитель тренда: метод и его средние метрики (для регистрации champion)."""

    method: str
    metrics: dict[str, float]


def build_ticker_frame(session: Session, ticker: str, settings: MlSettings) -> pd.DataFrame:
    """Собрать фрейм фич тренда по тикеру и отбросить строки без таргета (NaN-хвост forward).

    Последние ``horizon`` строк build_trend_frame не имеют будущего close → NaN-таргет; их
    нельзя подавать ни в обучение, ни в eval_set (CatBoost падает на NaN-метках). Warm-up-NaN
    в фичах оставляем — CatBoost их обрабатывает; критичен только NaN таргета.
    """
    candles = loader.load_candles(session, ticker)
    if candles.empty:
        raise ValueError(f"Нет котировок по тикеру {ticker}")
    frame = assemble.build_trend_frame(
        candles,
        loader.load_dividends(session, ticker),
        loader.load_splits(session, ticker),
        train_start=settings.train_start,
        horizon=settings.horizon_days,
    )
    return _drop_unlabelled(frame)


def _drop_unlabelled(frame: pd.DataFrame) -> pd.DataFrame:
    """Отбросить строки с NaN-таргетом (forward-хвост) — метки обязательны для fit/eval_set."""
    return frame.dropna(subset=[TREND_TARGET_COLUMN]).reset_index(drop=True)


def _validation_size(train_len: int, horizon: int) -> int:
    """Размер early-stop валидационного хвоста: max(2·H, 15 % train-окна) — документированная V."""
    return max(2 * horizon, math.ceil(0.15 * train_len))


def _fit_with_purged_validation(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    train_idx: npt.NDArray[np.intp],
    horizon: int,
    iterations: int | None = None,
) -> TrendModel:
    """Обучить TrendModel с early-stop валидационным хвостом и H-дневным purge перед ним.

    Правило карвинга (антиутечка — именованный риск проекта): ``val_idx = train_idx[-V:]``
    (хвост train), ``fit_idx = train_idx[:-(V+horizon)]``. ``horizon`` строк между fit и val
    выбрасываются: их forward-таргет перекрывает val-окно, иначе число деревьев тюнилось бы на
    метках, смежных с валидацией. Если fit_idx пуст (train слишком короткий) — фит без eval_set.
    ``iterations`` (опционально) переопределяет число деревьев — тесты задают крошечное число.
    """
    validation_size = _validation_size(len(train_idx), horizon)
    val_idx = train_idx[-validation_size:]
    fit_idx = train_idx[: -(validation_size + horizon)]
    model = TrendModel() if iterations is None else TrendModel(iterations=iterations)
    if len(fit_idx) == 0:
        return model.fit(x.iloc[train_idx], y.iloc[train_idx])
    return model.fit(
        x.iloc[fit_idx],
        y.iloc[fit_idx],
        eval_set=(x.iloc[val_idx], y.iloc[val_idx]),
    )


def _catboost_forecaster(horizon: int) -> walk_forward.Forecaster:
    """Форкастер тренда: фит CatBoost на purged train-окне → P(up) на test-строках."""

    def forecaster(
        frame: pd.DataFrame,
        train_idx: npt.NDArray[np.intp],
        test_idx: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float64]:
        x = frame[TREND_FEATURE_COLUMNS]
        y = frame[TREND_TARGET_COLUMN]
        model = _fit_with_purged_validation(x, y, train_idx=train_idx, horizon=horizon)
        return model.predict_proba(x.iloc[test_idx])

    return forecaster


def _always_up_forecaster(
    frame: pd.DataFrame,
    train_idx: npt.NDArray[np.intp],
    test_idx: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    """Baseline always-up: P(up)=1 на каждой test-строке (ROC-AUC = 0.5)."""
    forecast = always_up_forecast(frame.iloc[test_idx].index).to_numpy(dtype=float)
    return np.asarray(forecast, dtype=np.float64)


def _forecasters(horizon: int) -> dict[str, walk_forward.Forecaster]:
    """Набор форкастеров тренда: обучаемый CatBoost + always-up baseline."""
    return {
        _METHOD_CATBOOST: _catboost_forecaster(horizon),
        _BASELINE: _always_up_forecaster,
    }


def evaluate_frame(frame: pd.DataFrame, settings: MlSettings, n_splits: int) -> Metrics:
    """Прогнать walk-forward по фрейму фич тренда; вернуть accuracy/F1/ROC-AUC по моделям."""
    return walk_forward.evaluate_trend(
        frame, _forecasters(settings.horizon_days), n_splits=n_splits, gap=settings.horizon_days
    )


def _mean_metric(results: dict[str, Metrics], method: str, metric: str) -> float | None:
    """Среднее значение метрики метода по тикерам, где он присутствует (None — если нигде)."""
    values = [m[method][metric] for m in results.values() if method in m]
    return sum(values) / len(values) if values else None


def select_winner(results: dict[str, Metrics], baseline: str = _BASELINE) -> WinnerSelection | None:
    """Агрегатный победитель: метод с максимальным средним ROC-AUC, строго бьющий baseline.

    Возвращает None (baseline-гейт), если ни один метод не превышает средний ROC-AUC baseline
    (always-up = 0.5). Сравнение строгое: ничья на baseline отвергается.
    """
    baseline_roc = _mean_metric(results, baseline, _GATE_METRIC)
    if baseline_roc is None:
        return None
    methods = {method for metrics in results.values() for method in metrics if method != baseline}
    best: WinnerSelection | None = None
    for method in sorted(methods):
        mean_roc = _mean_metric(results, method, _GATE_METRIC)
        if mean_roc is None or mean_roc <= baseline_roc:
            continue
        if best is None or mean_roc > best.metrics[_GATE_METRIC]:
            mean_accuracy = _mean_metric(results, method, _METRIC_ACCURACY)
            mean_f1 = _mean_metric(results, method, _METRIC_F1)
            assert mean_accuracy is not None  # метод присутствует → accuracy на тех же тикерах
            assert mean_f1 is not None  # и F1 на тех же тикерах
            best = WinnerSelection(
                method,
                {
                    _METRIC_ROC_AUC: mean_roc,
                    "roc_auc_baseline": baseline_roc,
                    _METRIC_ACCURACY: mean_accuracy,
                    _METRIC_F1: mean_f1,
                },
            )
    return best


def log_run(ticker: str, metrics: Metrics, settings: MlSettings, n_splits: int) -> None:
    """Залогировать прогон тикера в MLflow: параметры, метрики, превышение над baseline."""
    mlflow.set_experiment(_EXPERIMENT)
    with mlflow.start_run(run_name=ticker):
        mlflow.log_params(
            {
                "ticker": ticker,
                "n_splits": n_splits,
                "gap": settings.horizon_days,
                "horizon_days": settings.horizon_days,
                "train_start": str(settings.train_start),
            }
        )
        for model, values in metrics.items():
            mlflow.log_metric(f"{_METRIC_ACCURACY}_{model}", values[_METRIC_ACCURACY])
            mlflow.log_metric(f"{_METRIC_F1}_{model}", values[_METRIC_F1])
            mlflow.log_metric(f"{_METRIC_ROC_AUC}_{model}", values[_METRIC_ROC_AUC])
            if model != _BASELINE:
                gain = values[_METRIC_ROC_AUC] - _BASELINE_ROC_AUC
                mlflow.log_metric(f"{_METRIC_ROC_AUC}_gain_{model}", gain)


def _input_example(frame: pd.DataFrame) -> pd.DataFrame:
    """Репрезентативный вход serving (контракт TREND_FEATURE_COLUMNS) для signature артефакта."""
    return frame[TREND_FEATURE_COLUMNS].dropna().tail(_INPUT_EXAMPLE_ROWS).reset_index(drop=True)


def register_champion(
    settings: MlSettings,
    winner: WinnerSelection,
    frames: list[pd.DataFrame],
    client: MlflowClient | None = None,
    iterations: int | None = None,
) -> ModelInfo:
    """Обучить финальную модель на полном окне и зарегистрировать версию + алиас champion.

    Финальный фит — на пуле всех тикеров (как HAR-коэффициенты волатильности) с тем же
    early-stop карвингом, что и в walk-forward. Логируется НАТИВНО через ``mlflow.catboost``
    (артефакт самодостаточен). ``iterations`` опционален — тесты задают крошечное число.
    """
    client = client or MlflowClient()
    pooled = _drop_unlabelled(pd.concat(frames, ignore_index=True))
    x = pooled[TREND_FEATURE_COLUMNS]
    y = pooled[TREND_TARGET_COLUMN]
    train_idx = np.arange(len(pooled), dtype=np.intp)
    model = _fit_with_purged_validation(
        x, y, train_idx=train_idx, horizon=settings.horizon_days, iterations=iterations
    )
    example = _input_example(pooled)
    signature = infer_signature(example, model.predict_proba(example))
    mlflow.set_experiment(_EXPERIMENT)
    with mlflow.start_run(run_name=f"champion-{winner.method}"):
        info: ModelInfo = mlflow.catboost.log_model(
            model._model,
            name="model",
            registered_model_name=settings.trend_model_name,
            signature=signature,
            input_example=example,
        )
    version = info.registered_model_version
    assert version is not None  # registered_model_name задан → версия гарантированно создана
    promote.mark_champion(client, settings.trend_model_name, version)
    return info


def run(
    settings: MlSettings, tickers: list[str], n_splits: int, tracking_uri: str
) -> dict[str, Metrics]:
    """Оценить тикеры, залогировать прогоны, зарегистрировать champion (если бьёт baseline)."""
    mlflow.set_tracking_uri(tracking_uri)
    session_factory = loader.make_session_factory(str(settings.database_url))
    results: dict[str, Metrics] = {}
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        with session_factory() as session:
            frame = build_ticker_frame(session, ticker, settings)
        metrics = evaluate_frame(frame, settings, n_splits)
        log_run(ticker, metrics, settings, n_splits)
        results[ticker] = metrics
        frames.append(frame)
        _log.info("ticker_evaluated", ticker=ticker, roc_auc=metrics[_METHOD_CATBOOST]["roc_auc"])

    winner = select_winner(results)
    if winner is None:
        _log.warning("no_model_beats_baseline", tickers=tickers)
        return results
    info = register_champion(settings, winner, frames)
    _log.info(
        "champion_registered",
        method=winner.method,
        version=str(info.registered_model_version),
        metrics=winner.metrics,
    )
    return results


def main() -> None:
    """CLI: оценить тренд по тикерам, залогировать и зарегистрировать champion."""
    structlog.configure(processors=[structlog.processors.JSONRenderer()])
    parser = argparse.ArgumentParser(description="Оценка модели тренда (walk-forward)")
    parser.add_argument("--tickers", nargs="+", required=True, help="Тикеры MOEX (SBER GAZP ...)")
    parser.add_argument("--n-splits", type=int, default=5, help="Число фолдов walk-forward")
    parser.add_argument(
        "--mlflow-uri",
        default="sqlite:///mlruns.db",
        help="MLflow tracking URI (по умолчанию локальный sqlite; file-store закрыт в MLflow 3.x)",
    )
    args = parser.parse_args()

    settings = MlSettings.model_validate({})  # значения берутся из окружения
    results = run(settings, args.tickers, args.n_splits, args.mlflow_uri)
    for ticker, metrics in results.items():
        _log.info("result", ticker=ticker, metrics=metrics)


if __name__ == "__main__":
    main()
