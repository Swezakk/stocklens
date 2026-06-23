"""Оценка/обучение моделей волатильности по walk-forward (ml-spec §6, §7, §11, §12).

Поток: по каждому тикеру — загрузка котировок → сборка фич → walk-forward (QLIKE/RMSE для
baseline RW-RV, HAR-RV, GARCH), лог прогона в MLflow. Затем выбирается агрегатный победитель
(метод с лучшим средним QLIKE по тикерам, бьющий baseline; D6) и регистрируется в реестр под
алиасом ``champion``. **Baseline-гейт:** если ни один метод не бьёт baseline — регистрации
нет (лог + отказ, не молча). Промоушен ``champion``→``production`` — ручной (§12).

GARCH переносимого состояния не несёт (рефит на окне при инференсе), HAR — пулинговые
OLS-коэффициенты по всем тикерам. Запуск: ``python -m
stocklens_ml.training.train_volatility --tickers SBER GAZP --mlflow-uri http://localhost:5000``.
"""

import argparse
from dataclasses import dataclass

import mlflow
import pandas as pd
import structlog
from mlflow import MlflowClient
from mlflow.models.model import ModelInfo
from sqlalchemy.orm import Session

from stocklens_ml.config import MlSettings
from stocklens_ml.data import loader
from stocklens_ml.eval import walk_forward
from stocklens_ml.features import assemble
from stocklens_ml.models.har import HarRvModel
from stocklens_ml.registry import promote
from stocklens_ml.registry.pyfunc_volatility import (
    METHOD_HAR,
    SERVING_FEATURES,
    log_volatility_model,
)

_log = structlog.get_logger(__name__)

_EXPERIMENT = "volatility"
_BASELINE = "baseline_rw_rv"
#: Размер input_example для signature serving-артефакта (≥ _MIN_GARCH_OBS для валидации GARCH).
_INPUT_EXAMPLE_ROWS = 250
Metrics = dict[str, dict[str, float]]


@dataclass(frozen=True)
class WinnerSelection:
    """Агрегатный победитель волатильности: метод и его метрики vs baseline (для serving)."""

    method: str
    metrics: dict[str, float]


def build_ticker_frame(session: Session, ticker: str, settings: MlSettings) -> pd.DataFrame:
    """Собрать фрейм фич волатильности по тикеру (котировки/дивиденды/сплиты из БД)."""
    candles = loader.load_candles(session, ticker)
    if candles.empty:
        raise ValueError(f"Нет котировок по тикеру {ticker}")
    return assemble.build_volatility_frame(
        candles,
        loader.load_dividends(session, ticker),
        loader.load_splits(session, ticker),
        train_start=settings.train_start,
        horizon=settings.horizon_days,
    )


def evaluate_frame(frame: pd.DataFrame, settings: MlSettings, n_splits: int) -> Metrics:
    """Прогнать walk-forward по фрейму фич; вернуть QLIKE/RMSE по моделям."""
    return walk_forward.evaluate(
        frame, walk_forward.default_forecasters(), n_splits=n_splits, gap=settings.horizon_days
    )


def beats_baseline(metrics: Metrics, baseline: str = _BASELINE) -> dict[str, bool]:
    """Какие модели бьют baseline по QLIKE на одном тикере (меньше — лучше; D6)."""
    baseline_qlike = metrics[baseline]["qlike"]
    return {
        model: values["qlike"] < baseline_qlike
        for model, values in metrics.items()
        if model != baseline
    }


def _mean_metric(results: dict[str, Metrics], method: str, metric: str) -> float | None:
    """Среднее значение метрики метода по тикерам, где он присутствует (None — если нигде)."""
    values = [m[method][metric] for m in results.values() if method in m]
    return sum(values) / len(values) if values else None


def select_winner(results: dict[str, Metrics], baseline: str = _BASELINE) -> WinnerSelection | None:
    """Агрегатный победитель: метод с минимальным средним QLIKE по тикерам, бьющий baseline.

    Возвращает None (baseline-гейт), если ни один метод не бьёт baseline по среднему QLIKE.
    """
    baseline_qlike = _mean_metric(results, baseline, "qlike")
    if baseline_qlike is None:
        return None
    methods = {method for metrics in results.values() for method in metrics if method != baseline}
    best: WinnerSelection | None = None
    for method in sorted(methods):
        mean_qlike = _mean_metric(results, method, "qlike")
        if mean_qlike is None or mean_qlike >= baseline_qlike:
            continue
        if best is None or mean_qlike < best.metrics["qlike"]:
            mean_rmse = _mean_metric(results, method, "rmse")
            assert mean_rmse is not None  # метод присутствует → rmse считается по тем же тикерам
            best = WinnerSelection(
                method,
                {"qlike": mean_qlike, "qlike_baseline": baseline_qlike, "rmse": mean_rmse},
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
        baseline_qlike = metrics[_BASELINE]["qlike"]
        for model, values in metrics.items():
            mlflow.log_metric(f"qlike_{model}", values["qlike"])
            mlflow.log_metric(f"rmse_{model}", values["rmse"])
            if model != _BASELINE:
                mlflow.log_metric(f"qlike_gain_{model}", baseline_qlike - values["qlike"])


def _input_example(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Репрезентативный вход serving (контракт SERVING_FEATURES) для signature артефакта."""
    example = frames[0][SERVING_FEATURES].dropna()
    return example.tail(_INPUT_EXAMPLE_ROWS).reset_index(drop=True)


def _fit_har_coefficients(frames: list[pd.DataFrame]) -> tuple[list[float], float]:
    """Пулинговая OLS-оценка HAR по всем тикерам (единые коэффициенты в champion-артефакте)."""
    pooled = pd.concat(frames, ignore_index=True)
    coef, intercept = HarRvModel().fit(pooled, pooled["rv_target"]).coefficients()
    return coef.tolist(), intercept


def register_champion(
    settings: MlSettings,
    winner: WinnerSelection,
    frames: list[pd.DataFrame],
    client: MlflowClient | None = None,
) -> ModelInfo:
    """Залогировать serving-обёртку победителя, зарегистрировать версию и пометить champion."""
    client = client or MlflowClient()
    har_coef: list[float] | None = None
    har_intercept: float | None = None
    if winner.method == METHOD_HAR:
        har_coef, har_intercept = _fit_har_coefficients(frames)
    mlflow.set_experiment(_EXPERIMENT)
    with mlflow.start_run(run_name=f"champion-{winner.method}"):
        info = log_volatility_model(
            method=winner.method,
            metrics=winner.metrics,
            horizon=settings.horizon_days,
            input_example=_input_example(frames),
            har_coef=har_coef,
            har_intercept=har_intercept,
            registered_model_name=settings.volatility_model_name,
        )
    version = info.registered_model_version
    assert version is not None  # registered_model_name задан → версия гарантированно создана
    promote.mark_champion(client, settings.volatility_model_name, version)
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
        _log.info("ticker_evaluated", ticker=ticker, beats_baseline=beats_baseline(metrics))

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
    """CLI: оценить волатильность по тикерам, залогировать и зарегистрировать champion."""
    structlog.configure(processors=[structlog.processors.JSONRenderer()])
    parser = argparse.ArgumentParser(description="Оценка моделей волатильности (walk-forward)")
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
