"""Оценка/обучение моделей волатильности по walk-forward (ml-spec §6, §7, §12).

Для каждого тикера: загрузка котировок из БД → сборка фич → walk-forward (QLIKE/RMSE для
baseline RW-RV, HAR-RV, GARCH) → лог прогона в MLflow. Модель засчитывается, только если бьёт
baseline по среднему QLIKE (D6). Регистрация лучшей модели в реестр — отдельный шаг (нужен
MLflow-сервер); здесь — оценка и журналирование. Запуск: ``python -m
stocklens_ml.training.train_volatility --tickers SBER GAZP --mlflow-uri sqlite:///mlruns.db``.
"""

import argparse

import mlflow
import structlog
from sqlalchemy.orm import Session

from stocklens_ml.config import MlSettings
from stocklens_ml.data import loader
from stocklens_ml.eval import walk_forward
from stocklens_ml.features import assemble

_log = structlog.get_logger(__name__)

_EXPERIMENT = "volatility"
_BASELINE = "baseline_rw_rv"
Metrics = dict[str, dict[str, float]]


def evaluate_ticker(session: Session, ticker: str, settings: MlSettings, n_splits: int) -> Metrics:
    """Собрать фичи тикера и прогнать walk-forward; вернуть QLIKE/RMSE по моделям."""
    candles = loader.load_candles(session, ticker)
    if candles.empty:
        raise ValueError(f"Нет котировок по тикеру {ticker}")
    frame = assemble.build_volatility_frame(
        candles,
        loader.load_dividends(session, ticker),
        loader.load_splits(session, ticker),
        train_start=settings.train_start,
        horizon=settings.horizon_days,
    )
    return walk_forward.evaluate(
        frame, walk_forward.default_forecasters(), n_splits=n_splits, gap=settings.horizon_days
    )


def beats_baseline(metrics: Metrics, baseline: str = _BASELINE) -> dict[str, bool]:
    """Какие модели бьют baseline по среднему QLIKE (меньше — лучше; D6)."""
    baseline_qlike = metrics[baseline]["qlike"]
    return {
        model: values["qlike"] < baseline_qlike
        for model, values in metrics.items()
        if model != baseline
    }


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


def run(
    settings: MlSettings, tickers: list[str], n_splits: int, tracking_uri: str
) -> dict[str, Metrics]:
    """Прогнать оценку по тикерам, залогировать в MLflow, вернуть метрики по тикерам."""
    mlflow.set_tracking_uri(tracking_uri)
    session_factory = loader.make_session_factory(str(settings.database_url))
    results: dict[str, Metrics] = {}
    for ticker in tickers:
        with session_factory() as session:
            metrics = evaluate_ticker(session, ticker, settings, n_splits)
        log_run(ticker, metrics, settings, n_splits)
        results[ticker] = metrics
        _log.info("ticker_evaluated", ticker=ticker, beats_baseline=beats_baseline(metrics))
    return results


def main() -> None:
    """CLI: оценить волатильность по тикерам и залогировать в MLflow."""
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
