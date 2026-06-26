"""Подбор гиперпараметров модели тренда на walk-forward (ml-spec §5.4).

Стартовые гиперпараметры спеки (600/4/0.03/6.0) дали средний walk-forward ROC-AUC ≈ 0.49 по
тикерам — модель НЕ обошла always-up baseline (0.5). Спека §5.4 санкционирует финализацию
гиперпараметров на walk-forward («параметры — стартовые; финальные подбираются на walk-forward:
консервативно — малая глубина, сильная регуляризация, данных мало, риск переобучения высок»).
Этот модуль выполняет финализацию — **консервативно**, минимизируя bias отбора.

**Только поиск, не регистрация.** Каждый конфиг логируется отдельным прогоном в эксперимент
``trend-tuning`` (параметры + средние метрики), модель в реестр НЕ пишется. Выбранный конфиг
затем передаётся в ``train_trend`` (или зашивается в дефолты ``TrendHyperparams``) для реальной
регистрации champion. Артефакты sweep — лишь данные для решения, а не serving-модель.

**Дисциплина против bias отбора.** Грид мал и узок намеренно (см. ``_hyperparameter_grid``): чем
больше конфигов перебираем на тонком сигнале, тем выше шанс, что лучший «победил» случайно. Все
конфиги — мелкие (depth ≤ 4) и сильно регуляризованные (l2 ≥ 6.0); глубокие/слабо
регуляризованные варианты исключены осознанно — на этом сигнале они переобучатся, и их выбор был
бы подгонкой под walk-forward, а не честной финализацией.

Запуск: ``python -m stocklens_ml.training.tune_trend --tickers SBER GAZP``.
"""

import argparse
from dataclasses import dataclass

import pandas as pd
import structlog
from catboost import CatBoostError

import mlflow
from stocklens_ml.config import MlSettings
from stocklens_ml.data import loader
from stocklens_ml.eval import walk_forward
from stocklens_ml.models.trend import TrendHyperparams
from stocklens_ml.training import train_trend

_log = structlog.get_logger(__name__)

#: Эксперимент MLflow для прогонов sweep (отделён от боевого ``trend`` train_trend).
_EXPERIMENT = "trend-tuning"

#: ROC-AUC always-up baseline по определению = 0.5 (нет ранжирования) — точка отсчёта гейта.
_BASELINE_ROC_AUC = train_trend._BASELINE_ROC_AUC
#: Имя CatBoost-метода тренда (общий контракт с train_trend — без дублирования).
_METHOD_CATBOOST = train_trend._METHOD_CATBOOST

#: Имена метрик (контракт с :mod:`eval.classification_metrics`).
_METRIC_ACCURACY = "accuracy"
_METRIC_F1 = "f1"
_METRIC_ROC_AUC = "roc_auc"

#: Консервативная сетка финализации (ml-spec §5.4). Перебираемые оси — глубина и L2; шаг
#: обучения и число деревьев фиксированы (деревья early-stop'ятся валидационным хвостом).
_GRID_DEPTHS: tuple[int, ...] = (2, 3, 4)
_GRID_L2_LEAF_REGS: tuple[float, ...] = (6.0, 12.0)
_GRID_LEARNING_RATE = 0.03
#: Деревьев берём с запасом — реальное число подбирает early-stop, поэтому верхнюю границу
#: можно держать высокой без риска переобучения по числу итераций.
_GRID_ITERATIONS = 800


@dataclass(frozen=True)
class ConfigResult:
    """Агрегат метрик одного конфига по тикерам: средние ROC-AUC/accuracy/F1 и число тикеров."""

    hyperparams: TrendHyperparams
    mean_roc_auc: float
    mean_accuracy: float
    mean_f1: float
    n_tickers: int


@dataclass(frozen=True)
class SweepResult:
    """Итог sweep: отсортированные по ROC-AUC конфиги, лучший и факт превышения baseline."""

    ranked: list[ConfigResult]
    best: ConfigResult | None
    best_mean_roc_auc: float
    beats_baseline: bool


def _hyperparameter_grid() -> list[TrendHyperparams]:
    """Малая консервативная сетка для финализации тренда (ml-spec §5.4): depth × L2.

    6 конфигов: depth ∈ {2,3,4} × l2_leaf_reg ∈ {6.0, 12.0}, learning_rate = 0.03 фиксирован,
    iterations = 800 (реально подбирается early-stop). Грид намеренно мал и узок — все варианты
    мелкие и сильно регуляризованные. Глубокие (depth > 4) и слабо регуляризованные (l2 < 6.0)
    конфиги исключены осознанно: на тонком сигнале тренда они переобучатся, а их выбор был бы
    подгонкой под walk-forward (bias отбора), а не честной финализацией.
    """
    return [
        TrendHyperparams(
            iterations=_GRID_ITERATIONS,
            depth=depth,
            learning_rate=_GRID_LEARNING_RATE,
            l2_leaf_reg=l2_leaf_reg,
        )
        for depth in _GRID_DEPTHS
        for l2_leaf_reg in _GRID_L2_LEAF_REGS
    ]


def _evaluate_config(
    config: TrendHyperparams,
    frames: dict[str, pd.DataFrame],
    settings: MlSettings,
    n_splits: int,
) -> ConfigResult | None:
    """Прогнать конфиг по всем тикерам walk-forward, вернуть средние метрики (None — если пусто).

    Per-ticker изоляция: тикер, на котором ``evaluate_trend`` бросает, пропускается с логом
    ``ticker_skipped`` — sweep не падает из-за одного плохого тикера. Два штатных источника:
    одноклассовая объединённая test-выборка → ``ValueError`` из classification_metrics.roc_auc;
    одноклассовый train-фолд → ``CatBoostError`` (CatBoost.fit отвергает один класс — она наследует
    Exception напрямую, не ValueError, поэтому ловится явно). Если пропущены все тикеры, конфиг
    исключается (None): среднее по пустому набору не определено.
    """
    forecasters = train_trend._forecasters(settings.horizon_days, config)
    roc_aucs: list[float] = []
    accuracies: list[float] = []
    f1s: list[float] = []
    for ticker, frame in frames.items():
        try:
            metrics = walk_forward.evaluate_trend(
                frame, forecasters, n_splits=n_splits, gap=settings.horizon_days
            )
        except (ValueError, ArithmeticError, CatBoostError) as exc:
            _log.warning("ticker_skipped", ticker=ticker, reason=str(exc))
            continue
        catboost = metrics[_METHOD_CATBOOST]
        roc_aucs.append(catboost[_METRIC_ROC_AUC])
        accuracies.append(catboost[_METRIC_ACCURACY])
        f1s.append(catboost[_METRIC_F1])
    if not roc_aucs:
        _log.warning("config_skipped", reason="все тикеры пропущены", config=_config_params(config))
        return None
    return ConfigResult(
        hyperparams=config,
        mean_roc_auc=sum(roc_aucs) / len(roc_aucs),
        mean_accuracy=sum(accuracies) / len(accuracies),
        mean_f1=sum(f1s) / len(f1s),
        n_tickers=len(roc_aucs),
    )


def _config_params(config: TrendHyperparams) -> dict[str, float | int]:
    """Параметры конфига для лога MLflow/structlog (имена полей TrendHyperparams)."""
    return {
        "iterations": config.iterations,
        "depth": config.depth,
        "learning_rate": config.learning_rate,
        "l2_leaf_reg": config.l2_leaf_reg,
    }


def _log_config_run(result: ConfigResult, n_splits: int) -> None:
    """Залогировать конфиг отдельным прогоном в эксперимент ``trend-tuning`` (без регистрации)."""
    mlflow.set_experiment(_EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_params({**_config_params(result.hyperparams), "n_splits": n_splits})
        mlflow.log_metric("mean_roc_auc", result.mean_roc_auc)
        mlflow.log_metric("mean_accuracy", result.mean_accuracy)
        mlflow.log_metric("mean_f1", result.mean_f1)
        mlflow.log_metric("n_tickers", result.n_tickers)


def evaluate_grid(
    frames: dict[str, pd.DataFrame],
    settings: MlSettings,
    n_splits: int,
    grid: list[TrendHyperparams] | None = None,
) -> SweepResult:
    """Прогнать sweep по гриду на уже загруженных фреймах; залогировать каждый конфиг в MLflow.

    Фреймы передаются загруженными (DB-чтение — медленная часть, делается один раз вызывающим),
    sweep лишь переоценивает их под каждый конфиг. На каждый конфиг — отдельный прогон в
    ``trend-tuning`` (параметры + средние метрики), регистрации модели НЕТ. По завершении —
    отсортированный (убыв. ROC-AUC) лог-вывод, baseline (0.5) и событие ``tuning_complete`` с
    лучшим конфигом, его средним ROC-AUC и булевым ``beats_baseline`` (строгое превышение 0.5).
    """
    grid = grid if grid is not None else _hyperparameter_grid()
    results: list[ConfigResult] = []
    for config in grid:
        result = _evaluate_config(config, frames, settings, n_splits)
        if result is None:
            continue
        _log_config_run(result, n_splits)
        results.append(result)
        _log.info(
            "config_evaluated",
            **_config_params(config),
            mean_roc_auc=result.mean_roc_auc,
            n_tickers=result.n_tickers,
        )

    ranked = sorted(results, key=lambda r: r.mean_roc_auc, reverse=True)
    best = ranked[0] if ranked else None
    best_mean_roc_auc = best.mean_roc_auc if best is not None else _BASELINE_ROC_AUC
    beats_baseline = best_mean_roc_auc > _BASELINE_ROC_AUC

    for rank, result in enumerate(ranked, start=1):
        _log.info(
            "config_ranked",
            rank=rank,
            **_config_params(result.hyperparams),
            mean_roc_auc=result.mean_roc_auc,
        )
    _log.info(
        "tuning_complete",
        best=_config_params(best.hyperparams) if best is not None else None,
        mean_roc_auc=best_mean_roc_auc,
        baseline_roc_auc=_BASELINE_ROC_AUC,
        beats_baseline=beats_baseline,
    )
    return SweepResult(
        ranked=ranked,
        best=best,
        best_mean_roc_auc=best_mean_roc_auc,
        beats_baseline=beats_baseline,
    )


def run(settings: MlSettings, tickers: list[str], n_splits: int, tracking_uri: str) -> SweepResult:
    """Загрузить фреймы тикеров один раз и прогнать sweep гиперпараметров (только поиск)."""
    mlflow.set_tracking_uri(tracking_uri)
    session_factory = loader.make_session_factory(str(settings.database_url))
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        with session_factory() as session:
            frames[ticker] = train_trend.build_ticker_frame(session, ticker, settings)
        _log.info("ticker_loaded", ticker=ticker, rows=len(frames[ticker]))
    return evaluate_grid(frames, settings, n_splits)


def main() -> None:
    """CLI: подобрать гиперпараметры тренда на walk-forward (логирует в ``trend-tuning``)."""
    structlog.configure(processors=[structlog.processors.JSONRenderer()])
    parser = argparse.ArgumentParser(description="Подбор гиперпараметров тренда (walk-forward)")
    parser.add_argument("--tickers", nargs="+", required=True, help="Тикеры MOEX (SBER GAZP ...)")
    parser.add_argument("--n-splits", type=int, default=5, help="Число фолдов walk-forward")
    parser.add_argument(
        "--mlflow-uri",
        default="sqlite:///mlruns.db",
        help="MLflow tracking URI (по умолчанию локальный sqlite; file-store закрыт в MLflow 3.x)",
    )
    args = parser.parse_args()

    settings = MlSettings.model_validate({})  # значения берутся из окружения
    run(settings, args.tickers, args.n_splits, args.mlflow_uri)


if __name__ == "__main__":
    main()
