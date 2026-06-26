"""Сравнение классов моделей тренда — sanity check сигнала (ml-spec §5.4, §6.2, §8.5).

CatBoost-тренд на стартовых гиперпараметрах дал средний walk-forward ROC-AUC ≈ 0.49 — модель
не обошла always-up baseline (0.5). Чтобы доказать, что слаб **сигнал**, а не конкретная
модель, этот скрипт прогоняет ДРУГИЕ классы моделей (логистическую регрессию и случайный лес)
на ТЕХ ЖЕ фичах и том же walk-forward. Если ни одна семья не бьёт baseline — это подтверждает,
что предсказуемого направленного сигнала на горизонте просто нет (не-цель спеки — точечный
прогноз цены, §2).

**Одноразовый подтверждающий эксперимент.** НИКАКОГО подбора гиперпараметров альтернатив
(фиксированные разумные конфиги), НИКАКОЙ регистрации модели. Каждая семья логируется отдельным
прогоном в эксперимент ``trend-model-comparison`` (параметры конфига + средние метрики);
в реестр ничего не пишется. CatBoost включён как REFERENCE — internal-consistency проверка, что
он на чистом фрейме воспроизводит ≈0.49.

**Чистый фрейм.** В отличие от train_trend (CatBoost нативно терпит NaN в фичах), logreg и
random forest на NaN падают, поэтому warm-up строки с NaN в фичах отбрасываются
(:func:`_drop_unfeatured`). Это безопасно с точки зрения утечки — отбор по доступности фич,
а не по таргету.

**Антиутечка logreg.** ``StandardScaler`` фитится на TRAIN-строках фолда
(:func:`_fit_train_scaler`), не на всём фрейме — утечка через препроцессинг главный известный
риск проекта (§8.5). Деревьям случайного леса масштабирование не нужно.

Запуск: ``python -m stocklens_ml.training.compare_trend_models --tickers SBER GAZP``.
"""

import argparse
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog
from catboost import CatBoostError
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import mlflow
from stocklens_ml.config import MlSettings
from stocklens_ml.data import loader
from stocklens_ml.eval import walk_forward
from stocklens_ml.features.assemble import TREND_FEATURE_COLUMNS, TREND_TARGET_COLUMN
from stocklens_ml.training import train_trend

_log = structlog.get_logger(__name__)

#: Эксперимент MLflow для сравнения семей (отделён от боевого ``trend`` и от ``trend-tuning``).
_EXPERIMENT = "trend-model-comparison"

#: ROC-AUC always-up baseline = 0.5 (контракт с train_trend — без дублирования магической 0.5).
_BASELINE_ROC_AUC = train_trend._BASELINE_ROC_AUC
#: Имя метода-baseline в наборе форкастеров (always-up, P(up)=1) — точка отсчёта.
_BASELINE = train_trend._BASELINE

#: Имена метрик (контракт с :mod:`eval.classification_metrics`).
_METRIC_ACCURACY = "accuracy"
_METRIC_F1 = "f1"
_METRIC_ROC_AUC = "roc_auc"

#: Имена сравниваемых семей моделей (ключи форкастеров и метки в логах/MLflow).
FAMILY_CATBOOST = "catboost"
FAMILY_LOGREG = "logreg"
FAMILY_RANDOM_FOREST = "random_forest"

#: Фиксированный конфиг логистической регрессии (БЕЗ тюнинга — разумные значения по умолчанию).
#: L2-регуляризация задаётся через ``l1_ratio=0.0``: параметр ``penalty`` в sklearn 1.8 объявлен
#: устаревшим и удаляется в 1.10 (`l1_ratio=0` — точный эквивалент `penalty="l2"`, коэффициенты
#: совпадают), поэтому используем актуальный API, а не deprecated-аргумент.
_LOGREG_C = 1.0
_LOGREG_L1_RATIO = 0.0
_LOGREG_MAX_ITER = 1000
_LOGREG_RANDOM_STATE = 42

#: Фиксированный конфиг случайного леса (БЕЗ тюнинга — мелкие деревья, балансировка классов).
_RF_N_ESTIMATORS = 200
_RF_MAX_DEPTH = 4
_RF_RANDOM_STATE = 42
_RF_CLASS_WEIGHT = "balanced"

#: Минимум классов для обучения классификатора (RF без guard'а молча фитится на одном классе).
_MIN_CLASSES = 2


@dataclass(frozen=True)
class FamilyResult:
    """Агрегат метрик одной семьи по тикерам: средние ROC-AUC/accuracy/F1 и число тикеров."""

    family: str
    mean_roc_auc: float
    mean_accuracy: float
    mean_f1: float
    n_tickers: int

    @property
    def beats_baseline(self) -> bool:
        """Строгое превышение always-up baseline (0.5) по среднему ROC-AUC."""
        return self.mean_roc_auc > _BASELINE_ROC_AUC


def _drop_unfeatured(frame: pd.DataFrame) -> pd.DataFrame:
    """Отбросить строки с NaN хотя бы в одной фиче тренда (warm-up окон).

    CatBoost терпит NaN нативно, но logreg/random forest — нет. Отбор по доступности фич
    (не по таргету) безопасен с точки зрения утечки. Индекс сбрасывается: форкастеры берут
    ``X.iloc[train_idx]`` с позиционными индексами из ``TimeSeriesSplit`` — без reset дыры от
    выброшенных строк рассогласовали бы позиционную адресацию.
    """
    return frame.dropna(subset=TREND_FEATURE_COLUMNS).reset_index(drop=True)


def _fit_train_scaler(x: pd.DataFrame, train_idx: npt.NDArray[np.intp]) -> StandardScaler:
    """Фитнуть ``StandardScaler`` на TRAIN-строках фолда (антиутечка препроцессинга, §8.5).

    Скейлер видит только ``x.iloc[train_idx]`` — никогда весь фрейм. Фит на полном фрейме
    протёк бы статистикой test-окна в нормализацию train — главный известный риск проекта.
    """
    return StandardScaler().fit(x.iloc[train_idx])


def _logreg_forecaster(
    frame: pd.DataFrame,
    train_idx: npt.NDArray[np.intp],
    test_idx: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    """Логистическая регрессия: per-fold скейлер на train → P(up) на test (фикс. конфиг)."""
    x = frame[TREND_FEATURE_COLUMNS]
    y = frame[TREND_TARGET_COLUMN]
    scaler = _fit_train_scaler(x, train_idx)
    x_train = scaler.transform(x.iloc[train_idx])
    x_test = scaler.transform(x.iloc[test_idx])
    model = LogisticRegression(
        C=_LOGREG_C,
        l1_ratio=_LOGREG_L1_RATIO,
        max_iter=_LOGREG_MAX_ITER,
        random_state=_LOGREG_RANDOM_STATE,
    )
    model.fit(x_train, y.iloc[train_idx])  # одноклассовый train → ValueError (ловит изоляция)
    proba = model.predict_proba(x_test)[:, 1]
    return np.asarray(proba, dtype=np.float64)


def _random_forest_forecaster(
    n_estimators: int = _RF_N_ESTIMATORS,
) -> walk_forward.Forecaster:
    """Фабрика форкастера случайного леса (фикс. конфиг); ``n_estimators`` параметризуется тестами.

    Деревьям масштабирование не нужно — фит прямо на ``X.iloc[train_idx]``. RF, в отличие от
    logreg, на одноклассовом train НЕ бросает: ``classes_`` длиной 1, ``predict_proba`` даёт
    форму (n, 1) → ``[:, 1]`` упал бы IndexError мимо изоляции. Поэтому явный guard поднимает
    ValueError — он попадает в перехват изоляции, и тикер штатно пропускается всеми семьями.
    """

    def forecaster(
        frame: pd.DataFrame,
        train_idx: npt.NDArray[np.intp],
        test_idx: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float64]:
        x = frame[TREND_FEATURE_COLUMNS]
        y = frame[TREND_TARGET_COLUMN]
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=_RF_MAX_DEPTH,
            random_state=_RF_RANDOM_STATE,
            class_weight=_RF_CLASS_WEIGHT,
        )
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        if len(model.classes_) < _MIN_CLASSES:
            raise ValueError("random_forest: в train-фолде один класс — P(up) не определён")
        proba = model.predict_proba(x.iloc[test_idx])[:, 1]
        return np.asarray(proba, dtype=np.float64)

    return forecaster


def build_forecaster(
    family: str, *, horizon: int, rf_n_estimators: int = _RF_N_ESTIMATORS
) -> walk_forward.Forecaster:
    """Форкастер семьи ``family`` с фиксированным конфигом (без тюнинга альтернатив).

    CatBoost (REFERENCE) переиспользует ``train_trend._catboost_forecaster`` со стартовыми
    гиперпараметрами — internal-consistency проверка на чистом фрейме. logreg и random forest —
    sklearn с фиксированными разумными конфигами.
    """
    if family == FAMILY_CATBOOST:
        return train_trend._catboost_forecaster(horizon)
    if family == FAMILY_LOGREG:
        return _logreg_forecaster
    if family == FAMILY_RANDOM_FOREST:
        return _random_forest_forecaster(rf_n_estimators)
    raise ValueError(f"Неизвестная семья моделей: {family}")


def _family_config(family: str, rf_n_estimators: int) -> dict[str, str | int | float]:
    """Параметры фиксированного конфига семьи для лога MLflow/structlog."""
    if family == FAMILY_CATBOOST:
        hp = train_trend._DEFAULT_HYPERPARAMS
        return {
            "iterations": hp.iterations,
            "depth": hp.depth,
            "learning_rate": hp.learning_rate,
            "l2_leaf_reg": hp.l2_leaf_reg,
        }
    if family == FAMILY_LOGREG:
        return {
            "C": _LOGREG_C,
            "l1_ratio": _LOGREG_L1_RATIO,
            "max_iter": _LOGREG_MAX_ITER,
            "random_state": _LOGREG_RANDOM_STATE,
        }
    return {
        "n_estimators": rf_n_estimators,
        "max_depth": _RF_MAX_DEPTH,
        "random_state": _RF_RANDOM_STATE,
        "class_weight": _RF_CLASS_WEIGHT,
    }


def _evaluate_family(
    family: str,
    frames: dict[str, pd.DataFrame],
    settings: MlSettings,
    n_splits: int,
    rf_n_estimators: int,
) -> FamilyResult | None:
    """Прогнать семью по всем тикерам walk-forward, вернуть средние метрики (None — если пусто).

    Per-ticker изоляция (зеркало ``tune_trend._evaluate_config``): тикер, на котором семья
    бросает, пропускается с логом ``ticker_skipped``. Три штатных источника: одноклассовая
    объединённая test-выборка → ``ValueError`` из roc_auc; одноклассовый train-фолд → у logreg
    ``ValueError`` (sklearn), у CatBoost ``CatBoostError``, у random forest ``ValueError`` из
    явного guard'а. Если пропущены все тикеры — семья исключается (None).
    """
    forecaster = build_forecaster(
        family, horizon=settings.horizon_days, rf_n_estimators=rf_n_estimators
    )
    forecasters: dict[str, walk_forward.Forecaster] = {
        family: forecaster,
        _BASELINE: train_trend._always_up_forecaster,
    }
    roc_aucs: list[float] = []
    accuracies: list[float] = []
    f1s: list[float] = []
    for ticker, frame in frames.items():
        try:
            metrics = walk_forward.evaluate_trend(
                frame, forecasters, n_splits=n_splits, gap=settings.horizon_days
            )
        except (ValueError, ArithmeticError, CatBoostError) as exc:
            _log.warning("ticker_skipped", family=family, ticker=ticker, reason=str(exc))
            continue
        family_metrics = metrics[family]
        roc_aucs.append(family_metrics[_METRIC_ROC_AUC])
        accuracies.append(family_metrics[_METRIC_ACCURACY])
        f1s.append(family_metrics[_METRIC_F1])
    if not roc_aucs:
        _log.warning("family_skipped", family=family, reason="все тикеры пропущены")
        return None
    return FamilyResult(
        family=family,
        mean_roc_auc=sum(roc_aucs) / len(roc_aucs),
        mean_accuracy=sum(accuracies) / len(accuracies),
        mean_f1=sum(f1s) / len(f1s),
        n_tickers=len(roc_aucs),
    )


def _log_family_run(result: FamilyResult, n_splits: int, rf_n_estimators: int) -> None:
    """Залогировать семью отдельным прогоном в ``trend-model-comparison`` (БЕЗ регистрации)."""
    mlflow.set_experiment(_EXPERIMENT)
    with mlflow.start_run(run_name=result.family):
        mlflow.log_params(
            {
                "model_family": result.family,
                "n_splits": n_splits,
                **_family_config(result.family, rf_n_estimators),
            }
        )
        mlflow.log_metric("mean_roc_auc", result.mean_roc_auc)
        mlflow.log_metric("mean_accuracy", result.mean_accuracy)
        mlflow.log_metric("mean_f1", result.mean_f1)
        mlflow.log_metric("n_tickers", result.n_tickers)


def compare_families(
    frames: dict[str, pd.DataFrame],
    settings: MlSettings,
    n_splits: int,
    rf_n_estimators: int = _RF_N_ESTIMATORS,
) -> list[FamilyResult]:
    """Прогнать три семьи (catboost/logreg/random_forest) на уже загруженных чистых фреймах.

    Фреймы передаются загруженными (DB-чтение — медленная часть, делается один раз вызывающим).
    На каждую семью — отдельный прогон в ``trend-model-comparison`` (конфиг + средние метрики),
    регистрации модели НЕТ. По завершении — отсортированный (убыв. ROC-AUC) лог-вывод, baseline
    (0.5) и событие ``comparison_complete`` со списком всех семей и их превышением baseline.
    """
    results: list[FamilyResult] = []
    for family in (FAMILY_CATBOOST, FAMILY_LOGREG, FAMILY_RANDOM_FOREST):
        result = _evaluate_family(family, frames, settings, n_splits, rf_n_estimators)
        if result is None:
            continue
        _log_family_run(result, n_splits, rf_n_estimators)
        results.append(result)
        _log.info(
            "family_evaluated",
            family=family,
            mean_roc_auc=result.mean_roc_auc,
            n_tickers=result.n_tickers,
        )

    ranked = sorted(results, key=lambda r: r.mean_roc_auc, reverse=True)
    for rank, result in enumerate(ranked, start=1):
        _log.info(
            "family_ranked", rank=rank, family=result.family, mean_roc_auc=result.mean_roc_auc
        )
    _log.info(
        "comparison_complete",
        baseline_roc_auc=_BASELINE_ROC_AUC,
        families=[
            {
                "family": result.family,
                "mean_roc_auc": result.mean_roc_auc,
                "beats_baseline": result.beats_baseline,
            }
            for result in ranked
        ],
    )
    return ranked


def run(
    settings: MlSettings, tickers: list[str], n_splits: int, tracking_uri: str
) -> list[FamilyResult]:
    """Загрузить чистые фреймы тикеров один раз и сравнить три семьи моделей (без регистрации)."""
    mlflow.set_tracking_uri(tracking_uri)
    session_factory = loader.make_session_factory(str(settings.database_url))
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        with session_factory() as session:
            frame = train_trend.build_ticker_frame(session, ticker, settings)
        frames[ticker] = _drop_unfeatured(frame)
        _log.info("ticker_loaded", ticker=ticker, rows=len(frames[ticker]))
    return compare_families(frames, settings, n_splits)


def main() -> None:
    """CLI: сравнить классы моделей тренда walk-forward (лог в ``trend-model-comparison``)."""
    structlog.configure(processors=[structlog.processors.JSONRenderer()])
    parser = argparse.ArgumentParser(description="Сравнение классов моделей тренда (sanity check)")
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
