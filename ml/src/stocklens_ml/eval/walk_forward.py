"""Walk-forward валидация моделей волатильности (ml-spec §6.1, D6).

``TimeSeriesSplit`` с expanding-окном и ``gap=horizon``: gap исключает из конца train-окна
строки, чьи forward-таргеты перекрывают test (forward purging — иначе утечка). Случайный
K-fold запрещён (§8.5). Каждый форкастер реализует прогноз дисперсии для test-строк,
используя только данные ≤ соответствующей даты.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from stocklens_ml.config import HORIZON_DAYS
from stocklens_ml.eval import classification_metrics, metrics
from stocklens_ml.features.assemble import TREND_TARGET_COLUMN
from stocklens_ml.models import baselines, garch
from stocklens_ml.models.har import HAR_REGRESSORS, HarRvModel

#: Форкастер: (фрейм, индексы train, индексы test) → прогноз дисперсии для test-строк (доли²).
Forecaster = Callable[
    [pd.DataFrame, npt.NDArray[np.intp], npt.NDArray[np.intp]], npt.NDArray[np.float64]
]

_TARGET = "rv_target"
_RETURN = "r"


def evaluate(
    frame: pd.DataFrame,
    forecasters: dict[str, Forecaster],
    n_splits: int = 5,
    gap: int = HORIZON_DAYS,
) -> dict[str, dict[str, float]]:
    """Прогнать walk-forward и вернуть QLIKE/RMSE на дисперсиях по каждому форкастеру."""
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    realized_parts: list[npt.NDArray[np.float64]] = []
    forecast_parts: dict[str, list[npt.NDArray[np.float64]]] = {name: [] for name in forecasters}

    target = frame[_TARGET].to_numpy(dtype=float)
    for train_idx, test_idx in splitter.split(frame):
        realized_parts.append(target[test_idx])
        for name, forecaster in forecasters.items():
            forecast_parts[name].append(forecaster(frame, train_idx, test_idx))

    realized = np.concatenate(realized_parts)
    return {
        name: {
            "qlike": metrics.qlike(realized, np.concatenate(parts)),
            "rmse": metrics.rmse(realized, np.concatenate(parts)),
        }
        for name, parts in forecast_parts.items()
    }


def evaluate_trend(
    frame: pd.DataFrame,
    forecasters: dict[str, Forecaster],
    n_splits: int = 5,
    gap: int = HORIZON_DAYS,
) -> dict[str, dict[str, float]]:
    """Walk-forward оценка тренда: accuracy/F1/ROC-AUC на P(up) по каждому форкастеру (§6.2).

    Логика split-и-конкатенации идентична :func:`evaluate` (тот же ``TimeSeriesSplit`` с
    forward-purging через ``gap``); отличается лишь метрический словарь — классификационный.
    Форкастеры возвращают вероятность роста P(up) для test-строк (тот же контракт сигнатуры).

    ``roc_auc`` не определён на одноклассовой объединённой test-выборке — ``classification_metrics
    .roc_auc`` бросит ``ValueError`` с русскоязычным сообщением; ловить его здесь намеренно не
    будем (одноклассовый прогон — сигнал проблемы данных, не штатный путь).
    """
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    realized_parts: list[npt.NDArray[np.float64]] = []
    forecast_parts: dict[str, list[npt.NDArray[np.float64]]] = {name: [] for name in forecasters}

    target = frame[TREND_TARGET_COLUMN].to_numpy(dtype=float)
    for train_idx, test_idx in splitter.split(frame):
        realized_parts.append(target[test_idx])
        for name, forecaster in forecasters.items():
            forecast_parts[name].append(forecaster(frame, train_idx, test_idx))

    realized = np.concatenate(realized_parts)
    return {
        name: {
            "accuracy": classification_metrics.accuracy(realized, np.concatenate(parts)),
            "f1": classification_metrics.f1(realized, np.concatenate(parts)),
            "roc_auc": classification_metrics.roc_auc(realized, np.concatenate(parts)),
        }
        for name, parts in forecast_parts.items()
    }


def rw_rv_forecaster(
    frame: pd.DataFrame, train_idx: npt.NDArray[np.intp], test_idx: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """Baseline RW-RV: трейлинговая реализованная дисперсия в test-строках (§5.3)."""
    trailing = baselines.rw_rv_forecast(frame[_RETURN]).to_numpy(dtype=float)
    return np.asarray(trailing[test_idx], dtype=np.float64)


def har_forecaster(
    frame: pd.DataFrame, train_idx: npt.NDArray[np.intp], test_idx: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """HAR-RV: OLS на train-окне, прогноз на test-строках с валидными регрессорами (§5.2)."""
    train = frame.iloc[train_idx]
    model = HarRvModel().fit(train, train[_TARGET])
    test = frame.iloc[test_idx]
    features = test[HAR_REGRESSORS].to_numpy(dtype=float)
    forecast = np.full(len(test_idx), np.nan)
    valid = np.where(np.isfinite(features).all(axis=1))[0]
    if valid.size:
        forecast[valid] = model.predict(test.iloc[valid])
    return forecast


def garch_forecaster(
    frame: pd.DataFrame, train_idx: npt.NDArray[np.intp], test_idx: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """GARCH(1,1): рефит на доходностях ≤ каждой test-даты, прогноз H-дневной дисперсии (§5.1).

    История ``returns[:index+1]`` включает доходность самой test-даты (она известна на t и не
    относится к будущему таргету t+1..t+H), поэтому утечки нет.
    """
    returns = frame[_RETURN]
    forecast = np.full(len(test_idx), np.nan)
    for position, index in enumerate(test_idx):
        try:
            forecast[position] = garch.forecast_variance(returns.iloc[: index + 1])
        except ValueError:
            continue
    return forecast


def default_forecasters() -> dict[str, Forecaster]:
    """Стандартный набор: baseline RW-RV, HAR-RV, GARCH (порядок не важен)."""
    return {
        "baseline_rw_rv": rw_rv_forecaster,
        "har_rv": har_forecaster,
        "garch": garch_forecaster,
    }
