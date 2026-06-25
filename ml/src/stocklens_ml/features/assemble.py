"""Сборка фрейма фич волатильности per-ticker (ml-spec §2.1, §4.6).

Оркестрирует обязательный порядок (§3.1): сплит-коррекция → исключение weekend → доходности
и Паркинсон → HAR-регрессоры и RV-таргет. Возвращает фрейм с NaN (warm-up окон и хвостовой
таргет) — отбор обучающих строк (dropna) делает обучающий скрипт; инференс берёт последнюю
строку фич. Фичи — только трейлинговые (данные ≤ t); RV-таргет — forward (это таргет).
"""

from datetime import date

import numpy as np
import pandas as pd

from stocklens_ml.config import HORIZON_DAYS, TRAIN_START_DEFAULT
from stocklens_ml.data import adjust
from stocklens_ml.features import technical, volatility

#: Колонки-фичи (без утечек) — для отбора при проверке и обучении.
FEATURE_COLUMNS = ["rv_d", "rv_w", "rv_m"]
TARGET_COLUMN = "rv_target"

#: Контракт входного фрейма serving (единый источник для train-input и API-инференса):
#: доходность + HAR-регрессоры. Живёт здесь (чистый pandas, без arch), чтобы serving-слой
#: импортировал его без зависимости от тяжёлого pyfunc-модуля.
SERVING_FEATURES = ["r", *FEATURE_COLUMNS]

#: Лаги доходности тренда r_{t}..r_{t-4} (ml-spec §4.4); порядок фиксирован для fit/predict.
_TREND_RETURN_LAGS = 5

#: Колонки-фичи модели тренда (без утечек, трейлинговые) — точный упорядоченный контракт
#: матрицы фич для fit/predict и serving (ml-spec §4.4): лаги доходности → RSI → MACD →
#: z-объём → реализованная волатильность.
TREND_FEATURE_COLUMNS = [
    *(f"r_lag_{lag}" for lag in range(_TREND_RETURN_LAGS)),
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "volume_zscore",
    "realized_vol",
]
TREND_TARGET_COLUMN = "trend_target"


def build_volatility_frame(
    candles: pd.DataFrame,
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
    train_start: date = TRAIN_START_DEFAULT,
    horizon: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """Собрать фрейм фич и таргета волатильности для одной бумаги.

    Колонки: ``trade_date``, ``r`` (доходность), ``rv_d``/``rv_w``/``rv_m`` (HAR-регрессоры),
    ``rv_target``. Строки до ``train_start`` отбрасываются (структурный разрыв 2022, D8).
    """
    adjusted = adjust.apply_split_adjustment(candles, splits)
    adjusted = adjust.exclude_weekend(adjusted)
    adjusted = adjusted.sort_values("trade_date").reset_index(drop=True)

    returns = adjust.total_return_log(adjusted, dividends)
    parkinson = volatility.parkinson_variance(adjusted)
    har = volatility.har_regressors(parkinson)
    target = volatility.realized_variance_target(returns, horizon)

    frame = pd.DataFrame(
        {
            "trade_date": adjusted["trade_date"].to_numpy(),
            "r": returns.to_numpy(),
            "rv_d": har["rv_d"].to_numpy(),
            "rv_w": har["rv_w"].to_numpy(),
            "rv_m": har["rv_m"].to_numpy(),
            "rv_target": target.to_numpy(),
        }
    )
    return frame.loc[frame["trade_date"] >= train_start].reset_index(drop=True)


def build_trend_frame(
    candles: pd.DataFrame,
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
    train_start: date = TRAIN_START_DEFAULT,
    horizon: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """Собрать фрейм фич и бинарного таргета тренда для одной бумаги (ml-spec §4.4–4.5).

    Колонки: ``trade_date``, технические фичи :data:`TREND_FEATURE_COLUMNS` (трейлинговые,
    данные ≤ t), ``trend_target`` (forward: 1, если ln(close_{t+H}/close_t) > 0, иначе 0;
    ровно 0 → класс 0 по строгому >). Последние ``horizon`` строк не имеют будущего → NaN
    (обучающий скрипт их отбрасывает). Строки до ``train_start`` отбрасываются (структурный
    разрыв 2022, D8).
    """
    adjusted = adjust.apply_split_adjustment(candles, splits)
    adjusted = adjust.exclude_weekend(adjusted)
    adjusted = adjusted.sort_values("trade_date").reset_index(drop=True)

    returns = adjust.total_return_log(adjusted, dividends)
    close = adjusted["close"]
    lags = technical.return_lags(returns, n_lags=_TREND_RETURN_LAGS)
    macd = technical.macd(close)
    parkinson = volatility.parkinson_variance(adjusted)

    frame = pd.DataFrame({"trade_date": adjusted["trade_date"].to_numpy()})
    for lag in range(_TREND_RETURN_LAGS):
        frame[f"r_lag_{lag}"] = lags[f"r_lag_{lag}"].to_numpy()
    frame["rsi"] = technical.rsi(close).to_numpy()
    frame["macd"] = macd["macd"].to_numpy()
    frame["macd_signal"] = macd["macd_signal"].to_numpy()
    frame["macd_hist"] = macd["macd_hist"].to_numpy()
    frame["volume_zscore"] = technical.volume_zscore(adjusted["volume"]).to_numpy()
    frame["realized_vol"] = technical.realized_vol(parkinson).to_numpy()
    frame[TREND_TARGET_COLUMN] = _trend_target(close, horizon).to_numpy()

    return frame.loc[frame["trade_date"] >= train_start].reset_index(drop=True)


def _trend_target(close: pd.Series, horizon: int) -> pd.Series:
    """Бинарный таргет направления (§4.5): 1 при ln(close_{t+H}/close_t) > 0, иначе 0.

    Forward-доходность за горизонт; последние ``horizon`` строк не имеют будущего close →
    NaN (а не 0): без явной маски сравнение NaN > 0 дало бы False и ложный класс 0.
    """
    forward = close.astype(float).shift(-horizon)
    log_return = np.log(forward / close.astype(float))
    label = (log_return > 0.0).astype(float)
    return label.where(log_return.notna(), np.nan).rename(TREND_TARGET_COLUMN)
