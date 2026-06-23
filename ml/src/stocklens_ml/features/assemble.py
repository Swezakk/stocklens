"""Сборка фрейма фич волатильности per-ticker (ml-spec §2.1, §4.6).

Оркестрирует обязательный порядок (§3.1): сплит-коррекция → исключение weekend → доходности
и Паркинсон → HAR-регрессоры и RV-таргет. Возвращает фрейм с NaN (warm-up окон и хвостовой
таргет) — отбор обучающих строк (dropna) делает обучающий скрипт; инференс берёт последнюю
строку фич. Фичи — только трейлинговые (данные ≤ t); RV-таргет — forward (это таргет).
"""

from datetime import date

import pandas as pd

from stocklens_ml.config import HORIZON_DAYS, TRAIN_START_DEFAULT
from stocklens_ml.data import adjust
from stocklens_ml.features import volatility

#: Колонки-фичи (без утечек) — для отбора при проверке и обучении.
FEATURE_COLUMNS = ["rv_d", "rv_w", "rv_m"]
TARGET_COLUMN = "rv_target"


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
