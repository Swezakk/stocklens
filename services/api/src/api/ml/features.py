"""Сборка фрейма фич волатильности на инференсе (ml-spec §8.5).

Единый расчёт с обучением: делегирует в ``build_volatility_frame`` из ``stocklens_ml`` — тот
же код фич, что при обучении (без train/serve skew). Импортируется только чистый pandas-код
пакета (``features.assemble``), без ``arch``/``mlflow`` — этот модуль грузится и в unit-тестах
сервиса, где модель застаблена.
"""

from datetime import date

import pandas as pd
from stocklens_ml.features.assemble import SERVING_FEATURES, build_volatility_frame

__all__ = ["MIN_VOLATILITY_HISTORY", "SERVING_FEATURES", "build_serving_frame"]

#: Минимум валидных наблюдений доходности для устойчивого прогноза (= порог фита GARCH,
#: _MIN_GARCH_OBS); ниже — InsufficientHistoryError вместо падения модели.
MIN_VOLATILITY_HISTORY = 100


def build_serving_frame(
    candles: pd.DataFrame,
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    train_start: date,
    horizon: int,
) -> pd.DataFrame:
    """Собрать фрейм фич (``trade_date`` + SERVING_FEATURES + ``rv_target``) для инференса.

    ``rv_target`` (forward) на инференсе не используется — прогноз берётся как-of последней
    строки; ``horizon`` влияет только на него, поэтому serving-фичи от него не зависят.
    """
    if candles.empty:
        return pd.DataFrame(columns=["trade_date", *SERVING_FEATURES, "rv_target"])
    return build_volatility_frame(
        candles, dividends, splits, train_start=train_start, horizon=horizon
    )
