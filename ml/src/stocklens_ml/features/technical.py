"""Технические индикаторы для модели тренда (ml-spec §4.4).

Чистые функции над pandas; все окна — трейлинговые (данные ≤ t, без утечек). Sentiment-фича
тренда в этом заходе не включается (D1: новости ~2 недели). RSI — по сглаживанию Уайлдера.
"""

import numpy as np
import pandas as pd

_RSI_PERIOD = 14
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_VOLUME_WINDOW = 20
_REALIZED_VOL_WINDOW = 5
_RETURN_LAGS = 5
_RSI_MAX = 100.0


def return_lags(returns: pd.Series, n_lags: int = _RETURN_LAGS) -> pd.DataFrame:
    """Лаги доходности r_t..r_{t-(n_lags-1)} (сдвиг назад — только прошлое)."""
    return pd.DataFrame({f"r_lag_{lag}": returns.shift(lag) for lag in range(n_lags)})


def rsi(close: pd.Series, period: int = _RSI_PERIOD) -> pd.Series:
    """RSI по Уайлдеру (§4.4): сглаживание EMA с alpha=1/period; диапазон [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    relative_strength = avg_gain / avg_loss
    result = _RSI_MAX - _RSI_MAX / (1.0 + relative_strength)
    # avg_loss=0 (только рост) → RS=inf → RSI=100; формула даёт это в пределе.
    return result.where(avg_loss != 0.0, _RSI_MAX).where(~np.isnan(avg_gain), np.nan)


def macd(
    close: pd.Series,
    fast: int = _MACD_FAST,
    slow: int = _MACD_SLOW,
    signal: int = _MACD_SIGNAL,
) -> pd.DataFrame:
    """MACD(12,26,9) (§4.4): линия MACD, сигнальная линия, гистограмма."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        }
    )


def volume_zscore(volume: pd.Series, window: int = _VOLUME_WINDOW) -> pd.Series:
    """Z-оценка объёма (§4.4): (volume − mean_w) / std_w по трейлинговому окну."""
    numeric = volume.astype(float)
    mean = numeric.rolling(window, min_periods=window).mean()
    std = numeric.rolling(window, min_periods=window).std()
    return (numeric - mean) / std


def realized_vol(parkinson_var: pd.Series, window: int = _REALIZED_VOL_WINDOW) -> pd.Series:
    """Реализованная волатильность за окно (§4.4): sqrt(Σ Паркинсон-дисперсий за window дней)."""
    return np.sqrt(parkinson_var.rolling(window, min_periods=window).sum())
