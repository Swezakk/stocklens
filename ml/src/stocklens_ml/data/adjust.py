"""Коррекция дневных котировок перед расчётом фич (ml-spec §3.1).

Чистые функции над pandas.DataFrame — без обращения к БД, юнит-тестируемы. Корректный
порядок для расчёта доходностей: сплит-коррекция → исключение weekend → total-return.
Weekend исключается ДО доходностей, иначе доходность сессии после выходного считалась бы
относительно аномальной weekend-сессии (другой режим ликвидности).
"""

import numpy as np
import pandas as pd
import structlog
from stocklens_core.enums import Currency

_log = structlog.get_logger(__name__)

#: OHLC-колонки, корректируемые на сплит.
_PRICE_COLUMNS = ("open", "high", "low", "close")


def split_factors(trade_dates: pd.Series, splits: pd.DataFrame) -> pd.Series:
    """Кумулятивный коэффициент сплит-коррекции для каждой торговой даты.

    Цены строго ДО ``split_date`` умножаются на ``before/after`` (1 акция → after акций
    означает деление цены на after, т.е. коэффициент before/after). При нескольких сплитах
    коэффициенты перемножаются; на/после всех сплитов коэффициент равен 1.0.
    """
    factor = pd.Series(1.0, index=trade_dates.index)
    for split in splits.itertuples(index=False):
        ratio = float(split.before) / float(split.after)
        factor = factor.where(trade_dates >= split.split_date, factor * ratio)
    return factor


def apply_split_adjustment(candles: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Вернуть копию ``candles`` со сплит-скорректированными OHLC и колонкой ``split_factor``."""
    out = candles.copy()
    factor = split_factors(out["trade_date"], splits)
    out["split_factor"] = factor
    for column in _PRICE_COLUMNS:
        out[column] = out[column] * factor
    return out


def exclude_weekend(candles: pd.DataFrame) -> pd.DataFrame:
    """Убрать сессии выходного дня MOEX (ml-spec §3.1; ``is_weekend_session=True``)."""
    return candles.loc[~candles["is_weekend_session"]].reset_index(drop=True)


def total_return_log(candles_adj: pd.DataFrame, dividends: pd.DataFrame) -> pd.Series:
    """Логарифмическая total-return доходность с поправкой на дивиденды.

    ``candles_adj`` — сплит-скорректированные свечи (после :func:`apply_split_adjustment`)
    с исключёнными weekend-сессиями, отсортированы по ``trade_date`` по возрастанию.

    Дивиденд относится к биржевой ex-date — последней торговой сессии строго ДО
    ``db.ex_date`` (registryclosedate, T+1 — ml-spec §3.1). В этот день дивиденд (в той же
    сплит-шкале, ``value × split_factor``) добавляется к ``close`` перед расчётом доходности.
    Не-рублёвые дивиденды пропускаются с логом (конвертация по курсам — отдельная фаза).
    Вход сортируется по дате защитно: ``prior[-1]`` и ``close[:-1]`` опираются на
    возрастающий порядок.
    """
    candles_adj = candles_adj.sort_values("trade_date").reset_index(drop=True)
    close = candles_adj["close"].to_numpy(dtype=float)
    factor = candles_adj["split_factor"].to_numpy(dtype=float)
    trade_date = candles_adj["trade_date"]
    dividend_add = np.zeros(len(candles_adj), dtype=float)

    for dividend in dividends.itertuples(index=False):
        if dividend.currency != Currency.RUB:
            _log.warning(
                "dividend_non_rub_skipped",
                ex_date=str(dividend.ex_date),
                currency=str(dividend.currency),
            )
            continue
        prior = trade_date.index[trade_date < dividend.ex_date]
        if len(prior) == 0:
            continue
        gap_session = int(prior[-1])
        dividend_add[gap_session] += float(dividend.value) * factor[gap_session]

    adjusted_close = close + dividend_add
    previous_close = np.concatenate(([np.nan], close[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.log(adjusted_close / previous_close)
    return pd.Series(returns, index=candles_adj.index, name="r")
