"""Unit-tests for price corrections (ml-spec §3.1): split, dividend total-return, weekend."""

import math
from datetime import date

import pandas as pd
from stocklens_core.enums import Currency
from stocklens_ml.data import adjust


def _candles(rows: list[tuple[date, float, float, float, float, int, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["trade_date", "open", "high", "low", "close", "volume", "is_weekend_session"],
    )


def _splits(rows: list[tuple[date, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["split_date", "before", "after"])


def _dividends(rows: list[tuple[date, float, Currency]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ex_date", "value", "currency"])


def test_split_adjustment_scales_pre_split_prices() -> None:
    candles = _candles(
        [
            (date(2024, 1, 1), 100.0, 100.0, 100.0, 100.0, 10, False),
            (date(2024, 1, 2), 100.0, 100.0, 100.0, 100.0, 10, False),
            (date(2024, 1, 3), 100.0, 100.0, 100.0, 100.0, 10, False),
            (date(2024, 1, 4), 100.0, 100.0, 100.0, 100.0, 10, False),
        ]
    )
    # 1 акция -> 10: коэффициент before/after = 0.1 для дат строго до split_date.
    splits = _splits([(date(2024, 1, 3), 1, 10)])

    out = adjust.apply_split_adjustment(candles, splits)

    assert list(out["close"]) == [10.0, 10.0, 100.0, 100.0]
    assert list(out["split_factor"]) == [0.1, 0.1, 1.0, 1.0]


def test_split_adjustment_cumulative_for_multiple_splits() -> None:
    candles = _candles(
        [
            (date(2024, 1, 1), 0, 0, 0, 1000.0, 1, False),
            (date(2024, 1, 5), 0, 0, 0, 1000.0, 1, False),
            (date(2024, 1, 10), 0, 0, 0, 1000.0, 1, False),
        ]
    )
    # Два сплита 1:2 и 1:5: для самой ранней даты коэффициент = 0.5 * 0.2 = 0.1.
    splits = _splits([(date(2024, 1, 5), 1, 2), (date(2024, 1, 10), 1, 5)])

    out = adjust.apply_split_adjustment(candles, splits)

    assert out["close"].tolist() == [100.0, 200.0, 1000.0]


def test_total_return_adds_dividend_on_exchange_ex_date() -> None:
    candles = _candles(
        [
            (date(2024, 3, 1), 0, 0, 0, 100.0, 1, False),
            (date(2024, 3, 2), 0, 0, 0, 102.0, 1, False),
            (date(2024, 3, 3), 0, 0, 0, 101.0, 1, False),
        ]
    )
    candles = adjust.apply_split_adjustment(candles, _splits([]))
    # db.ex_date = registryclosedate = 3 марта; биржевая ex-date = сессия строго до = 2 марта.
    dividends = _dividends([(date(2024, 3, 3), 5.0, Currency.RUB)])

    r = adjust.total_return_log(candles, dividends)

    assert math.isnan(r.iloc[0])
    assert r.iloc[1] == math.log((102.0 + 5.0) / 100.0)  # дивиденд добавлен на 2 марта
    assert r.iloc[2] == math.log(101.0 / 102.0)  # на следующий день — без дивиденда


def test_total_return_dividend_scaled_by_split_factor() -> None:
    candles = _candles(
        [
            (date(2024, 1, 1), 0, 0, 0, 1000.0, 1, False),
            (date(2024, 1, 2), 0, 0, 0, 1000.0, 1, False),
            (date(2024, 1, 3), 0, 0, 0, 1000.0, 1, False),
            (date(2024, 1, 10), 0, 0, 0, 100.0, 1, False),
        ]
    )
    candles = adjust.apply_split_adjustment(candles, _splits([(date(2024, 1, 10), 1, 10)]))
    # Дивиденд 50 (в старых акциях) с db.ex_date=3 янв; биржевая ex-date=2 янв (factor 0.1).
    dividends = _dividends([(date(2024, 1, 3), 50.0, Currency.RUB)])

    r = adjust.total_return_log(candles, dividends)

    # close[2 янв] сплит-скорректирован: 1000*0.1=100; дивиденд тоже: 50*0.1=5.
    assert r.iloc[1] == math.log((100.0 + 5.0) / 100.0)


def test_non_rub_dividend_is_skipped() -> None:
    candles = _candles(
        [
            (date(2024, 3, 1), 0, 0, 0, 100.0, 1, False),
            (date(2024, 3, 2), 0, 0, 0, 102.0, 1, False),
            (date(2024, 3, 3), 0, 0, 0, 103.0, 1, False),
        ]
    )
    candles = adjust.apply_split_adjustment(candles, _splits([]))
    # db.ex_date=3 марта → биржевая ex-date=2 марта (индекс 1); USD-дивиденд должен быть пропущен.
    dividends = _dividends([(date(2024, 3, 3), 5.0, Currency.USD)])

    r = adjust.total_return_log(candles, dividends)

    # Доходность 2 марта — чистая (ln(102/100)), без добавления USD-дивиденда.
    assert r.iloc[1] == math.log(102.0 / 100.0)


def test_multiple_dividends_same_gap_day_accumulate() -> None:
    candles = _candles(
        [
            (date(2024, 3, 1), 0, 0, 0, 100.0, 1, False),
            (date(2024, 3, 2), 0, 0, 0, 102.0, 1, False),
            (date(2024, 3, 3), 0, 0, 0, 101.0, 1, False),
        ]
    )
    candles = adjust.apply_split_adjustment(candles, _splits([]))
    # Два дивиденда с одной db.ex_date=3 марта → один gap-день (2 марта), вклады складываются.
    dividends = _dividends(
        [(date(2024, 3, 3), 3.0, Currency.RUB), (date(2024, 3, 3), 2.0, Currency.RUB)]
    )

    r = adjust.total_return_log(candles, dividends)

    assert r.iloc[1] == math.log((102.0 + 5.0) / 100.0)


def test_total_return_handles_unsorted_input() -> None:
    candles = _candles(
        [
            (date(2024, 3, 3), 0, 0, 0, 101.0, 1, False),
            (date(2024, 3, 1), 0, 0, 0, 100.0, 1, False),
            (date(2024, 3, 2), 0, 0, 0, 102.0, 1, False),
        ]
    )
    candles = adjust.apply_split_adjustment(candles, _splits([]))

    r = adjust.total_return_log(candles, _dividends([]))

    # После защитной сортировки доходность считается в хронологическом порядке.
    assert math.isnan(r.iloc[0])
    assert r.iloc[1] == math.log(102.0 / 100.0)


def test_exclude_weekend_drops_weekend_sessions() -> None:
    candles = _candles(
        [
            (date(2025, 3, 14), 0, 0, 0, 100.0, 1, False),
            (date(2025, 3, 15), 0, 0, 0, 101.0, 1, True),  # суббота — weekend-сессия
            (date(2025, 3, 17), 0, 0, 0, 102.0, 1, False),
        ]
    )

    out = adjust.exclude_weekend(candles)

    assert out["trade_date"].tolist() == [date(2025, 3, 14), date(2025, 3, 17)]
    assert len(out) == 2
