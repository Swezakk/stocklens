"""Тесты парсинга ответов MOEX ISS в доменные объекты."""

from datetime import date
from decimal import Decimal

from ingestor.parsing import (
    ParsedCandle,
    ParsedDividend,
    ParsedSplit,
    parse_candle,
    parse_constituent,
    parse_dividend,
    parse_split,
)
from stocklens_core.enums import Currency


def _candle_row(
    trade_date: str = "2026-06-02",
    open_: float | None = 321.00,
    high: float | None = 324.00,
    low: float | None = 319.50,
    close: float | None = 323.00,
    volume: int = 17_000_000,
    value: float = 5_500_000_000.00,
    board: str = "TQBR",
    secid: str = "SBER",
) -> dict[str, object]:
    return {
        "BOARDID": board,
        "TRADEDATE": trade_date,
        "SECID": secid,
        "OPEN": open_,
        "HIGH": high,
        "LOW": low,
        "CLOSE": close,
        "VOLUME": volume,
        "VALUE": value,
    }


class TestParseCandle:
    def test_weekday_candle_is_not_weekend_session(self) -> None:
        row = _candle_row(trade_date="2026-06-02")  # понедельник
        result = parse_candle(row)
        assert result is not None
        assert result.is_weekend_session is False

    def test_saturday_candle_is_weekend_session(self) -> None:
        row = _candle_row(trade_date="2026-06-06")  # суббота
        result = parse_candle(row)
        assert result is not None
        assert result.is_weekend_session is True

    def test_sunday_candle_is_weekend_session(self) -> None:
        row = _candle_row(trade_date="2026-06-07")  # воскресенье
        result = parse_candle(row)
        assert result is not None
        assert result.is_weekend_session is True

    def test_decimal_exactness(self) -> None:
        row = _candle_row(close=322.99)
        result = parse_candle(row)
        assert result is not None
        assert result.close == Decimal("322.99")

    def test_null_ohlc_returns_none(self) -> None:
        row = _candle_row(open_=None, high=None, low=None, close=None)
        assert parse_candle(row) is None

    def test_partial_null_ohlc_returns_none(self) -> None:
        row = _candle_row(open_=None)
        assert parse_candle(row) is None

    def test_candle_fields_populated(self) -> None:
        row = _candle_row(
            trade_date="2026-06-02",
            open_=321.00,
            high=324.00,
            low=319.50,
            close=323.00,
            volume=17_000_000,
            value=5_500_000_000.00,
            board="TQBR",
            secid="SBER",
        )
        result = parse_candle(row)
        assert result == ParsedCandle(
            ticker="SBER",
            board="TQBR",
            trade_date=date(2026, 6, 2),
            open=Decimal("321.0"),
            high=Decimal("324.0"),
            low=Decimal("319.5"),
            close=Decimal("323.0"),
            volume=17_000_000,
            value=Decimal("5500000000.0"),
            is_weekend_session=False,
        )


class TestParseDividend:
    def test_rub_currency_mapped(self) -> None:
        row = {"registryclosedate": "2019-06-13", "value": 16.0, "currencyid": "RUB"}
        result = parse_dividend("SBER", row)
        assert result.currency is Currency.RUB

    def test_sur_mapped_to_rub(self) -> None:
        row = {"registryclosedate": "2021-05-12", "value": 18.7, "currencyid": "SUR"}
        result = parse_dividend("SBER", row)
        assert result.currency is Currency.RUB

    def test_unknown_currency_returns_none(self) -> None:
        row = {"registryclosedate": "2023-01-01", "value": 5.0, "currencyid": "XYZ"}
        result = parse_dividend("TEST", row)
        assert result.currency is None

    def test_decimal_exactness(self) -> None:
        row = {"registryclosedate": "2021-05-12", "value": 18.7, "currencyid": "RUB"}
        result = parse_dividend("SBER", row)
        assert result.value == Decimal("18.7")

    def test_dividend_fields(self) -> None:
        row = {"registryclosedate": "2019-06-13", "value": 16, "currencyid": "RUB"}
        result = parse_dividend("SBER", row)
        assert result == ParsedDividend(
            ticker="SBER",
            ex_date=date(2019, 6, 13),
            value=Decimal("16"),
            currency=Currency.RUB,
        )


class TestParseSplit:
    def test_split_row_parsed(self) -> None:
        row: dict[str, object] = {
            "tradedate": "2024-06-10",
            "secid": "TRNFP",
            "before": 1,
            "after": 100,
        }
        result = parse_split(row)
        assert result == ParsedSplit(
            ticker="TRNFP",
            split_date=date(2024, 6, 10),
            before=1,
            after=100,
        )


class TestParseConstituent:
    def test_constituent_parsed(self) -> None:
        row: dict[str, object] = {
            "ticker": "SBER",
            "shortnames": "Сбербанк",
            "secids": "SBER",
            "weight": 14.5,
        }
        result = parse_constituent(row)
        assert result.ticker == "SBER"
        assert result.name == "Сбербанк"
