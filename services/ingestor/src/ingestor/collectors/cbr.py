"""Сборщики данных ЦБ РФ: курсы валют и ключевая ставка.

Источники:
- XML_daily.asp — актуальные курсы на сегодня (encoding windows-1251).
- XML_dynamic.asp — диапазон курсов по одной валюте (backfill).
- SOAP DailyInfo.asmx/KeyRate — ключевая ставка за диапазон дат.

Номинал (Nominal) в XML_daily и XML_dynamic: ЦБ публикует Value за Nominal единиц.
Итоговый курс = Value / Nominal, независимо от валюты.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation

import requests
import structlog
from sqlalchemy.orm import Session, sessionmaker
from stocklens_core.enums import Currency

from ingestor import heartbeat
from ingestor.repositories import (
    last_currency_rate_date,
    last_key_rate_date,
    upsert_currency_rates,
    upsert_key_rates,
)
from ingestor.run_journal import collector_run
from ingestor.settings import IngestorSettings

log = structlog.get_logger(__name__)

_CBR_DAILY_URL = "https://www.cbr.ru/scripts/XML_daily.asp"
_CBR_DYNAMIC_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
_CBR_KEY_RATE_ENDPOINT = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
_REQUEST_TIMEOUT_SECONDS = 30

_BACKFILL_FROM_DATE = date(2013, 1, 1)

_CURRENCY_CBR_IDS: dict[Currency, str] = {
    Currency.USD: "R01235",
    Currency.EUR: "R01239",
    Currency.CNY: "R01375",
}

_CBR_DATE_FORMAT = "%d/%m/%Y"


@dataclass(frozen=True)
class _RateRow:
    """Курс одной валюты на одну дату."""

    currency: Currency
    rate_date: date
    rate: Decimal


def sync_currency_rates(
    session_factory: sessionmaker[Session],
    settings: IngestorSettings,
) -> None:
    """Загрузить актуальные курсы USD/EUR/CNY из XML_daily и upsert в БД.

    Args:
        session_factory: Фабрика синхронных SQLAlchemy-сессий.
        settings: Конфигурация ingestor.
    """
    with collector_run(session_factory, "cbr_rates") as journal:
        content = _fetch_bytes(_CBR_DAILY_URL)
        rows = _parse_daily_xml(content)

        with session_factory() as session:
            added = upsert_currency_rates(
                session,
                [{"currency": r.currency, "rate_date": r.rate_date, "rate": r.rate} for r in rows],
            )
            session.commit()

        journal.add_records(added)
        log.info("cbr_rates_synced", records=added)
        heartbeat.touch(settings.heartbeat_path)


def backfill_currency_rates(
    session_factory: sessionmaker[Session],
    settings: IngestorSettings,
) -> None:
    """Backfill курсов USD/EUR/CNY через XML_dynamic начиная с последней записи в БД.

    Если БД пуста — backfill с _BACKFILL_FROM_DATE. Три запроса (по одному на валюту),
    не день-за-днём.

    Args:
        session_factory: Фабрика синхронных SQLAlchemy-сессий.
        settings: Конфигурация ingestor.
    """
    with collector_run(session_factory, "cbr_rates_backfill") as journal:
        today = datetime.now(UTC).date()

        for currency in (Currency.USD, Currency.EUR, Currency.CNY):
            with session_factory() as session:
                last = last_currency_rate_date(session, currency)

            from_date = (last + timedelta(days=1)) if last else _BACKFILL_FROM_DATE
            if from_date > today:
                log.info("cbr_currency_backfill_skip", currency=currency, reason="up_to_date")
                continue

            cbr_id = _CURRENCY_CBR_IDS[currency]
            params = {
                "date_req1": from_date.strftime(_CBR_DATE_FORMAT),
                "date_req2": today.strftime(_CBR_DATE_FORMAT),
                "VAL_NM_RQ": cbr_id,
            }
            content = _fetch_bytes(_CBR_DYNAMIC_URL, params=params)
            rows = _parse_dynamic_xml(content, currency)

            with session_factory() as session:
                added = upsert_currency_rates(
                    session,
                    [
                        {"currency": r.currency, "rate_date": r.rate_date, "rate": r.rate}
                        for r in rows
                    ],
                )
                session.commit()

            journal.add_records(added)
            log.info("cbr_currency_backfill_done", currency=currency, records=added)
            heartbeat.touch(settings.heartbeat_path)


def sync_key_rate(
    session_factory: sessionmaker[Session],
    settings: IngestorSettings,
) -> None:
    """Загрузить ключевую ставку через SOAP DailyInfo.asmx за период с последней записи.

    Если БД пуста — backfill с _BACKFILL_FROM_DATE.

    Args:
        session_factory: Фабрика синхронных SQLAlchemy-сессий.
        settings: Конфигурация ingestor.
    """
    with collector_run(session_factory, "cbr_key_rate") as journal:
        today = datetime.now(UTC).date()

        with session_factory() as session:
            last = last_key_rate_date(session)

        from_date = (last + timedelta(days=1)) if last else _BACKFILL_FROM_DATE
        if from_date > today:
            log.info("cbr_key_rate_skip", reason="up_to_date")
            return

        rows = _fetch_key_rates(from_date, today)

        with session_factory() as session:
            added = upsert_key_rates(
                session,
                [{"rate_date": r.rate_date, "rate": r.rate} for r in rows],
            )
            session.commit()

        journal.add_records(added)
        log.info("cbr_key_rate_synced", records=added)
        heartbeat.touch(settings.heartbeat_path)


def _fetch_bytes(url: str, params: dict[str, str] | None = None) -> bytes:
    """Выполнить GET-запрос и вернуть тело ответа как bytes.

    ЦБ отдаёт XML в windows-1251; декодирование выполняется в парсерах,
    не здесь — ElementTree сам читает encoding из XML-декларации.

    Args:
        url: URL для GET.
        params: Опциональные query-параметры.

    Returns:
        Тело ответа в байтах.
    """
    response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def _parse_daily_xml(content: bytes) -> list[_RateRow]:
    """Распарсить XML_daily: вернуть строки только для USD/EUR/CNY.

    Args:
        content: Байты XML в encoding windows-1251.

    Returns:
        Список курсов на дату из атрибута ValCurs[@Date].
    """
    root = ET.fromstring(content.decode("windows-1251"))
    rate_date = _parse_cbr_date(root.attrib.get("Date", ""))

    char_to_currency = {
        "USD": Currency.USD,
        "EUR": Currency.EUR,
        "CNY": Currency.CNY,
    }

    rows: list[_RateRow] = []
    for valute in root.findall("Valute"):
        char_code_el = valute.find("CharCode")
        value_el = valute.find("Value")
        nominal_el = valute.find("Nominal")

        if char_code_el is None or value_el is None or nominal_el is None:
            continue

        char_code = (char_code_el.text or "").strip()
        currency = char_to_currency.get(char_code)
        if currency is None:
            continue

        rate = _parse_cbr_decimal(value_el.text or "", nominal_el.text or "")
        if rate is None:
            log.warning("cbr_daily_bad_value", char_code=char_code)
            continue

        rows.append(_RateRow(currency=currency, rate_date=rate_date, rate=rate))

    return rows


def _parse_dynamic_xml(content: bytes, currency: Currency) -> list[_RateRow]:
    """Распарсить XML_dynamic: серия курсов одной валюты за диапазон дат.

    Args:
        content: Байты XML в encoding windows-1251.
        currency: Валюта, которой соответствует этот XML.

    Returns:
        Список курсов по датам.
    """
    root = ET.fromstring(content.decode("windows-1251"))

    rows: list[_RateRow] = []
    for record in root.findall("Record"):
        date_attr = record.attrib.get("Date", "")
        value_el = record.find("Value")
        nominal_el = record.find("Nominal")

        if value_el is None or nominal_el is None:
            continue

        rate_date = _parse_cbr_date(date_attr)
        rate = _parse_cbr_decimal(value_el.text or "", nominal_el.text or "")
        if rate is None:
            log.warning("cbr_dynamic_bad_value", date=date_attr)
            continue

        rows.append(_RateRow(currency=currency, rate_date=rate_date, rate=rate))

    return rows


@dataclass(frozen=True)
class _KeyRateRow:
    """Ключевая ставка на одну дату."""

    rate_date: date
    rate: Decimal


def _fetch_key_rates(from_date: date, to_date: date) -> list[_KeyRateRow]:
    """Получить ключевые ставки через SOAP DailyInfo.asmx/KeyRate.

    Args:
        from_date: Начало диапазона.
        to_date: Конец диапазона (включительно).

    Returns:
        Список ставок по датам.
    """
    soap_body = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
        "<soap:Body>"
        '<KeyRate xmlns="http://web.cbr.ru/">'
        f"<fromDate>{from_date.isoformat()}T00:00:00</fromDate>"
        f"<ToDate>{to_date.isoformat()}T00:00:00</ToDate>"
        "</KeyRate>"
        "</soap:Body>"
        "</soap:Envelope>"
    )
    response = requests.post(
        _CBR_KEY_RATE_ENDPOINT,
        data=soap_body.encode("utf-8"),
        headers={
            "Content-Type": "text/xml; charset=utf-8",
            "SOAPAction": '"http://web.cbr.ru/KeyRate"',
        },
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return _parse_key_rate_soap(response.content)


def _parse_key_rate_soap(content: bytes) -> list[_KeyRateRow]:
    """Распарсить SOAP-ответ KeyRate: извлечь пары DT/Rate из diffgram.

    Ответ содержит элементы <KR><DT>2026-06-10T00:00:00+03:00</DT><Rate>14.50</Rate></KR>.

    Args:
        content: Байты SOAP-ответа (UTF-8).

    Returns:
        Список ставок по датам.
    """
    root = ET.fromstring(content.decode("utf-8"))

    rows: list[_KeyRateRow] = []
    for kr in root.iter("KR"):
        dt_el = kr.find("DT")
        rate_el = kr.find("Rate")

        if dt_el is None or rate_el is None:
            continue

        dt_text = (dt_el.text or "").strip()
        rate_text = (rate_el.text or "").strip()

        if not dt_text or not rate_text:
            continue

        rate_date = date.fromisoformat(dt_text[:10])
        try:
            rate = Decimal(rate_text)
        except InvalidOperation:
            log.warning("cbr_key_rate_bad_value", dt=dt_text, rate=rate_text)
            continue

        rows.append(_KeyRateRow(rate_date=rate_date, rate=rate))

    return rows


def _parse_cbr_date(date_str: str) -> date:
    """Распарсить дату в формате ЦБ (DD.MM.YYYY).

    Args:
        date_str: Строка даты.

    Returns:
        Объект date.
    """
    return date(int(date_str[6:10]), int(date_str[3:5]), int(date_str[0:2]))


def _parse_cbr_decimal(value_text: str, nominal_text: str) -> Decimal | None:
    """Вычислить курс: Value / Nominal с заменой десятичной запятой на точку.

    ЦБ публикует Value за Nominal единиц валюты — делим для получения курса за 1 единицу.

    Args:
        value_text: Строка значения (запятая как десятичный разделитель).
        nominal_text: Строка номинала.

    Returns:
        Decimal или None при ошибке парсинга.
    """
    try:
        value = Decimal(value_text.strip().replace(",", "."))
        nominal = Decimal(nominal_text.strip().replace(",", "."))
        return value / nominal
    except (InvalidOperation, ZeroDivisionError):
        return None
