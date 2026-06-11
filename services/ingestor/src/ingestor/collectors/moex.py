"""Сборщики данных MOEX ISS: ценные бумаги, свечи, индекс, дивиденды, сплиты.

Каждый сборщик оборачивается в collector_run и независим от остальных:
сбой одного не прерывает цикл.
"""

from datetime import timedelta

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from stocklens_core.models.market import Security
from stocklens_core.models.portfolio import PortfolioPosition

from ingestor import heartbeat
from ingestor.aliases_seed import TICKER_ALIASES
from ingestor.iss_client import MoexIssClient
from ingestor.parsing import (
    ParsedConstituent,
    parse_candle,
    parse_constituent,
    parse_dividend,
    parse_split,
)
from ingestor.repositories import (
    collection_tickers,
    get_known_tickers,
    last_candle_date,
    last_index_date,
    upsert_candles,
    upsert_dividends,
    upsert_index_values,
    upsert_securities,
    upsert_splits,
    watchlist_tickers,
)
from ingestor.run_journal import collector_run
from ingestor.settings import IngestorSettings

log = structlog.get_logger(__name__)

_IMOEX_INDEX_PATH = "history/engines/stock/markets/index/boards/SNDX/securities/IMOEX.json"
_IMOEX_ANALYTICS_PATH = "statistics/engines/stock/markets/index/analytics/IMOEX.json"
_CANDLES_PATH_TEMPLATE = "history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"

# Деактивируем вышедшие бумаги только при полном ответе MOEX
# — защита от ложной деактивации при пустом/усечённом ответе.
_MIN_CONSTITUENTS_FOR_DEACTIVATION = 30


def sync_securities(
    client: MoexIssClient,
    session_factory: sessionmaker[Session],
    settings: IngestorSettings,
) -> None:
    """Синхронизировать список ценных бумаг из IMOEX или явного списка тикеров.

    Режим IMOEX: постранично скачивает состав индекса из analytics-эндпоинта.
    Режим явного списка: запрашивает /iss/securities/{ticker}.json для каждого тикера.

    Args:
        client: Клиент MOEX ISS.
        session_factory: Фабрика синхронных сессий.
        settings: Конфигурация ingestor.
    """
    with collector_run(session_factory, "moex_securities") as journal:
        if settings.tickers_universe.upper() == "IMOEX":
            rows = client.fetch_block_paginated(
                _IMOEX_ANALYTICS_PATH,
                "analytics",
            )
            constituents = [parse_constituent(r) for r in rows]
            index_tickers = {c.ticker for c in constituents}
            deactivate = len(constituents) >= _MIN_CONSTITUENTS_FOR_DEACTIVATION

            with session_factory() as session:
                wl_tickers = watchlist_tickers(session)

            extra_constituents = _fetch_watchlist_extras(client, wl_tickers, index_tickers)
            all_constituents = constituents + extra_constituents

            with session_factory() as session:
                portfolio_tickers = _portfolio_held_tickers(session)
                protected = wl_tickers | portfolio_tickers
                added = upsert_securities(
                    session,
                    all_constituents,
                    deactivate_missing=deactivate,
                    alias_seed=TICKER_ALIASES,
                    protected_tickers=protected,
                )
                session.commit()
            journal.add_records(added)

            log.info(
                "securities_synced",
                count=len(constituents),
                watchlist_extras=len(extra_constituents),
                deactivated_missing=deactivate,
            )
        else:
            tickers = [t.strip() for t in settings.tickers_universe.split(",") if t.strip()]
            all_constituents = []
            for ticker in tickers:
                rows = client.fetch_block(
                    f"securities/{ticker}.json",
                    "description",
                )
                name = ticker
                for row in rows:
                    if str(row.get("name", "")).upper() == "SHORTNAME":
                        name = str(row.get("value", ticker))
                        break
                all_constituents.append(ParsedConstituent(ticker=ticker, name=name))

            with session_factory() as session:
                added = upsert_securities(
                    session,
                    all_constituents,
                    deactivate_missing=False,
                    alias_seed=TICKER_ALIASES,
                )
                session.commit()
            journal.add_records(added)


def sync_candles(
    client: MoexIssClient,
    session_factory: sessionmaker[Session],
    settings: IngestorSettings,
) -> None:
    """Синхронизировать дневные свечи для всех тикеров коллекции.

    Для каждого тикера определяет дату начала: last_candle_date + 1 день
    или отсутствие параметра from= (вся история) при первом запуске.
    Heartbeat обновляется после каждого тикера.

    Args:
        client: Клиент MOEX ISS.
        session_factory: Фабрика синхронных сессий.
        settings: Конфигурация ingestor.
    """
    with collector_run(session_factory, "moex_candles") as journal:
        with session_factory() as session:
            tickers = collection_tickers(session)

        for ticker, security_id in tickers:
            with session_factory() as session:
                from_date = last_candle_date(session, security_id)

            params: dict[str, str | int] = {}
            if from_date is not None:
                next_date = from_date + timedelta(days=1)
                params["from"] = next_date.isoformat()

            log.info("candles_sync_ticker", ticker=ticker, from_date=from_date)

            path = _CANDLES_PATH_TEMPLATE.format(ticker=ticker)
            rows = client.fetch_block_paginated(path, "history", params)

            candles = [c for r in rows if (c := parse_candle(r)) is not None]

            if candles:
                with session_factory() as session:
                    added = upsert_candles(session, security_id, candles)
                    session.commit()
                journal.add_records(added)

            heartbeat.touch(settings.heartbeat_path)

        log.info("candles_sync_done", total_records=journal.records_added)


def sync_index(
    client: MoexIssClient,
    session_factory: sessionmaker[Session],
    settings: IngestorSettings,
) -> None:
    """Синхронизировать значения индекса IMOEX.

    Args:
        client: Клиент MOEX ISS.
        session_factory: Фабрика синхронных сессий.
        settings: Конфигурация ingestor.
    """
    with collector_run(session_factory, "moex_index") as journal:
        with session_factory() as session:
            from_date = last_index_date(session, "IMOEX")

        params: dict[str, str | int] = {}
        if from_date is not None:
            next_date = from_date + timedelta(days=1)
            params["from"] = next_date.isoformat()

        rows = client.fetch_block_paginated(_IMOEX_INDEX_PATH, "history", params)

        with session_factory() as session:
            added = upsert_index_values(session, "IMOEX", rows)
            session.commit()

        journal.add_records(added)
        log.info("index_sync_done", records=added)


def sync_dividends(
    client: MoexIssClient,
    session_factory: sessionmaker[Session],
    settings: IngestorSettings,
) -> None:
    """Синхронизировать дивиденды для всех тикеров коллекции.

    Строки с неизвестным кодом валюты логируются и помечают запуск PARTIAL.

    Args:
        client: Клиент MOEX ISS.
        session_factory: Фабрика синхронных сессий.
        settings: Конфигурация ingestor.
    """
    with collector_run(session_factory, "moex_dividends") as journal:
        with session_factory() as session:
            tickers = collection_tickers(session)

        for ticker, security_id in tickers:
            rows = client.fetch_block(f"securities/{ticker}/dividends.json", "dividends")
            dividends = [parse_dividend(ticker, r) for r in rows]

            unknown = [d for d in dividends if d.currency is None]
            if unknown:
                log.warning(
                    "unknown_currency_in_dividends",
                    ticker=ticker,
                    count=len(unknown),
                )
                journal.mark_partial(f"unknown currency for {ticker}: {len(unknown)} rows skipped")

            with session_factory() as session:
                added = upsert_dividends(session, security_id, dividends)
                session.commit()

            journal.add_records(added)

        log.info("dividends_sync_done", total_records=journal.records_added)


def sync_splits(
    client: MoexIssClient,
    session_factory: sessionmaker[Session],
    settings: IngestorSettings,
) -> None:
    """Синхронизировать сплиты — полный снимок, фильтр до известных тикеров.

    Args:
        client: Клиент MOEX ISS.
        session_factory: Фабрика синхронных сессий.
        settings: Конфигурация ingestor.
    """
    with collector_run(session_factory, "moex_splits") as journal:
        rows = client.fetch_block("statistics/engines/stock/splits.json", "splits")
        splits = [parse_split(r) for r in rows]

        with session_factory() as session:
            known = get_known_tickers(session)
            added = upsert_splits(session, splits, known)
            session.commit()

        journal.add_records(added)
        log.info("splits_sync_done", records=added)


def _fetch_watchlist_extras(
    client: MoexIssClient,
    wl_tickers: set[str],
    index_tickers: set[str],
) -> list[ParsedConstituent]:
    """Запросить MOEX /securities/{ticker}.json для вотчлист-тикеров вне индекса.

    Тикер с HTTP 404 или пустым ответом логируется как warning и пропускается —
    он не попадёт в БД, но запуск sync_securities не считается сбоем.

    Args:
        client: Клиент MOEX ISS.
        wl_tickers: Тикеры из вотчлиста.
        index_tickers: Тикеры, уже присутствующие в составе индекса.

    Returns:
        Список ParsedConstituent для тикеров вне индекса.
    """
    extras: list[ParsedConstituent] = []
    for ticker in sorted(wl_tickers - index_tickers):
        try:
            desc_rows = client.fetch_block(
                f"securities/{ticker}.json",
                "description",
            )
        except Exception:
            log.warning("watchlist_ticker_fetch_failed", ticker=ticker)
            continue

        if not desc_rows:
            log.warning("watchlist_ticker_not_found_on_moex", ticker=ticker)
            continue

        name = ticker
        for row in desc_rows:
            if str(row.get("name", "")).upper() == "SHORTNAME":
                name = str(row.get("value", ticker))
                break

        extras.append(ParsedConstituent(ticker=ticker, name=name))
        log.info("watchlist_ticker_materialized", ticker=ticker, name=name)

    return extras


def _portfolio_held_tickers(session: Session) -> set[str]:
    """Вернуть тикеры бумаг, удерживаемых в портфеле (через JOIN с securities).

    Args:
        session: Синхронная SQLAlchemy-сессия.

    Returns:
        Множество тикеров.
    """
    rows = (
        session.execute(
            select(Security.ticker).join(
                PortfolioPosition, PortfolioPosition.security_id == Security.id
            )
        )
        .scalars()
        .all()
    )
    return {str(r) for r in rows}


def run_all_collectors(
    client: MoexIssClient,
    session_factory: sessionmaker[Session],
    settings: IngestorSettings,
) -> None:
    """Запустить все MOEX-сборщики последовательно.

    Сбой одного сборщика не прерывает остальные — каждый обёрнут
    в collector_run, который подавляет исключения и пишет статус FAILED.
    Новостные и CBR-сборщики оркестрирует run_backfill в backfill.py.

    Args:
        client: Клиент MOEX ISS.
        session_factory: Фабрика синхронных сессий.
        settings: Конфигурация ingestor.
    """
    sync_securities(client, session_factory, settings)
    sync_candles(client, session_factory, settings)
    sync_index(client, session_factory, settings)
    sync_dividends(client, session_factory, settings)
    sync_splits(client, session_factory, settings)
