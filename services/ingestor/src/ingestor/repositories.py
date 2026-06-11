"""Репозитории для upsert рыночных данных в PostgreSQL.

Используют синхронную SQLAlchemy 2.0 с INSERT ... ON CONFLICT DO UPDATE
(postgresql dialect) — идемпотентность всех операций записи гарантирована.
"""

from datetime import date

import structlog
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from stocklens_core.models.market import Candle, Dividend, IndexValue, Security, Split
from stocklens_core.models.portfolio import PortfolioPosition

from ingestor.parsing import (
    ParsedCandle,
    ParsedConstituent,
    ParsedDividend,
    ParsedSplit,
)

log = structlog.get_logger(__name__)


def upsert_securities(
    session: Session,
    items: list[ParsedConstituent],
    deactivate_missing: bool,
) -> int:
    """Синхронизировать список ценных бумаг с таблицей securities.

    Для каждого тикера: обновляет name и board, выставляет is_active=True.
    Деактивация (is_active=False) бумаг, отсутствующих в списке, выполняется
    только при deactivate_missing=True — вызывающий устанавливает этот флаг
    лишь после полного успешного получения состава индекса (len >= 30).

    Args:
        session: Синхронная SQLAlchemy-сессия.
        items: Список распарсенных компонентов индекса.
        deactivate_missing: Деактивировать бумаги, не вошедшие в список.

    Returns:
        Количество затронутых строк.
    """
    if not items:
        return 0

    tickers = [item.ticker for item in items]

    values = [
        {"ticker": item.ticker, "name": item.name, "board": "TQBR", "is_active": True}
        for item in items
    ]
    stmt = insert(Security).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker"],
        set_={"name": stmt.excluded.name, "board": stmt.excluded.board, "is_active": True},
    )
    session.execute(stmt)

    if deactivate_missing:
        session.execute(
            update(Security).where(Security.ticker.notin_(tickers)).values(is_active=False)
        )
        log.info("securities_deactivated", active_tickers=tickers)

    return len(values)


def upsert_candles(
    session: Session,
    security_id: int,
    candles: list[ParsedCandle],
) -> int:
    """Upsert дневных свечей для одного инструмента.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        security_id: Идентификатор инструмента в БД.
        candles: Список распарсенных свечей.

    Returns:
        Количество затронутых строк.
    """
    if not candles:
        return 0

    values = [
        {
            "security_id": security_id,
            "trade_date": c.trade_date,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "value": c.value,
            "is_weekend_session": c.is_weekend_session,
        }
        for c in candles
    ]
    stmt = insert(Candle).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["security_id", "trade_date"],
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
            "value": stmt.excluded.value,
            "is_weekend_session": stmt.excluded.is_weekend_session,
        },
    )
    session.execute(stmt)
    return len(values)


def upsert_index_values(
    session: Session,
    index_code: str,
    rows: list[dict[str, object]],
) -> int:
    """Upsert значений биржевого индекса.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        index_code: Код индекса (например «IMOEX»).
        rows: Строки из блока history ISS (уже zip-ованные в dict).

    Returns:
        Количество затронутых строк.
    """
    if not rows:
        return 0

    values = [
        {
            "index_code": index_code,
            "trade_date": date.fromisoformat(str(r["TRADEDATE"])),
            "close": r["CLOSE"],
        }
        for r in rows
        if r.get("CLOSE") is not None
    ]
    if not values:
        return 0

    stmt = insert(IndexValue).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["index_code", "trade_date"],
        set_={"close": stmt.excluded.close},
    )
    session.execute(stmt)
    return len(values)


def upsert_dividends(
    session: Session,
    security_id: int,
    dividends: list[ParsedDividend],
) -> int:
    """Upsert дивидендных выплат. Строки с currency=None пропускаются (caller логирует).

    Args:
        session: Синхронная SQLAlchemy-сессия.
        security_id: Идентификатор инструмента в БД.
        dividends: Список распарсенных дивидендов (currency=None → пропуск).

    Returns:
        Количество затронутых строк.
    """
    valid = [d for d in dividends if d.currency is not None]
    if not valid:
        return 0

    values = [
        {
            "security_id": security_id,
            "ex_date": d.ex_date,
            "value": d.value,
            "currency": d.currency,
        }
        for d in valid
    ]
    stmt = insert(Dividend).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["security_id", "ex_date"],
        set_={"value": stmt.excluded.value, "currency": stmt.excluded.currency},
    )
    session.execute(stmt)
    return len(values)


def upsert_splits(
    session: Session,
    splits: list[ParsedSplit],
    known_tickers: set[str],
) -> int:
    """Upsert сплитов — только для тикеров из known_tickers.

    Endpoint /splits возвращает весь рынок; фильтруем до известных бумаг
    чтобы не нарушить FK securities.id.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        splits: Все сплиты с MOEX.
        known_tickers: Множество тикеров, присутствующих в БД.

    Returns:
        Количество затронутых строк.
    """
    filtered = [s for s in splits if s.ticker in known_tickers]
    if not filtered:
        return 0

    ticker_to_id: dict[str, int] = {
        row.ticker: row.id
        for row in session.execute(
            select(Security.ticker, Security.id).where(Security.ticker.in_(known_tickers))
        ).all()
    }

    values = [
        {
            "security_id": ticker_to_id[s.ticker],
            "split_date": s.split_date,
            "before": s.before,
            "after": s.after,
        }
        for s in filtered
        if s.ticker in ticker_to_id
    ]
    if not values:
        return 0

    stmt = insert(Split).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["security_id", "split_date"],
        set_={"before": stmt.excluded.before, "after": stmt.excluded.after},
    )
    session.execute(stmt)
    return len(values)


def last_candle_date(session: Session, security_id: int) -> date | None:
    """Найти дату последней свечи для инструмента.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        security_id: Идентификатор инструмента.

    Returns:
        Дата последней свечи или None если данных ещё нет.
    """
    row = session.execute(
        select(Candle.trade_date)
        .where(Candle.security_id == security_id)
        .order_by(Candle.trade_date.desc())
        .limit(1)
    ).scalar_one_or_none()
    return row


def last_index_date(session: Session, index_code: str) -> date | None:
    """Найти дату последнего значения индекса.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        index_code: Код индекса.

    Returns:
        Дата последнего значения или None.
    """
    row = session.execute(
        select(IndexValue.trade_date)
        .where(IndexValue.index_code == index_code)
        .order_by(IndexValue.trade_date.desc())
        .limit(1)
    ).scalar_one_or_none()
    return row


def collection_tickers(session: Session) -> list[tuple[str, int]]:
    """Вернуть тикеры для сбора: активные бумаги UNION бумаги из портфеля.

    Бумага, вышедшая из индекса (is_active=False), но удерживаемая в портфеле,
    продолжает собираться для корректного расчёта P&L.

    Args:
        session: Синхронная SQLAlchemy-сессия.

    Returns:
        Список пар (ticker, security_id).
    """
    active_q = select(Security.ticker, Security.id).where(Security.is_active.is_(True))

    portfolio_q = (
        select(Security.ticker, Security.id)
        .join(PortfolioPosition, PortfolioPosition.security_id == Security.id)
        .where(Security.is_active.is_(False))
    )

    rows = session.execute(active_q.union(portfolio_q)).all()
    return [(str(r[0]), int(r[1])) for r in rows]


def get_known_tickers(session: Session) -> set[str]:
    """Получить множество всех тикеров, зарегистрированных в БД.

    Args:
        session: Синхронная SQLAlchemy-сессия.

    Returns:
        Множество строковых тикеров.
    """
    rows = session.execute(select(Security.ticker)).scalars().all()
    return set(rows)


def get_security_id(session: Session, ticker: str) -> int | None:
    """Найти security_id по тикеру.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        ticker: Тикер инструмента.

    Returns:
        ID записи или None если тикер не найден.
    """
    return session.execute(
        select(Security.id).where(Security.ticker == ticker)
    ).scalar_one_or_none()
