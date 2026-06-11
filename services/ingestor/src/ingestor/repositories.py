"""Репозитории для upsert рыночных данных в PostgreSQL.

Используют синхронную SQLAlchemy 2.0 с INSERT ... ON CONFLICT DO UPDATE
(postgresql dialect) — идемпотентность всех операций записи гарантирована.
"""

from datetime import date, datetime

import structlog
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from stocklens_core.enums import Currency, SentimentLabel
from stocklens_core.models.market import (
    Candle,
    CurrencyRate,
    Dividend,
    IndexValue,
    KeyRate,
    Security,
    Split,
)
from stocklens_core.models.news import NewsArticle, NewsSentiment, NewsTicker
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
    alias_seed: dict[str, list[str]] | None = None,
) -> int:
    """Синхронизировать список ценных бумаг с таблицей securities.

    Для каждого тикера: обновляет name и board, выставляет is_active=True.
    Деактивация (is_active=False) бумаг, отсутствующих в списке, выполняется
    только при deactivate_missing=True — вызывающий устанавливает этот флаг
    лишь после полного успешного получения состава индекса (len >= 30).

    Если передан alias_seed, псевдонимы мержатся с уже сохранёнными через union —
    существующие вручную добавленные псевдонимы никогда не удаляются.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        items: Список распарсенных компонентов индекса.
        deactivate_missing: Деактивировать бумаги, не вошедшие в список.
        alias_seed: Словарь тикер → список псевдонимов для мержа.

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

    if alias_seed:
        _merge_aliases(session, alias_seed)

    if deactivate_missing:
        session.execute(
            update(Security).where(Security.ticker.notin_(tickers)).values(is_active=False)
        )
        log.info("securities_deactivated", active_tickers=tickers)

    return len(values)


def _merge_aliases(session: Session, alias_seed: dict[str, list[str]]) -> None:
    """Смержить seed-псевдонимы с уже сохранёнными в БД через union.

    Существующие псевдонимы (добавленные вручную или ранее) никогда не удаляются.
    Порядок в результирующем массиве не гарантируется, но уникальность обеспечена.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        alias_seed: Словарь тикер → список новых псевдонимов для добавления.
    """
    seed_tickers = list(alias_seed.keys())
    rows = session.execute(
        select(Security.ticker, Security.aliases).where(Security.ticker.in_(seed_tickers))
    ).all()

    for row in rows:
        ticker: str = str(row[0])
        # cast: JSONB возвращает list[str] на Python-стороне — проверяем тип явно
        existing: list[str] = row[1] if isinstance(row[1], list) else []
        seed_list = alias_seed.get(ticker, [])
        merged = list(dict.fromkeys(existing + seed_list))  # union с сохранением порядка
        session.execute(update(Security).where(Security.ticker == ticker).values(aliases=merged))


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


def get_alias_index(session: Session) -> dict[str, str]:
    """Построить индекс псевдоним→тикер для всех активных бумаг.

    Включает сам тикер как псевдоним (латиница, для прямых упоминаний в тексте).
    Используется для сопоставления упоминаний новостей с тикерами.

    Args:
        session: Синхронная SQLAlchemy-сессия.

    Returns:
        Словарь {псевдоним_в_нижнем_регистре: тикер}.
    """
    rows = session.execute(
        select(Security.ticker, Security.aliases).where(Security.is_active.is_(True))
    ).all()

    index: dict[str, str] = {}
    for row in rows:
        ticker: str = str(row[0])
        # JSONB может вернуть None для строк созданных до server_default
        aliases: list[str] = row[1] if isinstance(row[1], list) else []
        index[ticker.lower()] = ticker
        for alias in aliases:
            index[alias.lower()] = ticker
    return index


def insert_news_articles_returning_new(
    session: Session,
    articles: list[dict[str, object]],
) -> list[int]:
    """Вставить новостные статьи, вернуть id только новых строк.

    ON CONFLICT (url) DO NOTHING — повторный url не обновляется и не возвращается,
    что позволяет скорить sentiment только для свежих публикаций.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        articles: Список словарей с полями source/url/title/summary/published_at.

    Returns:
        Список id вставленных (новых) строк.
    """
    if not articles:
        return []

    stmt = (
        insert(NewsArticle)
        .values(articles)
        .on_conflict_do_nothing(index_elements=["url"])
        .returning(NewsArticle.id)
    )
    result = session.execute(stmt)
    return [int(row[0]) for row in result]


def upsert_news_sentiment(
    session: Session,
    article_id: int,
    label: SentimentLabel,
    score: float,
    model_version: str,
) -> None:
    """Upsert результата sentiment-классификации для статьи.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        article_id: ID статьи.
        label: Тональность.
        score: Уверенность модели (0..1).
        model_version: Строковый идентификатор модели.
    """
    stmt = insert(NewsSentiment).values(
        article_id=article_id,
        label=label,
        score=score,
        model_version=model_version,
    )
    stmt = stmt.on_conflict_do_nothing(index_elements=["article_id"])
    session.execute(stmt)


def insert_news_tickers(
    session: Session,
    article_id: int,
    security_ids: list[int],
) -> None:
    """Связать статью с упомянутыми инструментами.

    ON CONFLICT DO NOTHING — идемпотентно при повторном вызове.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        article_id: ID статьи.
        security_ids: Список id инструментов.
    """
    if not security_ids:
        return

    values = [{"article_id": article_id, "security_id": sid} for sid in security_ids]
    stmt = insert(NewsTicker).values(values).on_conflict_do_nothing()
    session.execute(stmt)


def upsert_currency_rates(
    session: Session,
    rates: list[dict[str, object]],
) -> int:
    """Upsert курсов валют ЦБ РФ.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        rates: Список словарей с полями currency/rate_date/rate.

    Returns:
        Количество затронутых строк.
    """
    if not rates:
        return 0

    stmt = insert(CurrencyRate).values(rates)
    stmt = stmt.on_conflict_do_update(
        index_elements=["currency", "rate_date"],
        set_={"rate": stmt.excluded.rate},
    )
    session.execute(stmt)
    return len(rates)


def upsert_key_rates(
    session: Session,
    rates: list[dict[str, object]],
) -> int:
    """Upsert ключевых ставок ЦБ РФ.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        rates: Список словарей с полями rate_date/rate.

    Returns:
        Количество затронутых строк.
    """
    if not rates:
        return 0

    stmt = insert(KeyRate).values(rates)
    stmt = stmt.on_conflict_do_update(
        index_elements=["rate_date"],
        set_={"rate": stmt.excluded.rate},
    )
    session.execute(stmt)
    return len(rates)


def last_currency_rate_date(session: Session, currency: Currency) -> date | None:
    """Найти дату последнего курса для валюты.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        currency: Код валюты.

    Returns:
        Дата последнего курса или None.
    """
    return session.execute(
        select(CurrencyRate.rate_date)
        .where(CurrencyRate.currency == currency)
        .order_by(CurrencyRate.rate_date.desc())
        .limit(1)
    ).scalar_one_or_none()


def last_key_rate_date(session: Session) -> date | None:
    """Найти дату последней ключевой ставки.

    Args:
        session: Синхронная SQLAlchemy-сессия.

    Returns:
        Дата последней ставки или None.
    """
    return session.execute(
        select(KeyRate.rate_date).order_by(KeyRate.rate_date.desc()).limit(1)
    ).scalar_one_or_none()


def get_security_ids_by_tickers(session: Session, tickers: list[str]) -> dict[str, int]:
    """Получить словарь тикер→id для переданных тикеров.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        tickers: Список тикеров.

    Returns:
        Словарь {тикер: security_id}.
    """
    if not tickers:
        return {}

    rows = session.execute(
        select(Security.ticker, Security.id).where(Security.ticker.in_(tickers))
    ).all()
    return {str(r[0]): int(r[1]) for r in rows}


def get_article_published_at(session: Session, article_id: int) -> datetime | None:
    """Получить дату публикации статьи по id.

    Args:
        session: Синхронная SQLAlchemy-сессия.
        article_id: ID статьи.

    Returns:
        Дата публикации или None если не найдена.
    """
    return session.execute(
        select(NewsArticle.published_at).where(NewsArticle.id == article_id)
    ).scalar_one_or_none()
