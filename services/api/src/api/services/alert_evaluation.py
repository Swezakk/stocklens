"""Сервис оценки алертов: определяет, какие подписки сработали с момента последней проверки.

Поддерживаемые виды: price_level, sentiment_spike, dividend_upcoming, volatility_regime.
"""

from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from typing import Protocol

import structlog
from stocklens_core.enums import AlertKind, SentimentLabel
from stocklens_core.models.market import Dividend
from stocklens_core.models.news import NewsArticle, NewsSentiment

from api.core.cache import AlertNxStore
from api.core.exceptions import InsufficientHistoryError, ModelNotLoadedError
from api.repositories.protocols import BotSubscriptionRepository, SecurityRepository
from api.schemas.bot import DIVIDEND_LEAD_DAYS_DEFAULT, PendingAlertOut
from api.schemas.predict import VolatilityRegime

logger = structlog.get_logger(__name__)

TodayProvider = Callable[[], date]

_DEDUP_TTL_PRICE_SECONDS = 86_400
_DEDUP_TTL_SENTIMENT_SECONDS = 259_200
_DEDUP_TTL_DIGEST_SECONDS = 72_000
_DEDUP_TTL_VOLATILITY_SECONDS = 86_400

_MIN_CLOSES_FOR_LEVEL_CHECK = 2


class CloseRepository(Protocol):
    """Последние два дневных закрытия для оценки price_level алертов."""

    async def latest_two_closes(self, security_id: int) -> list[Decimal]:
        """Вернуть последние два close (регулярная сессия), порядок: [prev, last]."""
        ...


class NewsAlertRepository(Protocol):
    """Негативные новости сегодняшнего дня для оценки sentiment_spike алертов."""

    async def list_news_for_alert(
        self,
        security_id: int,
        today_date: date,
    ) -> list[tuple[NewsArticle, NewsSentiment | None]]:
        """Вернуть статьи с sentiment=NEGATIVE за дату today_date."""
        ...


class DividendAlertRepository(Protocol):
    """Дивиденды в заданном диапазоне дат для оценки dividend_upcoming алертов."""

    async def list_upcoming(
        self,
        security_id: int,
        date_from: date,
        date_to: date,
    ) -> list[Dividend]:
        """Вернуть дивиденды с ex_date в диапазоне [date_from, date_to]."""
        ...


class VolatilityRegimeAssessor(Protocol):
    """Оценщик режима волатильности для volatility_regime алертов."""

    async def assess_volatility_regime(
        self, ticker: str, quantile: float, lookback: int
    ) -> VolatilityRegime:
        """Вернуть оценку режима волатильности по тикеру."""
        ...


class AlertEvaluationService:
    """Оценивает все активные подписки и возвращает список сработавших алертов.

    Использует Redis NX-дедупликацию для предотвращения повторной отправки одного алерта.
    При недоступности Redis работает в fail-open режиме (алерт проходит).
    """

    def __init__(
        self,
        bot_repo: BotSubscriptionRepository,
        security_repo: SecurityRepository,
        close_repo: CloseRepository,
        news_repo: NewsAlertRepository,
        dividend_repo: DividendAlertRepository,
        redis: AlertNxStore,
        today: TodayProvider,
        assessor: VolatilityRegimeAssessor | None = None,
        volatility_quantile: float = 0.80,
        volatility_lookback: int = 252,
    ) -> None:
        self._bot_repo = bot_repo
        self._security_repo = security_repo
        self._close_repo = close_repo
        self._news_repo = news_repo
        self._dividend_repo = dividend_repo
        self._redis = redis
        self._today = today
        self._assessor = assessor
        self._volatility_quantile = volatility_quantile
        self._volatility_lookback = volatility_lookback

    async def collect_pending(self) -> list[PendingAlertOut]:
        """Оценить все активные подписки и вернуть список сработавших алертов.

        Подписки с тикером, не найденным в БД, логируются как предупреждение и пропускаются.
        Volatility_regime пропускается, если assessor не передан (ML-слой не сконфигурирован).
        """
        today = self._today()
        subscriptions = await self._bot_repo.list_all_active()
        pending: list[PendingAlertOut] = []

        for sub in subscriptions:
            if sub.kind == AlertKind.VOLATILITY_REGIME and self._assessor is None:
                continue

            raw_ticker = sub.params.get("ticker")
            if not isinstance(raw_ticker, str):
                logger.warning(
                    "alert_subscription_missing_ticker",
                    sub_id=sub.id,
                    kind=sub.kind,
                )
                continue

            security = await self._security_repo.get_by_ticker(raw_ticker)
            if security is None:
                logger.warning(
                    "alert_subscription_unknown_ticker",
                    sub_id=sub.id,
                    kind=sub.kind,
                    ticker=raw_ticker,
                )
                continue

            alerts: list[PendingAlertOut] = []
            sid = security.id
            if sub.kind == AlertKind.PRICE_LEVEL:
                alerts = await self._evaluate_price_level(
                    sub.chat_id, raw_ticker, sid, sub.params, today
                )
            elif sub.kind == AlertKind.SENTIMENT_SPIKE:
                alerts = await self._evaluate_sentiment_spike(sub.chat_id, raw_ticker, sid, today)
            elif sub.kind == AlertKind.DIVIDEND_UPCOMING:
                alerts = await self._evaluate_dividend_upcoming(
                    sub.chat_id, raw_ticker, sid, sub.params, today
                )
            elif sub.kind == AlertKind.VOLATILITY_REGIME:
                alerts = await self._evaluate_volatility_regime(
                    sub.chat_id, raw_ticker, sid, sub.params, today
                )

            pending.extend(alerts)

        return pending

    async def digest_claim(self, for_date: date) -> bool:
        """Атомарно зарезервировать дайджест для даты. Возвращает True при первом вызове.

        Последующие вызовы того же дня возвращают False (дайджест уже отправлен).
        При недоступности Redis возвращает True (fail-open).
        """
        key = f"digest:{for_date.isoformat()}"
        return await self._redis.set_nx(key, _DEDUP_TTL_DIGEST_SECONDS)

    async def _evaluate_price_level(
        self,
        chat_id: int,
        ticker: str,
        security_id: int,
        params: dict[str, object],
        today: date,
    ) -> list[PendingAlertOut]:
        raw_level = params.get("level")
        if not isinstance(raw_level, int | float):
            return []
        level = Decimal(str(raw_level))

        closes = await self._close_repo.latest_two_closes(security_id)
        if len(closes) < _MIN_CLOSES_FOR_LEVEL_CHECK:
            return []

        prev_close, last_close = closes[-2], closes[-1]
        lo = min(prev_close, last_close)
        hi = max(prev_close, last_close)
        if not (lo <= level <= hi):
            return []

        key = f"alert:price:{chat_id}:{ticker}:{raw_level}:{today.isoformat()}"
        if not await self._redis.set_nx(key, _DEDUP_TTL_PRICE_SECONDS):
            return []

        return [
            PendingAlertOut(
                chat_id=chat_id,
                kind=AlertKind.PRICE_LEVEL,
                ticker=ticker,
                level=level,
                close=last_close,
            )
        ]

    async def _evaluate_sentiment_spike(
        self,
        chat_id: int,
        ticker: str,
        security_id: int,
        today: date,
    ) -> list[PendingAlertOut]:
        rows = await self._news_repo.list_news_for_alert(security_id, today)
        alerts: list[PendingAlertOut] = []

        for article, sentiment in rows:
            if sentiment is None or sentiment.label != SentimentLabel.NEGATIVE:
                continue

            key = f"alert:sent:{chat_id}:{ticker}:{article.id}"
            if not await self._redis.set_nx(key, _DEDUP_TTL_SENTIMENT_SECONDS):
                continue

            alerts.append(
                PendingAlertOut(
                    chat_id=chat_id,
                    kind=AlertKind.SENTIMENT_SPIKE,
                    ticker=ticker,
                    article_id=article.id,
                    article_title=article.title,
                    article_url=article.url,
                    article_published_at=article.published_at,
                )
            )

        return alerts

    async def _evaluate_dividend_upcoming(
        self,
        chat_id: int,
        ticker: str,
        security_id: int,
        params: dict[str, object],
        today: date,
    ) -> list[PendingAlertOut]:
        raw_lead = params.get("lead_days", DIVIDEND_LEAD_DAYS_DEFAULT)
        if not isinstance(raw_lead, int):
            return []
        lead_days = raw_lead

        date_from = today
        date_to = today + timedelta(days=lead_days)
        dividends = await self._dividend_repo.list_upcoming(security_id, date_from, date_to)
        alerts: list[PendingAlertOut] = []

        for dividend in dividends:
            key = f"alert:div:{chat_id}:{ticker}:{dividend.ex_date.isoformat()}"
            ttl = max(1, (dividend.ex_date - today).days + 1) * 86_400
            if not await self._redis.set_nx(key, ttl):
                continue

            alerts.append(
                PendingAlertOut(
                    chat_id=chat_id,
                    kind=AlertKind.DIVIDEND_UPCOMING,
                    ticker=ticker,
                    ex_date=dividend.ex_date,
                    dividend_value=dividend.value,
                    dividend_currency=dividend.currency,
                )
            )

        return alerts

    async def _evaluate_volatility_regime(
        self,
        chat_id: int,
        ticker: str,
        security_id: int,
        params: dict[str, object],
        today: date,
    ) -> list[PendingAlertOut]:
        assert self._assessor is not None  # гарантировано caller'ом в collect_pending

        raw_quantile = params.get("quantile", self._volatility_quantile)
        quantile = (
            float(raw_quantile)
            if isinstance(raw_quantile, int | float)
            else self._volatility_quantile
        )

        raw_lookback = params.get("lookback", self._volatility_lookback)
        lookback = int(raw_lookback) if isinstance(raw_lookback, int) else self._volatility_lookback

        try:
            regime = await self._assessor.assess_volatility_regime(ticker, quantile, lookback)
        except (ModelNotLoadedError, InsufficientHistoryError) as exc:
            logger.warning(
                "volatility_regime_skipped",
                ticker=ticker,
                reason=str(exc),
            )
            return []

        if not regime.is_elevated:
            return []

        key = f"alert:vol:{chat_id}:{ticker}:{today.isoformat()}"
        if not await self._redis.set_nx(key, _DEDUP_TTL_VOLATILITY_SECONDS):
            return []

        return [
            PendingAlertOut(
                chat_id=chat_id,
                kind=AlertKind.VOLATILITY_REGIME,
                ticker=ticker,
                volatility=regime.volatility,
                threshold=regime.threshold,
                regime_quantile=regime.quantile,
            )
        ]
