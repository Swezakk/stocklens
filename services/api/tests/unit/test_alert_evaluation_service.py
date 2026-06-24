"""Unit-тесты AlertEvaluationService и расширенной валидации параметров подписок."""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import cast

import pytest
from api.core.cache import AlertNxStore
from api.core.exceptions import (
    InsufficientDataError,
    InsufficientHistoryError,
    InvalidAlertParamsError,
    ModelNotLoadedError,
)
from api.repositories.protocols import BotSubscriptionRepository, SecurityRepository
from api.schemas.bot import SubscriptionIn
from api.schemas.predict import VolatilityRegime
from api.services.alert_evaluation import (
    AlertEvaluationService,
    CloseRepository,
    DividendAlertRepository,
    NewsAlertRepository,
    TodayProvider,
    VolatilityRegimeAssessor,
)
from api.services.bot import BotSubscriptionService
from stocklens_core.enums import AlertKind, Currency, SentimentLabel
from stocklens_core.models.market import Dividend, Security
from stocklens_core.models.news import NewsArticle, NewsSentiment
from stocklens_core.models.portfolio import BotSubscription
from structlog.testing import capture_logs


@dataclass
class FakeSecurity:
    id: int
    ticker: str
    name: str = "Test Co"
    board: str = "TQBR"
    aliases: list[str] = field(default_factory=list)
    is_active: bool = True


@dataclass
class FakeSubscription:
    id: int
    chat_id: int
    kind: AlertKind
    params: dict[str, object] = field(default_factory=dict)


@dataclass
class FakeArticle:
    id: int
    source: str
    url: str
    title: str
    published_at: datetime
    summary: str | None = None


@dataclass
class FakeSentiment:
    id: int
    article_id: int
    label: SentimentLabel
    score: float
    model_version: str


@dataclass
class FakeDividend:
    id: int
    security_id: int
    ex_date: date
    value: Decimal
    currency: Currency = Currency.RUB


def _fake_security(ticker: str = "SBER", sec_id: int = 1) -> Security:
    return cast(Security, FakeSecurity(id=sec_id, ticker=ticker))


def _fake_sub(
    sub_id: int,
    chat_id: int,
    kind: AlertKind,
    params: dict[str, object],
) -> BotSubscription:
    return cast(
        BotSubscription,
        FakeSubscription(id=sub_id, chat_id=chat_id, kind=kind, params=params),
    )


def _fake_article(
    article_id: int,
    published_at: datetime,
    label: SentimentLabel = SentimentLabel.NEGATIVE,
    title: str = "Плохие новости",
) -> tuple[NewsArticle, NewsSentiment | None]:
    article = cast(
        NewsArticle,
        FakeArticle(
            id=article_id,
            source="rbc",
            url=f"https://rbc.ru/{article_id}",
            title=title,
            published_at=published_at,
        ),
    )
    sentiment = cast(
        NewsSentiment,
        FakeSentiment(
            id=article_id * 10,
            article_id=article_id,
            label=label,
            score=0.95,
            model_version="rubert-tiny2-v1",
        ),
    )
    return article, sentiment


def _fake_dividend(ex_date: date) -> Dividend:
    return cast(
        Dividend,
        FakeDividend(
            id=1,
            security_id=1,
            ex_date=ex_date,
            value=Decimal("33.00"),
            currency=Currency.RUB,
        ),
    )


class FakeSecurityRepo:
    def __init__(self, securities: dict[str, Security] | None = None) -> None:
        self._securities = securities or {}

    async def get_by_ticker(self, ticker: str) -> Security | None:
        return self._securities.get(ticker)

    async def list_securities(
        self, is_active: bool | None, limit: int, offset: int
    ) -> tuple[list[Security], int]:
        return [], 0


class FakeBotRepo:
    def __init__(self, subs: list[BotSubscription] | None = None) -> None:
        self._subs = subs or []

    async def list_by_chat(self, chat_id: int) -> list[BotSubscription]:
        return [s for s in self._subs if s.chat_id == chat_id]

    async def list_all_active(self) -> list[BotSubscription]:
        return list(self._subs)

    async def create(
        self, chat_id: int, kind: AlertKind, params: dict[str, object]
    ) -> BotSubscription:
        sub = cast(
            BotSubscription,
            FakeSubscription(id=99, chat_id=chat_id, kind=kind, params=params),
        )
        self._subs.append(sub)
        return sub

    async def delete(self, sub_id: int) -> bool:
        return False


class FakeRedis:
    """Fake Redis с честной NX-семантикой для unit-тестов дедупликации."""

    def __init__(self, down: bool = False) -> None:
        self._store: dict[str, str] = {}
        self._down = down

    async def set_nx(self, key: str, ttl_seconds: int) -> bool:
        if self._down:
            return True
        if key in self._store:
            return False
        self._store[key] = "1"
        return True


class _FakeCloseRepo:
    """Фейк: последние два закрытия по security_id."""

    def __init__(self, closes: dict[int, list[Decimal]] | None = None) -> None:
        self._closes = closes or {}

    async def latest_two_closes(self, security_id: int) -> list[Decimal]:
        return self._closes.get(security_id, [])


class _FakeNewsRepo:
    """Фейк: список (статья, тональность) для алертов."""

    def __init__(
        self,
        rows: list[tuple[NewsArticle, NewsSentiment | None]] | None = None,
    ) -> None:
        self._rows = rows or []

    async def list_news_for_alert(
        self, security_id: int, today_date: date
    ) -> list[tuple[NewsArticle, NewsSentiment | None]]:
        return self._rows


class _FakeDividendRepo:
    """Фейк: список дивидендов для алертов."""

    def __init__(self, dividends: dict[int, list[Dividend]] | None = None) -> None:
        self._dividends = dividends or {}

    async def list_upcoming(
        self, security_id: int, date_from: date, date_to: date
    ) -> list[Dividend]:
        return self._dividends.get(security_id, [])


def _sub_service(
    subs: list[BotSubscription] | None = None,
    securities: dict[str, Security] | None = None,
) -> BotSubscriptionService:
    return BotSubscriptionService(
        repo=cast(BotSubscriptionRepository, FakeBotRepo(subs)),
        security_repo=cast(SecurityRepository, FakeSecurityRepo(securities)),
    )


_TODAY = date(2026, 6, 23)
_TODAY_DT = datetime(2026, 6, 23, 10, 0, tzinfo=UTC)


def _fixed_today() -> date:
    return _TODAY


class _FakeAssessor:
    """Фейк VolatilityRegimeAssessor: заданный VolatilityRegime или исключение."""

    def __init__(
        self,
        regime: VolatilityRegime | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._regime = regime
        self._raises = raises
        self.calls = 0

    async def assess_volatility_regime(
        self, ticker: str, quantile: float, lookback: int
    ) -> VolatilityRegime:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        assert self._regime is not None
        return self._regime


def _fake_regime(
    ticker: str = "SBER",
    volatility: float = 0.05,
    threshold: float = 0.03,
    is_elevated: bool = True,
) -> VolatilityRegime:
    return VolatilityRegime(
        ticker=ticker,
        predicted_for=_TODAY,
        volatility=volatility,
        threshold=threshold,
        is_elevated=is_elevated,
        quantile=0.80,
        lookback=252,
    )


def _eval_service(
    subs: list[BotSubscription],
    securities: dict[str, Security] | None = None,
    closes: dict[int, list[Decimal]] | None = None,
    news_rows: list[tuple[NewsArticle, NewsSentiment | None]] | None = None,
    dividends: dict[int, list[Dividend]] | None = None,
    redis: FakeRedis | None = None,
    today: TodayProvider | None = None,
    assessor: VolatilityRegimeAssessor | None = None,
    volatility_quantile: float = 0.80,
    volatility_lookback: int = 252,
) -> AlertEvaluationService:
    return AlertEvaluationService(
        bot_repo=cast(BotSubscriptionRepository, FakeBotRepo(subs)),
        security_repo=cast(SecurityRepository, FakeSecurityRepo(securities)),
        close_repo=cast(CloseRepository, _FakeCloseRepo(closes)),
        news_repo=cast(NewsAlertRepository, _FakeNewsRepo(news_rows)),
        dividend_repo=cast(DividendAlertRepository, _FakeDividendRepo(dividends)),
        redis=cast(AlertNxStore, redis or FakeRedis()),
        today=today or _fixed_today,
        assessor=cast(VolatilityRegimeAssessor, assessor),
        volatility_quantile=volatility_quantile,
        volatility_lookback=volatility_lookback,
    )


async def test_create_price_level_valid_params_succeeds() -> None:
    """create: price_level с ticker+level создаётся успешно."""
    svc = _sub_service(securities={"SBER": _fake_security("SBER")})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.PRICE_LEVEL,
        params={"ticker": "SBER", "level": 300.0},
    )
    result = await svc.create(data)
    assert result.kind == AlertKind.PRICE_LEVEL
    assert result.params["level"] == 300.0


async def test_create_price_level_without_ticker_raises_422() -> None:
    """create: price_level без 'ticker' → InvalidAlertParamsError 422."""
    svc = _sub_service()
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.PRICE_LEVEL,
        params={"level": 300.0},
    )
    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)
    assert exc_info.value.status == 422
    assert "ticker" in exc_info.value.detail


async def test_create_price_level_without_level_raises_422() -> None:
    """create: price_level без 'level' → InvalidAlertParamsError 422."""
    svc = _sub_service(securities={"SBER": _fake_security("SBER")})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.PRICE_LEVEL,
        params={"ticker": "SBER"},
    )
    with pytest.raises((InvalidAlertParamsError, InsufficientDataError)):
        await svc.create(data)


async def test_create_price_level_level_not_positive_raises_422() -> None:
    """create: price_level с level <= 0 → InvalidAlertParamsError 422."""
    svc = _sub_service(securities={"SBER": _fake_security("SBER")})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.PRICE_LEVEL,
        params={"ticker": "SBER", "level": -1.0},
    )
    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)
    assert exc_info.value.status == 422


async def test_create_price_level_unknown_ticker_raises_422() -> None:
    """create: price_level с тикером не из БД → InvalidAlertParamsError 422."""
    svc = _sub_service(securities={})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.PRICE_LEVEL,
        params={"ticker": "NOPE", "level": 300.0},
    )
    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)
    assert "NOPE" in exc_info.value.detail


async def test_create_sentiment_spike_valid_params_succeeds() -> None:
    """create: sentiment_spike с валидным ticker создаётся успешно."""
    svc = _sub_service(securities={"LKOH": _fake_security("LKOH", sec_id=2)})
    data = SubscriptionIn(
        chat_id=200,
        kind=AlertKind.SENTIMENT_SPIKE,
        params={"ticker": "LKOH"},
    )
    result = await svc.create(data)
    assert result.kind == AlertKind.SENTIMENT_SPIKE


async def test_create_sentiment_spike_without_ticker_raises_422() -> None:
    """create: sentiment_spike без 'ticker' → InvalidAlertParamsError 422."""
    svc = _sub_service()
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.SENTIMENT_SPIKE,
        params={},
    )
    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)
    assert "ticker" in exc_info.value.detail


async def test_create_sentiment_spike_unknown_ticker_raises_422() -> None:
    """create: sentiment_spike с неизвестным ticker → InvalidAlertParamsError 422."""
    svc = _sub_service(securities={})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.SENTIMENT_SPIKE,
        params={"ticker": "GHOST"},
    )
    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)
    assert "GHOST" in exc_info.value.detail


async def test_create_dividend_upcoming_valid_params_succeeds() -> None:
    """create: dividend_upcoming с ticker + lead_days создаётся успешно."""
    svc = _sub_service(securities={"GAZP": _fake_security("GAZP", sec_id=3)})
    data = SubscriptionIn(
        chat_id=300,
        kind=AlertKind.DIVIDEND_UPCOMING,
        params={"ticker": "GAZP", "lead_days": 5},
    )
    result = await svc.create(data)
    assert result.kind == AlertKind.DIVIDEND_UPCOMING


async def test_create_dividend_upcoming_without_ticker_raises_422() -> None:
    """create: dividend_upcoming без 'ticker' → InvalidAlertParamsError 422."""
    svc = _sub_service()
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.DIVIDEND_UPCOMING,
        params={"lead_days": 3},
    )
    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)
    assert "ticker" in exc_info.value.detail


async def test_create_dividend_upcoming_lead_days_out_of_range_raises_422() -> None:
    """create: dividend_upcoming с lead_days > 30 → InvalidAlertParamsError 422."""
    svc = _sub_service(securities={"SBER": _fake_security("SBER")})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.DIVIDEND_UPCOMING,
        params={"ticker": "SBER", "lead_days": 31},
    )
    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)
    assert exc_info.value.status == 422


async def test_create_dividend_upcoming_default_lead_days_is_valid() -> None:
    """create: dividend_upcoming без lead_days использует дефолт 3 (в диапазоне)."""
    svc = _sub_service(securities={"SBER": _fake_security("SBER")})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.DIVIDEND_UPCOMING,
        params={"ticker": "SBER"},
    )
    result = await svc.create(data)
    assert result.kind == AlertKind.DIVIDEND_UPCOMING


async def test_collect_pending_returns_empty_when_no_subscriptions() -> None:
    """collect_pending: нет подписок → пустой список."""
    svc = _eval_service(subs=[])
    result = await svc.collect_pending()
    assert result == []


async def test_collect_pending_skips_already_fired_price_alert() -> None:
    """collect_pending: Redis-ключ уже есть → цена не возвращается."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(1, 101, AlertKind.PRICE_LEVEL, {"ticker": "SBER", "level": 282.0})
    redis = FakeRedis()
    redis._store[f"alert:price:101:SBER:282.0:{_TODAY.isoformat()}"] = "1"

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        closes={1: [Decimal("280.00"), Decimal("283.00")]},
        redis=redis,
    )
    result = await svc.collect_pending()
    assert result == []


async def test_collect_pending_fires_price_alert_when_level_crossed() -> None:
    """collect_pending: уровень между двумя последними close → алерт срабатывает."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(1, 101, AlertKind.PRICE_LEVEL, {"ticker": "SBER", "level": 282.0})

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        closes={1: [Decimal("280.00"), Decimal("283.00")]},
        redis=FakeRedis(),
    )
    result = await svc.collect_pending()
    assert len(result) == 1
    assert result[0].kind == AlertKind.PRICE_LEVEL
    assert result[0].chat_id == 101


async def test_collect_pending_does_not_fire_price_alert_when_level_not_crossed() -> None:
    """collect_pending: уровень вне диапазона двух close → алерт не срабатывает."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(1, 101, AlertKind.PRICE_LEVEL, {"ticker": "SBER", "level": 300.0})

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        closes={1: [Decimal("280.00"), Decimal("283.00")]},
        redis=FakeRedis(),
    )
    result = await svc.collect_pending()
    assert result == []


async def test_collect_pending_fires_price_alert_both_directions() -> None:
    """collect_pending: crossed в обе стороны (prev > level > cur тоже считается)."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(1, 101, AlertKind.PRICE_LEVEL, {"ticker": "SBER", "level": 281.0})

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        closes={1: [Decimal("283.00"), Decimal("280.00")]},
        redis=FakeRedis(),
    )
    result = await svc.collect_pending()
    assert len(result) == 1


async def test_collect_pending_skips_price_alert_with_fewer_than_two_closes() -> None:
    """collect_pending: менее двух закрытий → пропустить подписку."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(1, 101, AlertKind.PRICE_LEVEL, {"ticker": "SBER", "level": 282.0})

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        closes={1: [Decimal("280.00")]},
        redis=FakeRedis(),
    )
    result = await svc.collect_pending()
    assert result == []


async def test_collect_pending_fires_sentiment_spike_for_negative_news_today() -> None:
    """collect_pending: свежая негативная новость сегодня → алерт срабатывает."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(2, 202, AlertKind.SENTIMENT_SPIKE, {"ticker": "SBER"})
    article, sentiment = _fake_article(article_id=55, published_at=_TODAY_DT)

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        news_rows=[(article, sentiment)],
        redis=FakeRedis(),
    )
    result = await svc.collect_pending()
    assert len(result) == 1
    assert result[0].kind == AlertKind.SENTIMENT_SPIKE
    assert result[0].article_id == 55


async def test_collect_pending_skips_already_fired_sentiment_alert() -> None:
    """collect_pending: дедупликация по article_id → повторный алерт не срабатывает."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(2, 202, AlertKind.SENTIMENT_SPIKE, {"ticker": "SBER"})
    article, sentiment = _fake_article(article_id=55, published_at=_TODAY_DT)

    redis = FakeRedis()
    redis._store["alert:sent:202:SBER:55"] = "1"

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        news_rows=[(article, sentiment)],
        redis=redis,
    )
    result = await svc.collect_pending()
    assert result == []


async def test_collect_pending_fires_dividend_upcoming_within_lead_days() -> None:
    """collect_pending: ex_date в пределах lead_days → алерт срабатывает."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(3, 303, AlertKind.DIVIDEND_UPCOMING, {"ticker": "SBER", "lead_days": 5})
    div = _fake_dividend(ex_date=date(2026, 6, 25))

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        dividends={1: [div]},
        redis=FakeRedis(),
    )
    result = await svc.collect_pending()
    assert len(result) == 1
    assert result[0].kind == AlertKind.DIVIDEND_UPCOMING
    assert result[0].ex_date == date(2026, 6, 25)


async def test_collect_pending_skips_dividend_alert_when_deduped() -> None:
    """collect_pending: Redis-ключ для этой ex_date уже есть → повтор не срабатывает."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(3, 303, AlertKind.DIVIDEND_UPCOMING, {"ticker": "SBER", "lead_days": 5})
    div = _fake_dividend(ex_date=date(2026, 6, 25))

    redis = FakeRedis()
    redis._store["alert:div:303:SBER:2026-06-25"] = "1"

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        dividends={1: [div]},
        redis=redis,
    )
    result = await svc.collect_pending()
    assert result == []


async def test_collect_pending_skips_unknown_ticker_subscription() -> None:
    """collect_pending: тикер из params не в БД → пропустить, не падать."""
    sub = _fake_sub(1, 101, AlertKind.PRICE_LEVEL, {"ticker": "GHOST", "level": 100.0})

    svc = _eval_service(
        subs=[sub],
        securities={},
        closes={},
        redis=FakeRedis(),
    )
    result = await svc.collect_pending()
    assert result == []


async def test_collect_pending_skips_volatility_regime_when_assessor_not_configured() -> None:
    """collect_pending: volatility_regime пропускается когда assessor=None."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(9, 909, AlertKind.VOLATILITY_REGIME, {"ticker": "SBER"})

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        redis=FakeRedis(),
    )
    result = await svc.collect_pending()
    assert result == []


async def test_collect_pending_redis_down_still_fires_once() -> None:
    """collect_pending: Redis недоступен → fail-open, алерт проходит."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(1, 101, AlertKind.PRICE_LEVEL, {"ticker": "SBER", "level": 282.0})

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        closes={1: [Decimal("280.00"), Decimal("283.00")]},
        redis=FakeRedis(down=True),
    )
    result = await svc.collect_pending()
    assert len(result) == 1


async def test_digest_claim_returns_true_on_first_call() -> None:
    """digest_claim: первый вызов для даты → claimed=True."""
    svc = _eval_service(subs=[], redis=FakeRedis())
    claimed = await svc.digest_claim(date(2026, 6, 23))
    assert claimed is True


async def test_digest_claim_returns_false_on_second_call() -> None:
    """digest_claim: Redis ключ уже присутствует → claimed=False."""
    redis = FakeRedis()
    redis._store["digest:2026-06-23"] = "1"

    svc = _eval_service(subs=[], redis=redis)
    claimed = await svc.digest_claim(date(2026, 6, 23))
    assert claimed is False


async def test_digest_claim_redis_down_returns_true() -> None:
    """digest_claim: Redis недоступен → fail-open, вернуть True."""
    svc = _eval_service(subs=[], redis=FakeRedis(down=True))
    claimed = await svc.digest_claim(date(2026, 6, 23))
    assert claimed is True


async def test_evaluate_volatility_regime_fires_alert_when_elevated_and_first_time() -> None:
    """_evaluate_volatility_regime: is_elevated=True + Redis свободен → PendingAlertOut."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(9, 909, AlertKind.VOLATILITY_REGIME, {"ticker": "SBER"})
    assessor = _FakeAssessor(regime=_fake_regime(is_elevated=True))

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        redis=FakeRedis(),
        assessor=assessor,
    )
    result = await svc.collect_pending()

    assert len(result) == 1
    alert = result[0]
    assert alert.kind == AlertKind.VOLATILITY_REGIME
    assert alert.chat_id == 909
    assert alert.ticker == "SBER"
    assert alert.volatility == pytest.approx(0.05)
    assert alert.threshold == pytest.approx(0.03)
    assert alert.regime_quantile == pytest.approx(0.80)


async def test_evaluate_volatility_regime_no_alert_when_not_elevated() -> None:
    """_evaluate_volatility_regime: is_elevated=False → пустой список."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(9, 909, AlertKind.VOLATILITY_REGIME, {"ticker": "SBER"})
    assessor = _FakeAssessor(regime=_fake_regime(is_elevated=False))

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        redis=FakeRedis(),
        assessor=assessor,
    )
    result = await svc.collect_pending()
    assert result == []


async def test_evaluate_volatility_regime_dedup_blocks_second_alert() -> None:
    """_evaluate_volatility_regime: Redis-ключ уже есть → повторный алерт не проходит."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(9, 909, AlertKind.VOLATILITY_REGIME, {"ticker": "SBER"})
    assessor = _FakeAssessor(regime=_fake_regime(is_elevated=True))

    redis = FakeRedis()
    redis._store[f"alert:vol:909:SBER:{_TODAY.isoformat()}"] = "1"

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        redis=redis,
        assessor=assessor,
    )
    result = await svc.collect_pending()
    assert result == []


async def test_evaluate_volatility_regime_skips_and_logs_on_model_not_loaded() -> None:
    """_evaluate_volatility_regime: ModelNotLoadedError → пропустить, залогировать warning."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(9, 909, AlertKind.VOLATILITY_REGIME, {"ticker": "SBER"})
    assessor = _FakeAssessor(raises=ModelNotLoadedError("stocklens-volatility"))

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        redis=FakeRedis(),
        assessor=assessor,
    )
    with capture_logs() as logs:
        result = await svc.collect_pending()

    assert result == []
    assert any(entry["event"] == "volatility_regime_skipped" for entry in logs)


async def test_evaluate_volatility_regime_skips_and_logs_on_insufficient_history() -> None:
    """_evaluate_volatility_regime: InsufficientHistoryError → пропустить, залогировать warning."""
    sber = _fake_security("SBER", sec_id=1)
    sub = _fake_sub(9, 909, AlertKind.VOLATILITY_REGIME, {"ticker": "SBER"})
    assessor = _FakeAssessor(raises=InsufficientHistoryError("SBER", 5, 60))

    svc = _eval_service(
        subs=[sub],
        securities={"SBER": sber},
        redis=FakeRedis(),
        assessor=assessor,
    )
    with capture_logs() as logs:
        result = await svc.collect_pending()

    assert result == []
    assert any(entry["event"] == "volatility_regime_skipped" for entry in logs)


async def test_create_volatility_regime_valid_ticker_only_succeeds() -> None:
    """create: volatility_regime с только ticker создаётся успешно."""
    svc = _sub_service(securities={"SBER": _fake_security("SBER")})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.VOLATILITY_REGIME,
        params={"ticker": "SBER"},
    )
    result = await svc.create(data)
    assert result.kind == AlertKind.VOLATILITY_REGIME


async def test_create_volatility_regime_valid_with_quantile_and_lookback_succeeds() -> None:
    """create: volatility_regime с ticker+quantile+lookback создаётся успешно."""
    svc = _sub_service(securities={"SBER": _fake_security("SBER")})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.VOLATILITY_REGIME,
        params={"ticker": "SBER", "quantile": 0.90, "lookback": 500},
    )
    result = await svc.create(data)
    assert result.kind == AlertKind.VOLATILITY_REGIME


async def test_create_volatility_regime_invalid_quantile_raises_422() -> None:
    """create: volatility_regime с quantile=0.1 (< 0.5) → InvalidAlertParamsError 422."""
    svc = _sub_service(securities={"SBER": _fake_security("SBER")})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.VOLATILITY_REGIME,
        params={"ticker": "SBER", "quantile": 0.1},
    )
    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)
    assert exc_info.value.status == 422
    assert "quantile" in exc_info.value.detail


async def test_create_volatility_regime_invalid_lookback_raises_422() -> None:
    """create: volatility_regime с lookback=10 (< 60) → InvalidAlertParamsError 422."""
    svc = _sub_service(securities={"SBER": _fake_security("SBER")})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.VOLATILITY_REGIME,
        params={"ticker": "SBER", "lookback": 10},
    )
    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)
    assert exc_info.value.status == 422
    assert "lookback" in exc_info.value.detail
