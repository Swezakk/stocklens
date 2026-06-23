"""Unit-тесты BotSubscriptionService с фиктивными репозиториями (без БД)."""

from dataclasses import dataclass, field
from typing import cast

import pytest
from api.core.exceptions import InvalidAlertParamsError, SubscriptionNotFoundError
from api.repositories.protocols import BotSubscriptionRepository, SecurityRepository
from api.schemas.bot import SubscriptionIn
from api.services.bot import BotSubscriptionService
from stocklens_core.enums import AlertKind
from stocklens_core.models.market import Security
from stocklens_core.models.portfolio import BotSubscription


@dataclass
class FakeSubscription:
    """Подмена BotSubscription для unit-тестов."""

    id: int
    chat_id: int
    kind: AlertKind
    params: dict[str, object] = field(default_factory=dict)


@dataclass
class FakeSecurity:
    """Подмена Security для unit-тестов."""

    id: int
    ticker: str
    name: str = "Test Co"
    board: str = "TQBR"
    aliases: list[str] = field(default_factory=list)
    is_active: bool = True


def _fake_sub(
    sub_id: int = 1,
    chat_id: int = 100,
    kind: AlertKind = AlertKind.SENTIMENT_SPIKE,
) -> BotSubscription:
    return cast(
        BotSubscription,
        FakeSubscription(id=sub_id, chat_id=chat_id, kind=kind),
    )


class _BotRepoWithData:
    def __init__(self) -> None:
        self.subs: list[BotSubscription] = [_fake_sub()]

    async def list_by_chat(self, chat_id: int) -> list[BotSubscription]:
        return [s for s in self.subs if s.chat_id == chat_id]

    async def list_all_active(self) -> list[BotSubscription]:
        return list(self.subs)

    async def create(
        self,
        chat_id: int,
        kind: AlertKind,
        params: dict[str, object],
    ) -> BotSubscription:
        sub = cast(
            BotSubscription,
            FakeSubscription(id=99, chat_id=chat_id, kind=kind, params=params),
        )
        self.subs.append(sub)
        return sub

    async def delete(self, sub_id: int) -> bool:
        return True


class _BotRepoEmpty:
    async def list_by_chat(self, chat_id: int) -> list[BotSubscription]:
        return []

    async def list_all_active(self) -> list[BotSubscription]:
        return []

    async def create(
        self,
        chat_id: int,
        kind: AlertKind,
        params: dict[str, object],
    ) -> BotSubscription:
        return cast(
            BotSubscription,
            FakeSubscription(id=1, chat_id=chat_id, kind=kind, params=params),
        )

    async def delete(self, sub_id: int) -> bool:
        return False


class _SecurityRepo:
    """Fake SecurityRepository с предзагруженным словарём тикеров."""

    def __init__(self, known: dict[str, int] | None = None) -> None:
        self._known: dict[str, int] = known or {"SBER": 1}

    async def get_by_ticker(self, ticker: str) -> Security | None:
        sec_id = self._known.get(ticker)
        if sec_id is None:
            return None
        return cast(Security, FakeSecurity(id=sec_id, ticker=ticker))

    async def list_securities(
        self, is_active: bool | None, limit: int, offset: int
    ) -> tuple[list[Security], int]:
        return [], 0


def _service(
    repo: BotSubscriptionRepository,
    known_tickers: dict[str, int] | None = None,
) -> BotSubscriptionService:
    return BotSubscriptionService(
        repo=repo,
        security_repo=cast(SecurityRepository, _SecurityRepo(known_tickers)),
    )


async def test_list_by_chat_returns_subscriptions() -> None:
    """list_by_chat: возвращает подписки для chat_id."""
    svc = _service(_BotRepoWithData())
    result = await svc.list_by_chat(chat_id=100)

    assert len(result) == 1
    assert result[0].chat_id == 100


async def test_create_sentiment_spike_with_valid_ticker_succeeds() -> None:
    """create: sentiment_spike с известным ticker создаётся успешно."""
    svc = _service(_BotRepoEmpty(), known_tickers={"SBER": 1})
    data = SubscriptionIn(chat_id=100, kind=AlertKind.SENTIMENT_SPIKE, params={"ticker": "SBER"})
    result = await svc.create(data)

    assert result.kind == AlertKind.SENTIMENT_SPIKE
    assert result.chat_id == 100


async def test_create_sentiment_spike_without_ticker_raises_422() -> None:
    """create: sentiment_spike без ticker → InvalidAlertParamsError 422."""
    svc = _service(_BotRepoEmpty())
    data = SubscriptionIn(chat_id=100, kind=AlertKind.SENTIMENT_SPIKE, params={})

    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)

    assert exc_info.value.status == 422
    assert "ticker" in exc_info.value.detail


async def test_create_price_level_subscription_with_valid_params_succeeds() -> None:
    """create: price_level с ticker+level создаётся успешно."""
    svc = _service(_BotRepoEmpty(), known_tickers={"SBER": 1})
    data = SubscriptionIn(
        chat_id=200,
        kind=AlertKind.PRICE_LEVEL,
        params={"ticker": "SBER", "level": 300.0},
    )
    result = await svc.create(data)

    assert result.kind == AlertKind.PRICE_LEVEL
    assert result.params["level"] == 300.0


async def test_create_price_level_without_level_raises_422() -> None:
    """create: price_level без ключа 'level' → InvalidAlertParamsError 422."""
    svc = _service(_BotRepoEmpty(), known_tickers={"SBER": 1})
    data = SubscriptionIn(chat_id=100, kind=AlertKind.PRICE_LEVEL, params={"ticker": "SBER"})

    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)

    assert exc_info.value.status == 422
    assert "level" in exc_info.value.detail


async def test_create_price_level_with_non_numeric_level_raises_422() -> None:
    """create: price_level с нечисловым 'level' → InvalidAlertParamsError 422."""
    svc = _service(_BotRepoEmpty(), known_tickers={"SBER": 1})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.PRICE_LEVEL,
        params={"ticker": "SBER", "level": "not-a-number"},
    )

    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)

    assert exc_info.value.status == 422


async def test_create_price_level_with_non_positive_level_raises_422() -> None:
    """create: price_level с level <= 0 → InvalidAlertParamsError 422."""
    svc = _service(_BotRepoEmpty(), known_tickers={"SBER": 1})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.PRICE_LEVEL,
        params={"ticker": "SBER", "level": 0.0},
    )

    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)

    assert exc_info.value.status == 422


async def test_create_price_level_unknown_ticker_raises_422() -> None:
    """create: price_level с неизвестным ticker → InvalidAlertParamsError 422."""
    svc = _service(_BotRepoEmpty(), known_tickers={})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.PRICE_LEVEL,
        params={"ticker": "GHOST", "level": 300.0},
    )

    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)

    assert "GHOST" in exc_info.value.detail


async def test_create_dividend_upcoming_with_valid_params_succeeds() -> None:
    """create: dividend_upcoming с ticker+lead_days создаётся успешно."""
    svc = _service(_BotRepoEmpty(), known_tickers={"GAZP": 2})
    data = SubscriptionIn(
        chat_id=300,
        kind=AlertKind.DIVIDEND_UPCOMING,
        params={"ticker": "GAZP", "lead_days": 5},
    )
    result = await svc.create(data)

    assert result.kind == AlertKind.DIVIDEND_UPCOMING


async def test_create_dividend_upcoming_lead_days_out_of_range_raises_422() -> None:
    """create: dividend_upcoming с lead_days вне [1..30] → InvalidAlertParamsError 422."""
    svc = _service(_BotRepoEmpty(), known_tickers={"SBER": 1})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.DIVIDEND_UPCOMING,
        params={"ticker": "SBER", "lead_days": 31},
    )

    with pytest.raises(InvalidAlertParamsError) as exc_info:
        await svc.create(data)

    assert exc_info.value.status == 422


async def test_create_dividend_upcoming_default_lead_days_is_valid() -> None:
    """create: dividend_upcoming без lead_days использует дефолт 3."""
    svc = _service(_BotRepoEmpty(), known_tickers={"SBER": 1})
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.DIVIDEND_UPCOMING,
        params={"ticker": "SBER"},
    )
    result = await svc.create(data)
    assert result.kind == AlertKind.DIVIDEND_UPCOMING


async def test_delete_raises_404_when_subscription_absent() -> None:
    """delete: подписка не найдена → SubscriptionNotFoundError 404."""
    svc = _service(_BotRepoEmpty())

    with pytest.raises(SubscriptionNotFoundError) as exc_info:
        await svc.delete(sub_id=999)

    assert exc_info.value.sub_id == 999
    assert exc_info.value.status == 404
