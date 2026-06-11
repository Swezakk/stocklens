"""Unit-тесты BotSubscriptionService с фиктивными репозиториями (без БД)."""

from dataclasses import dataclass, field
from typing import cast

import pytest
from api.core.exceptions import InsufficientDataError, SubscriptionNotFoundError
from api.repositories.protocols import BotSubscriptionRepository
from api.schemas.bot import SubscriptionIn
from api.services.bot import BotSubscriptionService
from stocklens_core.enums import AlertKind
from stocklens_core.models.portfolio import BotSubscription


@dataclass
class FakeSubscription:
    """Подмена BotSubscription для unit-тестов."""

    id: int
    chat_id: int
    kind: AlertKind
    params: dict[str, object] = field(default_factory=dict)


def _fake_sub(
    sub_id: int = 1,
    chat_id: int = 100,
    kind: AlertKind = AlertKind.SENTIMENT_SPIKE,
) -> BotSubscription:
    return cast(
        BotSubscription,
        FakeSubscription(id=sub_id, chat_id=chat_id, kind=kind),
    )


@dataclass
class _BotRepoWithData:
    subs: list[BotSubscription] = field(default_factory=lambda: [_fake_sub()])

    async def list_by_chat(self, chat_id: int) -> list[BotSubscription]:
        return [s for s in self.subs if s.chat_id == chat_id]

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


def _service(repo: BotSubscriptionRepository) -> BotSubscriptionService:
    return BotSubscriptionService(repo=repo)


async def test_list_by_chat_returns_subscriptions() -> None:
    """list_by_chat: возвращает подписки для chat_id."""
    svc = _service(_BotRepoWithData())
    result = await svc.list_by_chat(chat_id=100)

    assert len(result) == 1
    assert result[0].chat_id == 100


async def test_create_sentiment_spike_subscription_without_params() -> None:
    """create: sentiment_spike не требует параметров."""
    svc = _service(_BotRepoEmpty())
    data = SubscriptionIn(chat_id=100, kind=AlertKind.SENTIMENT_SPIKE, params={})
    result = await svc.create(data)

    assert result.kind == AlertKind.SENTIMENT_SPIKE
    assert result.chat_id == 100


async def test_create_price_level_subscription_with_valid_level() -> None:
    """create: price_level с валидным уровнем создаётся успешно."""
    svc = _service(_BotRepoEmpty())
    data = SubscriptionIn(
        chat_id=200,
        kind=AlertKind.PRICE_LEVEL,
        params={"level": 300.0},
    )
    result = await svc.create(data)

    assert result.kind == AlertKind.PRICE_LEVEL
    assert result.params["level"] == 300.0


async def test_create_price_level_without_level_raises_422() -> None:
    """create: price_level без ключа 'level' → InsufficientDataError 422."""
    svc = _service(_BotRepoEmpty())
    data = SubscriptionIn(chat_id=100, kind=AlertKind.PRICE_LEVEL, params={})

    with pytest.raises(InsufficientDataError) as exc_info:
        await svc.create(data)

    assert exc_info.value.status == 422
    assert "level" in exc_info.value.detail


async def test_create_price_level_with_non_numeric_level_raises_422() -> None:
    """create: price_level с нечисловым 'level' → InsufficientDataError 422."""
    svc = _service(_BotRepoEmpty())
    data = SubscriptionIn(
        chat_id=100,
        kind=AlertKind.PRICE_LEVEL,
        params={"level": "not-a-number"},
    )

    with pytest.raises(InsufficientDataError) as exc_info:
        await svc.create(data)

    assert exc_info.value.status == 422


async def test_delete_raises_404_when_subscription_absent() -> None:
    """delete: подписка не найдена → SubscriptionNotFoundError 404."""
    svc = _service(_BotRepoEmpty())

    with pytest.raises(SubscriptionNotFoundError) as exc_info:
        await svc.delete(sub_id=999)

    assert exc_info.value.sub_id == 999
    assert exc_info.value.status == 404
