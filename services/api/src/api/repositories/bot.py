"""Реализация BotSubscriptionRepository на AsyncSession (SQLAlchemy 2.0)."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.enums import AlertKind
from stocklens_core.models.portfolio import BotSubscription


class SqlBotSubscriptionRepository:
    """Читает и записывает Telegram-подписки через PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_by_chat(self, chat_id: int) -> list[BotSubscription]:
        """Вернуть все подписки для указанного chat_id, сортировка по id."""
        result = await self._session.execute(
            select(BotSubscription)
            .where(BotSubscription.chat_id == chat_id)
            .order_by(BotSubscription.id)
        )
        return list(result.scalars().all())

    async def list_all_active(self) -> list[BotSubscription]:
        """Вернуть все подписки по всем chat_id, сортировка по id.

        «Активные» — все строки таблицы: удалённая подписка физически удаляется.
        """
        result = await self._session.execute(select(BotSubscription).order_by(BotSubscription.id))
        return list(result.scalars().all())

    async def create(
        self,
        chat_id: int,
        kind: AlertKind,
        params: dict[str, object],
    ) -> BotSubscription:
        """Создать подписку. Коммитит транзакцию."""
        subscription = BotSubscription(chat_id=chat_id, kind=kind, params=params)
        self._session.add(subscription)
        await self._session.commit()
        await self._session.refresh(subscription)
        return subscription

    async def delete(self, sub_id: int) -> bool:
        """Удалить подписку по id. Возвращает True если строка существовала."""
        result = await self._session.execute(
            select(BotSubscription).where(BotSubscription.id == sub_id)
        )
        subscription = result.scalar_one_or_none()
        if subscription is None:
            return False
        await self._session.delete(subscription)
        await self._session.commit()
        return True
