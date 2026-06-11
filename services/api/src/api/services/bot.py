"""Сервис управления Telegram-подписками на алерты."""

from stocklens_core.enums import AlertKind
from stocklens_core.models.portfolio import BotSubscription

from api.core.exceptions import InsufficientDataError, SubscriptionNotFoundError
from api.repositories.protocols import BotSubscriptionRepository
from api.schemas.bot import SubscriptionIn, SubscriptionOut

_PRICE_LEVEL_REQUIRED_KEY = "level"


class BotSubscriptionService:
    """Управляет Telegram-подписками пользователей.

    Валидирует параметры по типу алерта: price_level требует числовой ключ 'level'.
    """

    def __init__(self, repo: BotSubscriptionRepository) -> None:
        self._repo = repo

    async def list_by_chat(self, chat_id: int) -> list[SubscriptionOut]:
        """Вернуть все подписки для указанного chat_id."""
        subscriptions = await self._repo.list_by_chat(chat_id)
        return [_to_out(s) for s in subscriptions]

    async def create(self, data: SubscriptionIn) -> SubscriptionOut:
        """Создать подписку.

        Raises:
            InsufficientDataError: если параметры не соответствуют типу алерта.
        """
        _validate_params(data.kind, data.params)
        subscription = await self._repo.create(
            chat_id=data.chat_id,
            kind=data.kind,
            params=data.params,
        )
        return _to_out(subscription)

    async def delete(self, sub_id: int) -> None:
        """Удалить подписку.

        Raises:
            SubscriptionNotFoundError: если подписка не найдена.
        """
        deleted = await self._repo.delete(sub_id)
        if not deleted:
            raise SubscriptionNotFoundError(sub_id)


def _validate_params(kind: AlertKind, params: dict[str, object]) -> None:
    """Проверить соответствие параметров типу алерта.

    Raises:
        InsufficientDataError: если обязательный ключ отсутствует или имеет неверный тип.
    """
    if kind == AlertKind.PRICE_LEVEL:
        level = params.get(_PRICE_LEVEL_REQUIRED_KEY)
        if level is None:
            raise InsufficientDataError(
                f"Для подписки price_level обязателен параметр {_PRICE_LEVEL_REQUIRED_KEY!r} "
                f"с числовым значением целевого уровня цены"
            )
        if not isinstance(level, int | float):
            raise InsufficientDataError(
                f"Параметр {_PRICE_LEVEL_REQUIRED_KEY!r} должен быть числом, "
                f"получено: {type(level).__name__}"
            )


def _to_out(subscription: BotSubscription) -> SubscriptionOut:
    """Преобразовать ORM-объект в DTO."""
    return SubscriptionOut.model_validate(subscription)
