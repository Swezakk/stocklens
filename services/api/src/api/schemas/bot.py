"""DTO для управления Telegram-подписками на алерты."""

from pydantic import BaseModel, Field
from stocklens_core.enums import AlertKind


class SubscriptionIn(BaseModel):
    """Входные данные для создания подписки."""

    chat_id: int = Field(description="Telegram chat_id пользователя")
    kind: AlertKind = Field(description="Тип алерта")
    params: dict[str, object] = Field(
        default_factory=dict,
        description="Параметры алерта. Для price_level обязателен ключ 'level' (число).",
    )


class SubscriptionOut(BaseModel):
    """Выходные данные подписки."""

    model_config = {"from_attributes": True}

    id: int
    chat_id: int
    kind: AlertKind
    params: dict[str, object]
