"""Настройка профиля бота в Telegram: команды, описания, кнопка меню.

Вызывается один раз при старте (idempotent — повторный вызов безопасен).
Изменения видны пользователю после перезапуска чата или контейнера.
"""

from typing import Protocol

from aiogram.types import (
    BotCommand,
    BotCommandScopeDefault,
    BotCommandScopeUnion,
    MenuButtonCommands,
    MenuButtonUnion,
)

_COMMANDS = [
    BotCommand(command="portfolio", description="📊 Сводка портфеля"),
    BotCommand(command="digest", description="📰 Дайджест за сегодня"),
    BotCommand(command="subscribe", description="🔔 Новая подписка на алерт"),
    BotCommand(command="unsubscribe", description="Мои подписки"),
    BotCommand(command="help", description="Справка"),
]

_SHORT_DESCRIPTION = "Аналитика MOEX: портфель, дивиденды, алерты."

_DESCRIPTION = (
    "StockLens — персональная аналитика российского фондового рынка.\n\n"
    "Что умею:\n"
    "• сводка портфеля и сравнение с IMOEX\n"
    "• ежедневный дайджест: отсечки и негативные новости\n"
    "• алерты на уровень цены, всплеск негатива и дивиденды\n\n"
    "Нажмите «Запустить», чтобы начать."
)


class BotProfileApi(Protocol):
    """Минимальный интерфейс Bot, необходимый для настройки профиля."""

    async def set_my_commands(
        self,
        commands: list[BotCommand],
        scope: BotCommandScopeUnion | None = None,
        language_code: str | None = None,
        request_timeout: int | None = None,
    ) -> bool: ...

    async def set_my_short_description(
        self,
        short_description: str | None = None,
        language_code: str | None = None,
        request_timeout: int | None = None,
    ) -> bool: ...

    async def set_my_description(
        self,
        description: str | None = None,
        language_code: str | None = None,
        request_timeout: int | None = None,
    ) -> bool: ...

    async def set_chat_menu_button(
        self,
        chat_id: int | None = None,
        menu_button: MenuButtonUnion | None = None,
        request_timeout: int | None = None,
    ) -> bool: ...


async def setup_bot_profile(bot: BotProfileApi) -> None:
    """Установить команды, описания и кнопку меню через Bot API (идемпотентно)."""
    await bot.set_my_commands(_COMMANDS, scope=BotCommandScopeDefault())
    await bot.set_my_short_description(_SHORT_DESCRIPTION)
    await bot.set_my_description(_DESCRIPTION)
    await bot.set_chat_menu_button(menu_button=MenuButtonCommands())
