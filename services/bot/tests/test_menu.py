"""Тесты setup_bot_profile: проверяем, что все четыре Bot API-вызовы произошли.

Используем типизированный fake, реализующий BotProfileApi Protocol — без AsyncMock.
"""

from aiogram.types import (
    BotCommand,
    BotCommandScopeDefault,
    BotCommandScopeUnion,
    MenuButtonCommands,
    MenuButtonUnion,
)
from bot.menu import (
    _COMMANDS,
    _DESCRIPTION,
    _SHORT_DESCRIPTION,
    BotProfileApi,
    setup_bot_profile,
)


class _FakeBot:
    """Заглушка Bot, реализующая BotProfileApi Protocol, записывает вызовы для проверки."""

    def __init__(self) -> None:
        self.commands_set: list[BotCommand] | None = None
        self.commands_scope: BotCommandScopeUnion | None = None
        self.short_description_set: str | None = None
        self.description_set: str | None = None
        self.menu_button_set: MenuButtonUnion | None = None

    async def set_my_commands(
        self,
        commands: list[BotCommand],
        scope: BotCommandScopeUnion | None = None,
        language_code: str | None = None,
        request_timeout: int | None = None,
    ) -> bool:
        self.commands_set = commands
        self.commands_scope = scope
        return True

    async def set_my_short_description(
        self,
        short_description: str | None = None,
        language_code: str | None = None,
        request_timeout: int | None = None,
    ) -> bool:
        self.short_description_set = short_description
        return True

    async def set_my_description(
        self,
        description: str | None = None,
        language_code: str | None = None,
        request_timeout: int | None = None,
    ) -> bool:
        self.description_set = description
        return True

    async def set_chat_menu_button(
        self,
        chat_id: int | None = None,
        menu_button: MenuButtonUnion | None = None,
        request_timeout: int | None = None,
    ) -> bool:
        self.menu_button_set = menu_button
        return True


def _make_fake() -> _FakeBot:
    fake = _FakeBot()
    _: BotProfileApi = fake
    return fake


async def test_setup_bot_profile_sets_commands() -> None:
    fake = _make_fake()
    await setup_bot_profile(fake)
    assert fake.commands_set == _COMMANDS


async def test_setup_bot_profile_uses_default_scope() -> None:
    fake = _make_fake()
    await setup_bot_profile(fake)
    assert isinstance(fake.commands_scope, BotCommandScopeDefault)


async def test_setup_bot_profile_sets_short_description() -> None:
    fake = _make_fake()
    await setup_bot_profile(fake)
    assert fake.short_description_set == _SHORT_DESCRIPTION


async def test_setup_bot_profile_sets_description() -> None:
    fake = _make_fake()
    await setup_bot_profile(fake)
    assert fake.description_set == _DESCRIPTION


async def test_setup_bot_profile_sets_menu_button_commands() -> None:
    fake = _make_fake()
    await setup_bot_profile(fake)
    assert isinstance(fake.menu_button_set, MenuButtonCommands)


def test_commands_list_includes_required_commands() -> None:
    command_names = {cmd.command for cmd in _COMMANDS}
    assert "portfolio" in command_names
    assert "digest" in command_names
    assert "subscribe" in command_names
    assert "unsubscribe" in command_names
    assert "help" in command_names


def test_commands_list_does_not_include_start() -> None:
    command_names = {cmd.command for cmd in _COMMANDS}
    assert "start" not in command_names
