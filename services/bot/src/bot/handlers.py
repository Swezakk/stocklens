"""Хендлеры команд бота (aiogram 3.x Router) — тонкая обёртка над ``responses`` (DESIGN §11).

Хендлер только извлекает из aiogram-объектов примитивы (chat_id, args), делегирует сборку
текста в ``responses`` (там вся логика и ветки ошибок) и отправляет ответ. API-клиент
инъектируется в polling как kwarg ``api_client`` (DI aiogram через Dispatcher/start_polling).
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from aiogram import Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message

from bot import responses
from bot.api_client.client import ApiClient

router = Router(name="commands")

#: «Сегодня» для дайджеста — по московскому календарю (правила данных: отображение в Europe/Moscow).
_MOSCOW_TZ = ZoneInfo("Europe/Moscow")


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """/start — приветствие и список команд."""
    await message.answer(responses.start_response())


@router.message(Command("portfolio"))
async def cmd_portfolio(message: Message, api_client: ApiClient) -> None:
    """/portfolio — сводка портфеля."""
    await message.answer(await responses.portfolio_response(api_client))


@router.message(Command("digest"))
async def cmd_digest(message: Message, api_client: ApiClient) -> None:
    """/digest — дайджест по портфелю."""
    today = datetime.now(tz=_MOSCOW_TZ).date()
    await message.answer(await responses.digest_response(api_client, today))


@router.message(Command("subscribe"))
async def cmd_subscribe(message: Message, command: CommandObject, api_client: ApiClient) -> None:
    """/subscribe — управление подписками на алерты."""
    await message.answer(
        await responses.subscribe_response(api_client, message.chat.id, command.args)
    )


@router.message(Command("unsubscribe"))
async def cmd_unsubscribe(message: Message, command: CommandObject, api_client: ApiClient) -> None:
    """/unsubscribe — удалить подписку или показать список."""
    await message.answer(
        await responses.unsubscribe_response(api_client, message.chat.id, command.args)
    )
