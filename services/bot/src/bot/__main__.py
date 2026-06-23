"""Точка входа Telegram-бота: long-polling aiogram + heartbeat для healthcheck (DESIGN §11).

Бот — долгоживущий async-процесс (аналог ingestor, но на asyncio): конфигурирует логирование,
строит Bot/Dispatcher, инъектирует API-клиент в хендлеры (DI aiogram через ``start_polling``),
параллельно тикает heartbeat-файлом для healthcheck контейнера и корректно закрывает ресурсы
при остановке. Запуск: ``python -m bot``.
"""

import asyncio
from pathlib import Path

import structlog
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot import handlers
from bot.dependencies import build_api_client
from bot.logging_setup import configure_logging
from bot.settings import get_settings

_log = structlog.get_logger()

#: Период обновления heartbeat-файла (сек); healthcheck контейнера проверяет его свежесть.
_HEARTBEAT_INTERVAL_SECONDS = 30.0


async def _heartbeat_loop(path: Path, interval: float) -> None:
    """Периодически обновлять heartbeat-файл, пока бот жив (сигнал живости для healthcheck)."""
    while True:
        path.write_text("ok", encoding="utf-8")
        await asyncio.sleep(interval)


async def main() -> None:
    """Запустить бота: логирование → Bot/Dispatcher → heartbeat → long-polling → graceful close."""
    settings = get_settings()
    configure_logging(settings.log_pretty)

    bot = Bot(
        token=settings.telegram_bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)
    api_client = build_api_client(settings)

    heartbeat = asyncio.create_task(
        _heartbeat_loop(settings.heartbeat_path, _HEARTBEAT_INTERVAL_SECONDS)
    )
    _log.info("bot_starting", api_base_url=settings.api_base_url)
    try:
        await dispatcher.start_polling(bot, api_client=api_client)
    finally:
        heartbeat.cancel()
        await api_client.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
