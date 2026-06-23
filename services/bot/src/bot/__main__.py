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
from aiogram.fsm.storage.memory import MemoryStorage

from bot import handlers
from bot.dependencies import build_api_client
from bot.logging_setup import configure_logging
from bot.menu import setup_bot_profile
from bot.scheduler import build_scheduler
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
    """Запустить бота: логирование → Bot → проверка связи → heartbeat → polling → close."""
    settings = get_settings()
    configure_logging(settings.log_pretty)

    bot = Bot(
        token=settings.telegram_bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.include_router(handlers.router)
    api_client = build_api_client(settings)
    scheduler = build_scheduler(bot, api_client, settings)

    try:
        # Heartbeat стартует только ПОСЛЕ успешного get_me. Иначе healthcheck врёт: при
        # недостижимом Telegram контейнер рестартует быстрее окна healthcheck и вечно
        # числится healthy при crash-loop polling. Результат get_me кешируется в bot —
        # start_polling переиспользует его без второго сетевого вызова.
        identity = await bot.get_me()
        _log.info(
            "bot_starting",
            username=identity.username,
            api_base_url=settings.api_base_url,
        )
        try:
            await setup_bot_profile(bot)
        except Exception:
            # Профиль (меню/описания) — косметика: его сбой НЕ должен валить polling/алерты
            # и ронять honest-healthcheck (heartbeat ниже). Логируем и продолжаем.
            _log.exception("bot_profile_setup_failed")
        heartbeat = asyncio.create_task(
            _heartbeat_loop(settings.heartbeat_path, _HEARTBEAT_INTERVAL_SECONDS)
        )
        scheduler.start()
        try:
            await dispatcher.start_polling(bot, api_client=api_client)
        finally:
            heartbeat.cancel()
            scheduler.shutdown(wait=False)
    finally:
        await api_client.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
