# bot — Telegram-бот StockLens

Async-бот на aiogram: команды, подписки на алерты, утренний дайджест владельца и запуск
ежедневной генерации прогнозов.

## Назначение

Долгоживущий процесс long-polling. Обрабатывает команды и мастер настройки подписок,
рассылает сработавшие алерты по чатам и шлёт владельцу утренний дайджест (08:30 МСК).
Подписки (`bot_subscriptions`) хранит API — бот лишь дёргает его ручки. Бизнес-решения
(какие алерты сработали, состав дайджеста) принимает сервер.

## Место в архитектуре

Ходит **только в API по HTTP** — прямого доступа к БД нет. `ApiClient` поверх `httpx`
логинится учётными данными владельца (JWT, `token_manager`). Внешняя система — Telegram
Bot API (aiogram). Сервис async (`asyncio`); планировщик `AsyncIOScheduler` живёт в том же
event-loop, что и polling. Контракт —
[docs/specs/2026-06-11-stocklens-design.md](../../docs/specs/2026-06-11-stocklens-design.md).

## Стек

aiogram 3.x (long-polling, FSM `MemoryStorage`) · APScheduler 3.x (`AsyncIOScheduler`) ·
httpx · Pydantic v2 + pydantic-settings · structlog.

## Запуск

В составе стека — `docker compose up -d --build` (зависит от `api`; healthcheck по свежести
heartbeat-файла). Запуск контейнера — `python -m bot`. Локально:

```bash
uv sync --project services/bot
TELEGRAM_BOT_TOKEN=… API_URL=http://localhost:8000 DIGEST_CHAT_ID=… \
  uv run --project services/bot python -m bot
```

## Тесты

```bash
uv run --project services/bot pytest services/bot/tests
uv run --project services/bot mypy services/bot/src services/bot/tests
```

## Структура

- `__main__.py` — точка входа: `Bot`/`Dispatcher` → `get_me` → heartbeat → `start_polling`.
- `scheduler.py` — три задания (МСК): `alert_sweep` (опрос алертов каждые N минут),
  `digest_daily` (08:30, claim-once-per-day), `forecast_refresh_daily` (запуск пакетного
  ML-инференса через API).
- `handlers.py`, `callbacks.py`, `wizard.py`, `states.py` — команды, инлайн-колбэки и FSM
  мастера подписок.
- `subscriptions.py`, `digest.py`, `digest_model.py` — логика подписок и сборка дайджеста.
- `send.py`, `formatting.py`, `responses.py`, `keyboards.py`, `menu.py` — отправка и
  HTML-разметка сообщений.
- `api_client/` — `client`, `token_manager` (JWT владельца), `dto`, `errors`.
- `dependencies.py`, `settings.py`, `logging_setup.py`.
