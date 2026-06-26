# api — HTTP-ядро StockLens

FastAPI-сервис: единственная точка доступа к данным и ML-прогнозам для дашборда и бота.

## Назначение

Async REST-API, отдающий рыночные данные, новости, портфель, наблюдаемость и ML-инференс
(волатильность/тренд). Владеет записью в `portfolio_positions`, `bot_subscriptions`,
`watchlist` и `predictions` (идемпотентный upsert при инференсе); рыночные и новостные
таблицы — только на чтение (их пишет ingestor). Слоистая архитектура
`routers → services → repositories` с DTO (`schemas/`), отделёнными от ORM-моделей.

## Место в архитектуре

Читает PostgreSQL через `asyncpg`/`AsyncSession`, кэширует в Redis (`redis.asyncio`),
на старте грузит модели из реестра MLflow (`lifespan` → threadpool, чтобы не блокировать
event-loop). Потребители — `dashboard` и `bot` (HTTP-only, JWT владельца). Сервис
**полностью асинхронный**; CPU-bound инференс выносится в threadpool.
Контракт — [docs/specs/2026-06-11-stocklens-design.md](../../docs/specs/2026-06-11-stocklens-design.md).

## Стек

FastAPI · Pydantic v2 + pydantic-settings · SQLAlchemy 2.0 async (asyncpg) ·
redis.asyncio · MLflow + CatBoost + `arch` (инференс) · structlog · uvicorn ·
RFC 9457 Problem Details.

## Запуск

В составе стека — `docker compose up -d --build` (зависит от `db`, `migrations`, `redis`,
`mlflow`; слушает `127.0.0.1:8000`). Локально:

```bash
uv sync --project services/api
uv run --project services/api uvicorn api.main:create_app --factory --reload --port 8000
```

Swagger — `/api/docs`, ReDoc — `/api/redoc`, OpenAPI — `/api/openapi.json`.

## Тесты

```bash
uv run --project services/api pytest services/api/tests           # unit + integration (testcontainers PG+Redis)
uv run --project services/api mypy services/api/src services/api/tests
```

## Структура

- `main.py` — фабрика `create_app` (uvicorn `--factory`): middleware, exception-handlers,
  подключение роутеров с зависимостью `require_auth`.
- `routers/` — маршруты под `/api/v1`: `auth` (OAuth2 password → JWT), `data`, `portfolio`,
  `watchlist`, `bot`, `predict`, `monitoring`, `health`.
- `services/` — бизнес-логика (свечи, дивиденды, прогнозы, подписки, оценка алертов).
- `repositories/` — доступ к данным; интерфейсы через `typing.Protocol` (`protocols.py`).
- `schemas/` — Pydantic-DTO запросов/ответов, отделённые от ORM.
- `core/` — `settings`, `lifespan`, `db`, `cache`, `middleware`, `problem` (RFC 9457),
  `pagination`, `exceptions`, `auth/`.
- `ml/` — `loader` (модели из MLflow по алиасу), `bundle`, `trend`, `features`, `deps`.
