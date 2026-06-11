# CLAUDE.md — StockLens

## Контекст проекта

**StockLens** — персональный аналитический веб-сервис по российскому фондовому рынку:
ежедневный сбор котировок MOEX и финансовых новостей → детерминированная аналитика
и ML-прогнозы → Streamlit-дашборд + Telegram-алерты.

Назначение двойное: семестровый проект курса «Python & Data Science» (рамка DataPulse,
целевая оценка «5» + бонусы) **и** реальный инструмент владельца для старта в инвестициях.

**Единственный источник истины по дизайну** —
[docs/specs/2026-06-11-stocklens-design.md](docs/specs/2026-06-11-stocklens-design.md).
Перед любым архитектурным решением — сверяться со спекой. Расхождение кода со спекой —
либо баг кода, либо осознанное обновление спеки отдельным коммитом; молча отклоняться нельзя.

**Статус:** репозиторий создан, реализация — по фазам: core-пакет → миграции →
ingestor MOEX → RSS/sentiment → API → dashboard → bot → ML → деплой (Dokploy на VPS
владельца). Деплой прод-ingestor — как можно раньше (новости из RSS невосстановимы).

## Рабочий процесс

Все модифицирующие задачи идут через **`mainframe:task-workflow`** (triage → recon →
план → исполнение → верификация → коммит). Superpowers-цикл (brainstorming /
writing-plans) для этого проекта **не используется** — он применялся только на этапе
проектирования спеки и закрыт.

## Стандарт качества

Критерии курса — нижняя граница, не цель. Целевой стандарт — production-grade система:
эталонная слоистая архитектура, код enterprise-уровня. Слово «учебный» не является
оправданием упрощений, ломающих архитектуру. Применяются все правила глобального
`~/.claude/CLAUDE.md` владельца (TDD, запрет хардкода, обработка ошибок, recon-first);
ниже — только проектные конкретизации.

## Архитектурные инварианты

Нарушение любого пункта = регрессия архитектуры, требует явного согласования с владельцем.

1. **7 сервисов** в Docker Compose: `db` (PostgreSQL 16), `redis`, `ingestor`, `api`,
   `dashboard`, `bot`, `mlflow`. Не сливать и не дробить без обновления спеки.
2. **Один write-путь.** В рыночные и новостные таблицы пишет только `ingestor`
   (плюс Alembic-миграции). API владеет записью только в `portfolio_positions`
   и `bot_subscriptions`.
3. **`dashboard` и `bot` не знают про БД** — только HTTP-вызовы API. Прямой импорт
   SQLAlchemy-моделей или коннект к PostgreSQL из этих сервисов запрещён.
4. **Общий код** (SQLAlchemy-модели, доменные `StrEnum`, настройки) — пакет
   `packages/stocklens-core`. Дублирование схемы данных в сервисах запрещено.
5. **API — слоистый:** `routers → services → repositories` + `schemas/` (Pydantic-DTO),
   `core/` (конфиг, DI, исключения, логирование), `ml/` (загрузка моделей, инференс).
   Зависимости направлены только вниз; DTO строго отделены от ORM-моделей;
   repository-интерфейсы — через `typing.Protocol` (unit-тесты сервисов не поднимают БД).
6. **API полностью async** (asyncpg, `AsyncSession`, `redis.asyncio`); CPU-bound
   инференс — через threadpool. **Ingestor — синхронный осознанно** (батчевый сбор):
   не «улучшать» его переводом на async.
7. **Sentiment новостей скорится в `ingestor`** сразу при сборе (модель локальная,
   лёгкая) — API отдаёт готовые оценки из БД. В API живёт только инференс
   волатильности/тренда. Не переносить sentiment-скоринг в API.

## Стек (зафиксирован спекой)

Python 3.12 · uv (pyproject + uv.lock на каждый сервис) · FastAPI + Pydantic v2 +
pydantic-settings · SQLAlchemy 2.0 (async в api, sync в ingestor) · Alembic ·
PostgreSQL 16 · Redis 7 · APScheduler · aiogram · Streamlit + Plotly · MLflow · uvicorn ·
scikit-learn + CatBoost + `arch` (GARCH) + SHAP · rubert-tiny2 (sentiment) · structlog ·
pytest + testcontainers (HTTP-моки: `respx` для async httpx в api, `responses` для
sync requests в ingestor) · ruff (строгий набор: `B`, `UP`, `SIM`, `PL`) +
mypy strict (все сервисы).

Замена любого компонента — только через обновление спеки. Перед реализацией модуля —
сверка API библиотеки с актуальной документацией (context7), не с training data:
особенно FastAPI lifespan/DI, SQLAlchemy 2.0 async, aiogram 3.x, MLflow, `arch`.

## Правила данных

- **Время:** в БД — UTC (`timestamptz`); торговые даты — `date` по календарю биржи;
  отображение — Europe/Moscow.
- **Идемпотентность:** все записи сборщиков — upsert по натуральному ключу
  (`(security_id, trade_date)`, `url`, …). Повторный запуск любой задачи безопасен.
- **Статусы и типы** — только `StrEnum` из `stocklens-core` (`CollectorRunStatus`,
  `SentimentLabel`, `PredictionKind`, `Currency`, `AlertKind`); строковые литералы
  в логике запрещены.
- **MOEX — вежливый клиент:** ≤1 запрос/сек, retry с экспоненциальным backoff,
  каждый запуск сборщика журналируется в `collector_runs` (ошибка одного источника
  не валит остальные — статус `partial`).
- **Сессии выходного дня MOEX** (торгуются с 03.2025) помечаются `is_weekend_session`
  и исключаются из обучения моделей.
- **Backfill:** котировки/дивиденды/курсы восстановимы из истории API — ingestor
  при старте догоняет пропуск. **Новости невосстановимы** (RSS без архива) — поэтому
  прод-ingestor на VPS запускается как можно раньше и не останавливается надолго.

## ML-методология (обязательно)

- **Валидация только walk-forward / `TimeSeriesSplit`.** Случайный K-fold запрещён.
  Скейлеры и фичи фитятся только на train-окне — утечка через препроцессинг
  это главный известный риск проекта.
- **Каждая модель сравнивается с naive baseline** (волатильность — random-walk RV;
  тренд — «всегда вверх»; sentiment — словарный метод). Метрики и параметры —
  в MLflow, лучшая модель — в реестр.
- **Точечный прогноз цены/доходности не делаем** — это не-цель спеки (раздел 2)
  с обоснованием. Не добавлять «по просьбе» без обновления спеки.
- **Доходности корректируются на дивиденды** (таблица `dividends`) при расчёте фич.
- **Структурный разрыв 2022 г.** (остановка торгов 28.02–24.03.2022, смена режима
  рынка): основной вариант обучения — на данных пост-2022; сравнение периодов — в EDA.
- **Версия модели** пишется в `predictions.model_version` и отображается в UI.

## Правила API

- Все маршруты — под `/api/v1`; `response_model` обязателен на каждом маршруте.
- Ошибки: иерархия доменных исключений → централизованные handlers → RFC 9457
  Problem Details; тексты 4xx — по-русски, с сущностью и причиной.
- Жизненный цикл — `lifespan` (не `on_event`); конфигурация — только
  `pydantic-settings`, чтение `os.environ` по коду запрещено.
- Наблюдаемость: structlog (JSON в проде), request-id middleware, раздельные
  `/health/live` и `/health/ready`.
- Пагинация/сортировка/фильтры — единый переиспользуемый механизм через `Depends`,
  не копипаста по маршрутам.

## Тестирование

- TDD для всей бизнес-логики — действует из глобальных правил; проектная
  конкретика ниже.
- Пирамида: unit (сервисы с подменёнными repository-протоколами, без БД) →
  интеграционные (repositories против PostgreSQL в testcontainers; API через
  `httpx.AsyncClient` + ASGI) → smoke compose-стека в CI.
- Покрытие: порог CI — 50%; целевой ориентир для сервисного и repository-слоя — ≥80%.
- Имена тестов — английские, по сценарию:
  `predict_volatility_returns_404_for_unknown_ticker`.

## Коммиты

Conventional Commits: `type(scope)` на английском, описание — на русском.
Запрет AI-атрибуции действует из глобальных правил.

## Команды

Раздел отражает реальное состояние репозитория и расширяется вместе с кодом.

Все команды запускаются из корня репозитория: mypy и pytest подхватывают конфиг
из корневого `pyproject.toml`, версия Python пиннится `.python-version` (3.12).

- `uv sync --project packages/stocklens-core` — окружение пакета core.
- `uv run --project packages/stocklens-core pytest packages/stocklens-core/tests` — тесты core.
- `uv run --project packages/stocklens-core mypy packages/stocklens-core/src packages/stocklens-core/tests` — типизация core.
- `uv sync` — корневое окружение (alembic, psycopg, testcontainers, stocklens-core editable).
- `docker compose up -d db` — локальный PostgreSQL 16 (loopback 5432).
- `DATABASE_URL=postgresql+psycopg://…@localhost:5432/stocklens uv run alembic upgrade head` — миграции.
- `uv run pytest tests -m integration` — интеграционные тесты миграций (нужен Docker).
  **Всегда с явным путём `tests`**: голый `uv run pytest` из корня соберёт по `testpaths`
  ещё и тесты пакетов в чужом для них окружении.
- `uv run mypy alembic tests` — типизация корневого слоя (alembic + интеграционные тесты).
- `uv sync --project services/ingestor` / `uv run --project services/ingestor pytest services/ingestor/tests`
  / `… mypy services/ingestor/src services/ingestor/tests` — сервис ingestor.
- `uv sync --project services/api` / `uv run --project services/api pytest services/api/tests`
  / `… mypy services/api/src services/api/tests` — сервис api (async, testcontainers PG+Redis).
- `docker compose up -d --build` — стек: db → migrations → ingestor → redis → api.
- `uvx ruff check .` / `uvx ruff format --check .` — линт всего репозитория.
- Python субпроектов пиннится локальным `.python-version` (3.12) в каждом
  пакете/сервисе — `uv sync --project` не наследует корневой пин.
- Целевой набор после появления сервисов: `docker compose up --build` (весь стек),
  аналогичные `uv … --project services/<имя>` команды на сервис.
