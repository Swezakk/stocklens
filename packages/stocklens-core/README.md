# stocklens-core — общий пакет StockLens

Единый источник схемы данных, доменных перечислений и настроек подключений для всех
сервисов StockLens.

## Назначение

Устраняет дублирование контракта данных между сервисами: SQLAlchemy 2.0 ORM-модели,
доменные `StrEnum` (статусы, метки, типы) и базовые pydantic-settings живут здесь и
импортируются как `stocklens_core`. Любое расхождение схемы в сервисах — баг: дублировать
определения таблиц или статусов запрещено.

## Место в архитектуре

Устанавливается editable-зависимостью в `ingestor`, `api`, `ml` и корневое окружение
(Alembic-миграции). Сам пакет ни к чему не подключается и не содержит бизнес-логики —
только определения. Контракт —
[docs/specs/2026-06-11-stocklens-design.md](../../docs/specs/2026-06-11-stocklens-design.md).

## Стек

SQLAlchemy 2.0 (декларативные `Mapped`-модели, `Enum(native_enum=False)`) ·
Pydantic v2 + pydantic-settings · Python 3.12 `StrEnum`.

## Запуск

Пакет — библиотека, отдельного процесса нет. Подготовка окружения:

```bash
uv sync --project packages/stocklens-core
```

В сервисах подключается как зависимость их `pyproject.toml` (editable).

## Тесты

```bash
uv run --project packages/stocklens-core pytest packages/stocklens-core/tests
uv run --project packages/stocklens-core mypy packages/stocklens-core/src packages/stocklens-core/tests
```

## Структура

- `__init__.py` — публичный API: реэкспорт моделей, перечислений и `CoreSettings`.
- `enums.py` — доменные `StrEnum`: `CollectorRunStatus`, `SentimentLabel`, `PredictionKind`,
  `TrendDirection`, `Currency`, `AlertKind`.
- `settings.py` — `CoreSettings` (DSN PostgreSQL sync/async, Redis; pydantic-settings).
- `models/base.py` — декларативный `Base`.
- `models/market.py` — `Security`, `Candle`, `Dividend`, `Split`, `IndexValue`,
  `CurrencyRate`, `KeyRate`.
- `models/news.py` — `NewsArticle`, `NewsSentiment`, `NewsTicker`.
- `models/prediction.py` — `Prediction`. `models/operations.py` — `CollectorRun`.
- `models/portfolio.py` — `PortfolioPosition`, `BotSubscription`, `Watchlist`.
