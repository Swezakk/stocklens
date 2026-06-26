# ingestor — сборщик рыночных и новостных данных

Синхронный батч-сервис: по расписанию собирает котировки MOEX, корпоративные данные,
курсы ЦБ и новости RSS, скорит тональность и пишет всё в БД.

## Назначение

Единственный write-путь в рыночные и новостные таблицы (`securities`, `candles`,
`dividends`, `splits`, `index_values`, `currency_rates`, `key_rates`, `news_*`). При старте
выполняет backfill (догоняет пропуск истории), затем работает планировщиком. Тональность
новостей скорится здесь же при сборе (RuBERT-tiny через ONNX) — API отдаёт готовые оценки.
Все записи — идемпотентный upsert по натуральному ключу; каждый запуск журналируется
в `collector_runs` (ошибка одного источника даёт статус `partial`, не валит остальные).

## Место в архитектуре

Пишет напрямую в PostgreSQL (синхронная SQLAlchemy 2.0). Внешние источники — MOEX ISS
(`iss_client`, вежливый клиент: ≤1 req/s, retry с backoff), CBR, RSS-ленты. С другими
сервисами не общается. **Синхронный осознанно** (батчевый сбор) — не переводить на async.
Контракт — [docs/specs/2026-06-11-stocklens-design.md](../../docs/specs/2026-06-11-stocklens-design.md).

## Стек

APScheduler 3.x (`BlockingScheduler`) · requests · SQLAlchemy 2.0 sync (psycopg) ·
RuBERT-tiny (sentiment, ONNX Runtime) · structlog · `responses` (HTTP-моки в тестах).

## Запуск

В составе стека — `docker compose up -d --build` (зависит от `migrations`; healthcheck по
свежести heartbeat-файла). Запуск контейнера — `python -m ingestor`. Локально:

```bash
uv sync --project services/ingestor
DATABASE_URL=postgresql+psycopg://…@localhost:5432/stocklens uv run --project services/ingestor python -m ingestor
```

## Тесты

```bash
uv run --project services/ingestor pytest services/ingestor/tests
uv run --project services/ingestor mypy services/ingestor/src services/ingestor/tests
```

## Структура

- `__main__.py` — точка входа: настройка → `wait_for_schema` → `run_backfill` → планировщик.
- `scheduler.py` — задания `BlockingScheduler` (Europe/Moscow): 10:00 утренний синк
  свечей+индекса, 23:55 вечерний backstop, 08:00 бумаги/дивиденды/сплиты, 13:00 курсы и
  ключевая ставка ЦБ, каждые 30 мин — новости, каждые 60 с — heartbeat.
- `collectors/` — `moex` (свечи, индекс, бумаги, дивиденды, сплиты), `cbr`, `rss`.
- `iss_client.py` — вежливый клиент MOEX ISS (rate-limit, retry).
- `sentiment.py` — `OnnxSentimentScorer` (RuBERT-tiny).
- `repositories.py` — upsert-запись по натуральным ключам.
- `backfill.py` — догон истории при старте. `run_journal.py` — `collector_runs`.
- `matching.py` / `aliases_seed.py` — привязка новостей к тикерам. `parsing.py`,
  `schema_wait.py`, `heartbeat.py`, `settings.py`, `logging_setup.py`.
