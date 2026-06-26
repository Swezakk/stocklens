# dashboard — веб-интерфейс StockLens

Streamlit-дашборд владельца: рынок, бумаги, новости, прогнозы, портфель и мониторинг
в едином защищённом паролем интерфейсе.

## Назначение

Шесть страниц с графиками Plotly поверх данных API. Парольный гейт владельца (= логин в API)
до построения навигации; пароль удерживается в `st.session_state` для проактивного refresh
JWT. Дашборд только отображает решения сервера — бизнес-правила и статусы он не принимает.

## Место в архитектуре

Ходит **только в API по HTTP** — прямого доступа к БД нет (импорт ORM/коннект к PostgreSQL
запрещён). `ApiClient` поверх `httpx` с провайдером токена и хуком на 401. Знает лишь
`API_URL` и `AUTH_OWNER_USERNAME`; секретов в окружении нет. Контракт —
[docs/specs/2026-06-11-stocklens-design.md](../../docs/specs/2026-06-11-stocklens-design.md).

## Стек

Streamlit (`st.navigation`/`st.Page`) · Plotly · httpx · Pydantic v2 + pydantic-settings ·
тонкий CSS-слой (`assets/dashboard.css`).

## Запуск

В составе стека — `docker compose up -d --build` (зависит от `api`; слушает
`127.0.0.1:8501`). Запуск контейнера — `streamlit run src/dashboard/app.py`. Локально:

```bash
uv sync --project services/dashboard
API_URL=http://localhost:8000 uv run --project services/dashboard streamlit run services/dashboard/src/dashboard/app.py
```

## Тесты

```bash
uv run --project services/dashboard pytest services/dashboard/tests
uv run --project services/dashboard mypy services/dashboard/src services/dashboard/tests
```

## Структура

- `app.py` — точка входа: `set_page_config` → CSS → `require_auth` → лого → `st.navigation`
  из шести страниц.
- `auth.py` — парольный гейт `require_auth` и `TokenManager` (кэш JWT, проактивный refresh,
  single-flight, реактивный 401).
- `pages/` — шесть `render`-страниц: «Обзор» (`overview`), «Акции» (`stocks`),
  «Новости» (`news`), «Портфель» (`portfolio`), «Прогнозы» (`forecasts`),
  «Мониторинг» (`monitoring`).
- `api_client/` — `client` (httpx), `fetch` (кэш ресурса), `dto`, `errors`.
- `components/` — `sidebar`, `charts`, `kpi`, `filters`, `sentiment`, `layout`, `feedback`,
  `transforms`.
- `theme.py`, `settings.py` — оформление и конфигурация (pydantic-settings).
