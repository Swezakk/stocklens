FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /build

# venv создаётся сразу по финальному пути /app/.venv (консистентно с другими сервисами).
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# stocklens-core копируется первым — он нужен как path-зависимость до uv sync.
COPY packages/stocklens-core /build/packages/stocklens-core
COPY services/bot/pyproject.toml /build/services/bot/pyproject.toml
COPY services/bot/uv.lock /build/services/bot/uv.lock

# Исходники бота нужны перед sync: uv устанавливает пакет из локального src-дерева
# с --no-editable, делая копию в site-packages venv.
COPY services/bot/src /build/services/bot/src

# Только prod-зависимости; пакет собирается без editable-ссылки.
RUN uv sync \
    --frozen \
    --no-dev \
    --no-editable \
    --project services/bot


FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Venv содержит bot и все prod-зависимости (aiogram, httpx, stocklens-core).
COPY --from=builder /app/.venv /app/.venv

CMD ["python", "-m", "bot"]
