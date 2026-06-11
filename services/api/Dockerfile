FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /build

# stocklens-core копируется первым — он нужен как path-зависимость до uv sync.
COPY packages/stocklens-core /build/packages/stocklens-core
COPY services/api/pyproject.toml /build/services/api/pyproject.toml
COPY services/api/uv.lock /build/services/api/uv.lock

# Исходники нужны перед sync: uv устанавливает пакет из локального src-дерева.
COPY services/api/src /build/services/api/src

# Устанавливаем только prod-зависимости; пакет собирается без editable-ссылки.
RUN uv sync \
    --frozen \
    --no-dev \
    --no-editable \
    --project services/api


FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /build/services/api/.venv /app/.venv

EXPOSE 8000

CMD ["uvicorn", "api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
