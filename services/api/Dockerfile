FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /build

# venv создаётся сразу по финальному пути /app/.venv: иначе shebang консольных
# скриптов (uvicorn) указывал бы на путь сборки и ломался после COPY в runtime.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# stocklens-core и stocklens-ml копируются первыми — path-зависимости до uv sync
# (stocklens-ml даёт единый расчёт фич с обучением для ML-serving, §8.5).
COPY packages/stocklens-core /build/packages/stocklens-core
COPY ml /build/ml
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

COPY --from=builder /app/.venv /app/.venv

EXPOSE 8000

CMD ["uvicorn", "api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
