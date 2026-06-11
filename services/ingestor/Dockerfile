FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /build

# stocklens-core копируется первым — он нужен как path-зависимость до uv sync.
COPY packages/stocklens-core /build/packages/stocklens-core
COPY services/ingestor/pyproject.toml /build/services/ingestor/pyproject.toml
COPY services/ingestor/uv.lock /build/services/ingestor/uv.lock
COPY alembic/ /build/alembic/
COPY alembic.ini /build/alembic.ini

# Исходники ingestor нужны перед sync: uv устанавливает пакет из локального
# src-дерева с --no-editable, делая копию в site-packages venv.
COPY services/ingestor/src /build/services/ingestor/src

# Устанавливаем только prod-зависимости; пакет собирается без editable-ссылки.
RUN uv sync \
    --frozen \
    --no-dev \
    --no-editable \
    --project services/ingestor


FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Venv содержит ingestor и все prod-зависимости (в т.ч. alembic).
COPY --from=builder /build/services/ingestor/.venv /app/.venv
# alembic/ и alembic.ini нужны сервису migrations (override CMD в compose),
# который работает из того же образа с командой "alembic upgrade head".
COPY --from=builder /build/alembic /app/alembic
COPY --from=builder /build/alembic.ini /app/alembic.ini

CMD ["python", "-m", "ingestor"]
