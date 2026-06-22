FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /build

# venv создаётся сразу по финальному пути /app/.venv: иначе shebang консольных
# скриптов (streamlit) указывал бы на путь сборки и ломался после COPY в runtime.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# stocklens-core копируется первым — он нужен как path-зависимость до uv sync.
COPY packages/stocklens-core /build/packages/stocklens-core
COPY services/dashboard/pyproject.toml /build/services/dashboard/pyproject.toml
COPY services/dashboard/uv.lock /build/services/dashboard/uv.lock

# Исходники нужны перед sync: uv устанавливает пакет из локального src-дерева.
COPY services/dashboard/src /build/services/dashboard/src

# Устанавливаем только prod-зависимости; пакет собирается без editable-ссылки.
RUN uv sync \
    --frozen \
    --no-dev \
    --no-editable \
    --project services/dashboard


FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# streamlit run принимает путь к файлу (не module-ref, как uvicorn --factory),
# а --no-editable ставит пакет в site-packages, не в src/. Поэтому исходники
# нужны в рантайме как файлы по пути src/dashboard/app.py.
COPY services/dashboard/src /app/src

# Тема (config.toml) и тонкий CSS-слой не входят в пакет — копируются явно.
COPY services/dashboard/.streamlit /app/.streamlit
COPY services/dashboard/assets /app/assets

EXPOSE 8501

# Streamlit отдаёт собственный healthcheck-эндпоинт /_stcore/health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).read() == b'ok' else 1)"

CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
