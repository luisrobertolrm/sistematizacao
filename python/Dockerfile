# syntax=docker/dockerfile:1

# --- estagio 1: dependencias -------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# camada de dependencias em cache: so muda quando pyproject/lock mudarem
COPY pyproject.toml uv.lock* README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# --- estagio 2: runtime ------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    MPLCONFIGDIR=/tmp/matplotlib \
    DIR_MODELOS=/app/modelos \
    DIR_DADOS=/app/dados \
    DIR_RELATORIOS=/app/relatorios \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

RUN useradd --create-home --uid 1000 app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY modelos ./modelos
COPY dados ./dados

RUN mkdir -p /app/relatorios && chown -R app:app /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "sistematizacao.web:app", "--host", "0.0.0.0", "--port", "8000"]
