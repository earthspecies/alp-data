# syntax=docker/dockerfile:1.6
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/dashboard/.venv \
    NODE_VERSION=20

# System deps + Node (Reflex needs node/bun for frontend export) + Caddy.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates xz-utils unzip git build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && curl -fsSL "https://github.com/caddyserver/caddy/releases/download/v2.8.4/caddy_2.8.4_linux_amd64.tar.gz" \
        | tar -xz -C /usr/local/bin caddy \
    && rm -rf /var/lib/apt/lists/*

# uv
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /uvx /usr/local/bin/

WORKDIR /app

# esp_data sources (dashboard depends on it via local path).
COPY pyproject.toml uv.lock README.md ./
COPY esp_data ./esp_data

# Dashboard project.
COPY dashboard ./dashboard
WORKDIR /app/dashboard
RUN uv sync --no-dev

# Build the Reflex frontend. REFLEX_API_URL controls where the frontend
# bundle expects the backend to live; for Cloud Run pass the public URL
# at build time (otherwise leave default for local container testing).
ARG REFLEX_API_URL=http://localhost:8080
ENV API_URL=${REFLEX_API_URL}
RUN uv run reflex init --template blank \
    && uv run reflex export --frontend-only --no-zip

# Caddy + entrypoint.
COPY dashboard/Caddyfile /etc/caddy/Caddyfile
COPY dashboard/start.sh /start.sh
RUN chmod +x /start.sh

ENV PORT=8080
EXPOSE 8080

CMD ["/start.sh"]
