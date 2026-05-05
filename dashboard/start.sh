#!/bin/sh
set -e

cd /app/dashboard

# Reflex backend (FastAPI + state) on internal port 8000.
uv run reflex run --env prod --backend-only \
	--backend-host 0.0.0.0 --backend-port 8000 &

# Caddy serves the Reflex static frontend on $PORT and reverse-proxies
# /api, /_event, /_upload, /ping to the backend.
exec caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
