# esp-dashboard

Interactive dashboard for `esp_data`, built for a conference demo.

Stack: **Reflex** (frontend) + **FastAPI** (backend) in a single container,
**DuckDB** for embedded analytics, **Recharts** for visualization.

## Local development

From this directory:

```bash
uv sync
uv run reflex init   # one-time, generates web/ assets
uv run reflex run    # starts frontend (3000) + backend (8000)
```

Open <http://localhost:3000>. The FastAPI health route lives at
<http://localhost:8000/api/health>.

## Container (Cloud Run target)

The container bundles the Reflex frontend (static export) + the Reflex/FastAPI
backend, fronted by Caddy on `$PORT` (default `8080`). Caddy routes:

- `/` → static frontend (`/app/dashboard/.web/_static`)
- `/api/*` → FastAPI on `127.0.0.1:8000`
- `/_event*`, `/_upload*`, `/ping` → Reflex backend on `127.0.0.1:8000`

Build (from the **repo root**, not `dashboard/`):

```bash
docker build -f dashboard/Dockerfile -t esp-dashboard:dev .
docker run --rm -p 8080:8080 esp-dashboard:dev
# open http://localhost:8080
# curl http://localhost:8080/api/health
```

For a Cloud Run build, override `REFLEX_API_URL` so the frontend bundle
points at the deployed public URL:

```bash
docker build -f dashboard/Dockerfile \
  --build-arg REFLEX_API_URL=https://<service>-<hash>-<region>.a.run.app \
  -t esp-dashboard:cr .
```

## Status

- [x] Step 1a — scaffold (landing placeholder + health route)
- [x] Step 1b — local run verified
- [x] Step 1c — Docker container (Caddy + Reflex export, port 8080)
- [ ] Step 1d — Cloud Run deploy
- [ ] Step 2 — aggregate stats precompute → DuckDB
- [ ] Step 3 — landing page (stats + sunburst)
- [ ] Step 4 — per-dataset asset precompute
- [ ] Step 5 — per-dataset views
- [ ] Step 6 — polish, caching
- [ ] Step 7 — NL→SQL tab (last)

See `claude_plans_todos/dashboard_plan.md` at the repo root for the full
plan.
