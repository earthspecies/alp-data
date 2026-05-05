# ESP-Data Dashboard — Implementation Plan

Conference demo dashboard for `esp_data`. Interactive exploration of curated bioacoustic datasets, with an aggregate landing page and an experimental NL→SQL tab.

## Audience & demo context

- Researchers, ML engineers; possible (unlikely) funders
- Single laptop, booth setup, low concurrency, no auth
- Live demo — no canned fallback mode ("let it break")

## Scope

- **Aggregate** landing across all public datasets (`license != private`)
- **Curated 5–6 datasets** for deep-dive exploration: iNaturalist (preferred) and/or Xeno-Canto, AudioSet, plus a few species-specific datasets

## Repo layout

- New top-level `dashboard/` directory, sibling to `esp_data/`
- Precompute scripts under `scripts/dashboard/`

## Stack

- **Reflex** frontend + **FastAPI** backend in a single container
- **Recharts** for visualization (native to Reflex)
- **DuckDB** single-file embedded database
- **LRU cache** on hot backend queries
- **Claude Sonnet 4.6** for NL→SQL

## Hosting

- **Google Cloud Run**, `min-instances=1` (no cold start)
- Single container, image-bundled DuckDB + audio + spectrograms
- Manual deploy via `gcloud run deploy --source`
- Cloud Run stdout logs only — no Sentry / structured logs

## Data layer

- One DuckDB table per dataset (schemas vary too much to unify)
- Plus a `common` view stitching shared columns: taxonomy (kingdom/phylum/class/order/family/genus/species), duration, license, lat/lon, dataset_name
- DuckDB file bundled inside container image
- Read-only connection used by the NL→SQL tab

## Precompute pipeline

Single offline script: `scripts/dashboard/build_assets.py`.

### Aggregate stats (all public datasets)

Hybrid source:
- (A) Reuse existing `manifest.json` files in each dataset's `data_root` for counts, durations, file lists (fast)
- (B) Loader dataframes for taxonomy/label rollups — project only needed columns (`taxonomic_name`, `family`, `order`, `license`, `latitudeDecimal`, `longitudeDecimal`)

### Per-dataset assets (curated 6)

For each curated dataset:
1. Load via `esp_data` loader
2. Pick 10 random samples, **fixed seed** for reproducibility
3. For **strong-label** datasets (selection tables): pick events ≥ 2s
4. For **weak-label** datasets: raw file, ≥ 2s, **center-crop to 30s**
5. Compute log-mel spectrogram → PNG (per-dataset spectrogram params: n_fft, hop, n_mels, fmin, fmax — different sample rates / frequency ranges across taxa)
6. Transcode audio for browser (small mp3/ogg)
7. Write per-dataset manifest JSON

## Landing page

- **Primary**: aggregate stats — total recordings, total hours, # species, # datasets, # families, # countries
- **Secondary (hero viz)**: interactive **sunburst** taxonomy chart (kingdom → phylum → class → order → family) with click-drill-down

## Per-dataset view

Panels for each of the curated 6:

- **(a)** Metadata header — name, citation, license, # samples, hours
- **(b)** Class distribution — top-50 by duration, "other" bucket. For taxonomy-heavy datasets (e.g. iNaturalist, Xeno-Canto), aggregate at **family** or **order** level
- **(d)** Duration histogram (cheap — manifests already store durations)
- **(e)** Audio + spectrogram player — 10 fixed-seed random samples, pre-rendered

## "Talk to data" tab — BUILD LAST

Build only after the essentials (landing page, per-dataset views, audio player) are solid.

- Separate tab in the UI
- NL → DuckDB SQL via **Claude Sonnet 4.6**
- Read-only DuckDB connection enforced
- Full schema injected in system prompt (sourced from `claude_plans_todos/dataset_schemas.md`)
- Auto-run generated SQL, **show the SQL** above results
- Show error message on SQL failure (no retry loop)
- Auto-charted results via **Recharts** with simple shape heuristic:
  - 1 categorical + 1 numeric → bar
  - 2 numeric → scatter
  - time series → line
- Clickable **example query chips** to seed users
- No rate limit / no cost guard (low traffic, trusted)

## Build / deploy

- Local `docker build`
- `gcloud run deploy --source` manually before the conference

## Implementation order

1. Skeleton: `dashboard/` Reflex + FastAPI single-container scaffold, deployable to Cloud Run with a hello-world page
2. Precompute pipeline: `scripts/dashboard/build_assets.py` for aggregate stats → DuckDB
3. Landing page: aggregate stats panel + sunburst (Recharts)
4. Precompute pipeline: per-dataset assets (sampling, cropping, spectrograms, transcoded audio, per-dataset manifests)
5. Per-dataset views: metadata, class distribution, duration histogram, audio+spectrogram player
6. Polish: caching, layout, loading states
7. **NL→SQL tab** (last)
8. Final deploy + smoke test from booth wifi

## Deferred / known risks

- Per-record CC license filtering not done — small risk that an NC-licensed sample plays at the booth
- No demo-safe canned-response mode — if the LLM call or Cloud Run hiccups during demo, the tab visibly fails
- AnimalSpeak schema discrepancy (4 declared cols vs 46 actual) noted in `dataset_schemas.md` — may need handling when building DuckDB tables
- A few datasets have empty-name index columns (ArcticBirdSounds, NocturnalBirdMigration) — strip on ingest
