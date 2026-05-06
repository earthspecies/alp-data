"""Offline precompute pipeline for the esp-data dashboard.

Currently implements step 2a only: list the public dataset inventory and
verify each dataset's ``manifest.json`` is reachable on the canonical
bucket. Subsequent steps will extend this script to populate a DuckDB
file (`scripts/dashboard/build/dashboard.duckdb`) with per-dataset tables
and a `common` view, and to render audio + spectrogram assets for the
curated subset of datasets surfaced in the per-dataset views.

Usage
-----
List the inventory only::

    uv run python -m scripts.dashboard.build_assets list

Probe each manifest URL (default ``--workers 8``)::

    uv run python -m scripts.dashboard.build_assets check-manifests
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb
import librosa
import numpy as np
import polars as pl

from esp_data import datasets as ds_module
from esp_data.backends.polars_backend import PolarsBackend
from esp_data.io import filesystem_from_path
from scripts.dashboard.inventory import (
    DatasetInventoryEntry,
    build_inventory,
)

# Candidate columns we project from each dataset's CSV. Order matters for
# `_first_present`: we pick the first column that exists in the dataset.
_TAXONOMY_LEVELS: tuple[str, ...] = (
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
)
_SPECIES_COLUMNS: tuple[str, ...] = (
    "species_scientific",
    "canonical_name",
    "binomial_name",
    "scientific_name_unified",
)
_LICENSE_COLUMNS: tuple[str, ...] = ("license", "media_license")
_COUNTRY_COLUMNS: tuple[str, ...] = ("country",)
_LAT_COLUMNS: tuple[str, ...] = ("latitudeDecimal", "lat")
_LON_COLUMNS: tuple[str, ...] = ("longitudeDecimal", "lng", "lon")

DEFAULT_DB_PATH = Path("dashboard/assets/dashboard.duckdb")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dashboard.build_assets")


def _read_manifest(url: str) -> dict[str, Any] | None:
    """Read a ``manifest.json`` from local or cloud storage.

    Parameters
    ----------
    url : str
        Fully qualified URL to a JSON file (e.g. ``gs://...``).

    Returns
    -------
    dict[str, Any] or None
        Parsed manifest, or ``None`` if the file is missing or unreadable.
    """
    fs = filesystem_from_path(url)
    try:
        with fs.open(url, "rb") as f:
            return json.loads(f.read())
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("read %s failed: %s", url, exc)
        return None


def _probe_one(entry: DatasetInventoryEntry) -> dict[str, Any]:
    """Probe a single dataset's manifest and return a status row.

    Parameters
    ----------
    entry : DatasetInventoryEntry
        The inventory record to probe.

    Returns
    -------
    dict[str, Any]
        A row with the inventory fields plus ``found``, ``num_files``,
        ``total_duration_hours``, and ``mean_duration_seconds``.
    """
    row = asdict(entry)
    manifest = None
    hit_url: str | None = None
    for url in entry.manifest_candidates:
        manifest = _read_manifest(url)
        if manifest is not None:
            hit_url = url
            break
    if manifest is None:
        row.update(found=False, num_files=None, total_duration_hours=None, manifest_url=None)
        return row
    stats = manifest.get("audio_stats", {})
    total_s = float(stats.get("total_duration_seconds", 0.0))
    row.update(
        found=True,
        manifest_url=hit_url,
        num_files=stats.get("num_files"),
        total_duration_hours=total_s / 3600.0,
        mean_duration_seconds=stats.get("mean_duration_seconds"),
    )
    return row


def cmd_list() -> None:
    """Print the inventory in JSON, one entry per line.

    Used as a quick sanity check that we have the right list of public
    datasets and that manifest URLs look reasonable.
    """
    for entry in build_inventory():
        print(json.dumps(asdict(entry)))


def _probe_all(workers: int) -> list[dict[str, Any]]:
    """Probe every dataset's manifest in parallel.

    Parameters
    ----------
    workers : int
        Maximum number of concurrent GCS reads.

    Returns
    -------
    list[dict[str, Any]]
        One dict per inventory entry, sorted by `info_name`. See
        `_probe_one` for the row schema.
    """
    entries = build_inventory()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_probe_one, e): e for e in entries}
        for fut in as_completed(futures):
            rows.append(fut.result())
    rows.sort(key=lambda r: r["info_name"])
    return rows


def cmd_build_stats(workers: int, db_path: Path) -> None:
    """Probe all manifests and persist them to DuckDB.

    Creates (or replaces) the table `dataset_manifest_stats` in the
    DuckDB file at `db_path`. Each row corresponds to one public dataset
    in `scripts.dashboard.inventory.build_inventory`.

    Parameters
    ----------
    workers : int
        Concurrency for the manifest probe.
    db_path : Path
        Path to the DuckDB file to write. Parent directory is created if
        missing. The file is opened with `read_only=False`.
    """
    rows = _probe_all(workers=workers)
    found_rows = [r for r in rows if r["found"]]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE OR REPLACE TABLE dataset_manifest_stats (
                info_name              VARCHAR PRIMARY KEY,
                class_name             VARCHAR,
                version                VARCHAR,
                license                VARCHAR,
                data_root              VARCHAR,
                manifest_url           VARCHAR,
                num_files              BIGINT,
                total_duration_seconds DOUBLE,
                mean_duration_seconds  DOUBLE
            )
            """
        )
        if found_rows:
            con.executemany(
                """
                INSERT INTO dataset_manifest_stats VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["info_name"],
                        r["class_name"],
                        r["version"],
                        r["license"],
                        r["data_root"],
                        r["manifest_url"],
                        r["num_files"],
                        (r.get("total_duration_hours") or 0.0) * 3600.0,
                        r.get("mean_duration_seconds"),
                    )
                    for r in found_rows
                ],
            )
        # Sanity totals via DuckDB.
        agg = con.execute(
            """
            SELECT COUNT(*)                                      AS n_datasets,
                   SUM(num_files)                                AS n_files,
                   SUM(total_duration_seconds) / 3600.0          AS total_hours
            FROM dataset_manifest_stats
            """
        ).fetchone()
    finally:
        con.close()

    logger.info(
        "Wrote %s — %d datasets, %s files, %.1f h",
        db_path,
        agg[0],
        f"{agg[1]:,}" if agg[1] is not None else "0",
        agg[2] or 0.0,
    )
    missing = [r["info_name"] for r in rows if not r["found"]]
    if missing:
        logger.warning("Skipped (manifest not found): %s", ", ".join(missing))


def _first_present(backend: PolarsBackend, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column that exists in `backend`.

    Parameters
    ----------
    backend : PolarsBackend
        The backend to probe (typically streaming, so this only reads
        the CSV schema).
    candidates : tuple of str
        Column names to check, in order of preference.

    Returns
    -------
    str or None
        The first present column name, or `None` if none match.
    """
    for col in candidates:
        if backend.column_exists(col):
            return col
    return None


def _scan_dataset(entry: DatasetInventoryEntry) -> dict[str, Any]:
    """Compute per-dataset card stats and joint-taxonomy rows.

    Loads the dataset's first split CSV via `PolarsBackend` in streaming
    mode, projects only the columns needed for the dashboard, and
    materializes that slim frame. Returns a dict carrying:

    - the card-stats row for `dataset_card_stats`
    - a list of joint-taxonomy rows for `dataset_taxonomy` (may be empty)

    Parameters
    ----------
    entry : DatasetInventoryEntry
        Inventory entry describing the dataset to scan.

    Returns
    -------
    dict[str, Any]
        With keys ``card`` (dict | None) and ``taxonomy`` (list[dict]).
        ``card`` is `None` if the CSV could not be loaded.
    """
    cls = getattr(ds_module, entry.class_name)
    split_paths = getattr(cls.info, "split_paths", None) or {}
    if not split_paths:
        return {"card": None, "taxonomy": []}
    first_split = next(iter(split_paths.values()))

    try:
        be = PolarsBackend.from_csv(first_split, streaming=True)
    except Exception as exc:
        logger.warning("[%s] scan_csv failed: %s", entry.info_name, exc)
        return {"card": None, "taxonomy": []}

    species_col = _first_present(be, _SPECIES_COLUMNS)
    license_col = _first_present(be, _LICENSE_COLUMNS)
    country_col = _first_present(be, _COUNTRY_COLUMNS)
    lat_col = _first_present(be, _LAT_COLUMNS)
    lon_col = _first_present(be, _LON_COLUMNS)
    taxonomy_present = [c for c in _TAXONOMY_LEVELS if be.column_exists(c)]

    keep: list[str] = []
    for col in (species_col, license_col, country_col, lat_col, lon_col):
        if col and col not in keep:
            keep.append(col)
    for col in taxonomy_present:
        if col not in keep:
            keep.append(col)

    if not keep:
        # Nothing to project → just count rows for the card.
        n_rows = be.unwrap.select(pl.len()).collect().item()
        card = {
            "info_name": entry.info_name,
            "n_recordings": n_rows,
            "n_species": 0,
            "n_genera": 0,
            "n_families": 0,
            "n_orders": 0,
            "n_countries": 0,
            "dominant_license": None,
            "has_geo": False,
        }
        return {"card": card, "taxonomy": []}

    # Project + materialize the slim frame once. Subsequent calls reuse it.
    slim = be.select_columns(keep).collect()

    n_species = len(slim.get_unique(species_col)) if species_col else 0
    n_genera = len(slim.get_unique("genus")) if "genus" in keep else 0
    n_families = len(slim.get_unique("family")) if "family" in keep else 0
    n_orders = len(slim.get_unique("order")) if "order" in keep else 0
    n_countries = len(slim.get_unique(country_col)) if country_col else 0

    dominant_license = None
    if license_col:
        hist = slim.histogram(license_col)
        if hist:
            dominant_license = max(hist.items(), key=lambda kv: kv[1])[0]

    n_rows = len(slim)
    card = {
        "info_name": entry.info_name,
        "n_recordings": n_rows,
        "n_species": n_species,
        "n_genera": n_genera,
        "n_families": n_families,
        "n_orders": n_orders,
        "n_countries": n_countries,
        "dominant_license": dominant_license,
        "has_geo": bool(lat_col and lon_col),
    }

    # Joint-taxonomy rollup for the sunburst — only useful if at least
    # kingdom/family are present. We materialize on the slim frame.
    taxonomy_rows: list[dict[str, Any]] = []
    if "kingdom" in keep and "family" in keep:
        levels = [c for c in ("kingdom", "phylum", "class", "order", "family") if c in keep]
        df = slim.unwrap
        joint = df.group_by(levels).agg(pl.len().alias("n_recordings"))
        for row in joint.iter_rows(named=True):
            taxonomy_rows.append({"info_name": entry.info_name, **row})

    logger.info(
        "[%s] rows=%d species=%d families=%d countries=%d taxonomy_rows=%d",
        entry.info_name,
        n_rows,
        n_species,
        n_families,
        n_countries,
        len(taxonomy_rows),
    )
    return {"card": card, "taxonomy": taxonomy_rows}


def cmd_build_cards(workers: int, db_path: Path, only: list[str] | None) -> None:
    """Scan each public dataset and persist card + taxonomy tables.

    Writes (or replaces) two DuckDB tables in `db_path`:

    - ``dataset_card_stats`` — per-dataset count-distincts + dominant
      license + geo flag.
    - ``dataset_taxonomy`` — joint counts at (kingdom, phylum, class,
      order, family) for datasets with the full taxonomic hierarchy.

    Parameters
    ----------
    workers : int
        Concurrency for the per-dataset scans.
    db_path : Path
        DuckDB file path. Must already exist (built by `build-stats`).
    only : list[str] or None
        Optional whitelist of `info_name` values. If `None`, scans every
        non-private dataset.

    Raises
    ------
    SystemExit
        If `only` is provided but matches no inventory entries, or if
        `db_path` does not exist (run `build-stats` first to create it).
    """
    inv = build_inventory()
    if only:
        inv = [e for e in inv if e.info_name in set(only)]
        if not inv:
            raise SystemExit(f"No inventory entries match --only {only}")

    cards: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_dataset, e): e for e in inv}
        for fut in as_completed(futures):
            result = fut.result()
            if result["card"] is not None:
                cards.append(result["card"])
            taxonomy_rows.extend(result["taxonomy"])

    if not db_path.exists():
        raise SystemExit(f"{db_path} does not exist. Run `build-stats` first to create it.")

    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE OR REPLACE TABLE dataset_card_stats (
                info_name        VARCHAR PRIMARY KEY,
                n_recordings     BIGINT,
                n_species        BIGINT,
                n_genera         BIGINT,
                n_families       BIGINT,
                n_orders         BIGINT,
                n_countries      BIGINT,
                dominant_license VARCHAR,
                has_geo          BOOLEAN
            )
            """
        )
        if cards:
            con.executemany(
                """
                INSERT INTO dataset_card_stats VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c["info_name"],
                        c["n_recordings"],
                        c["n_species"],
                        c["n_genera"],
                        c["n_families"],
                        c["n_orders"],
                        c["n_countries"],
                        c["dominant_license"],
                        c["has_geo"],
                    )
                    for c in cards
                ],
            )

        con.execute(
            """
            CREATE OR REPLACE TABLE dataset_taxonomy (
                info_name    VARCHAR,
                kingdom      VARCHAR,
                phylum       VARCHAR,
                "class"      VARCHAR,
                "order"      VARCHAR,
                family       VARCHAR,
                n_recordings BIGINT
            )
            """
        )
        if taxonomy_rows:
            con.executemany(
                """
                INSERT INTO dataset_taxonomy
                    (info_name, kingdom, phylum, "class", "order", family, n_recordings)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["info_name"],
                        r.get("kingdom"),
                        r.get("phylum"),
                        r.get("class"),
                        r.get("order"),
                        r.get("family"),
                        r["n_recordings"],
                    )
                    for r in taxonomy_rows
                ],
            )

        n_cards, n_tax = con.execute(
            "SELECT (SELECT COUNT(*) FROM dataset_card_stats), "
            "(SELECT COUNT(*) FROM dataset_taxonomy)"
        ).fetchone()
    finally:
        con.close()

    logger.info(
        "Wrote %s — %d card rows, %d taxonomy rows",
        db_path,
        n_cards,
        n_tax,
    )


def _pick_present_column(row: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    """Return the first column in `candidates` whose row value is non-empty.

    Parameters
    ----------
    row : dict[str, Any]
        A single dataset row, as yielded by a dataset's `__getitem__`.
    candidates : tuple of str
        Column names to try, in priority order.

    Returns
    -------
    str or None
        The first candidate whose value in `row` is truthy and not the
        empty string, else `None`.
    """
    for col in candidates:
        val = row.get(col)
        if val is not None and val != "":
            return col
    return None


def _process_one_sample(
    audio: np.ndarray, sr: int, target_sr: int, cfg: dict[str, Any]
) -> np.ndarray:
    """Resample, mono-mix and crop a clip into a presentation-ready array.

    Parameters
    ----------
    audio : np.ndarray
        Source samples (mono or stereo, any dtype).
    sr : int
        Source sample rate.
    target_sr : int
        Sample rate to resample to (also used for spectrogram + mp3).
    cfg : dict[str, Any]
        Per-dataset config (used only for `MAX_DURATION_S` here, but
        kept as a hook for future per-dataset tuning).

    Returns
    -------
    np.ndarray
        Float32 mono audio at `target_sr`, center-cropped to at most
        `MAX_DURATION_S` seconds.
    """
    from scripts.dashboard.render_samples import center_crop
    from scripts.dashboard.sample_config import MAX_DURATION_S

    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=tuple(range(1, audio.ndim)))
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return center_crop(audio, target_sr, MAX_DURATION_S)


def _build_one_dataset_samples(
    info_name: str, cfg: dict[str, Any], assets_root: Path
) -> list[dict[str, Any]]:
    """Render samples + metadata for one curated dataset.

    Instantiates the dataset class, draws random rows with a fixed
    seed, and writes audio + spectrogram artifacts under
    ``<assets_root>/<info_name>/``. Rows that are too short or fail to
    decode are skipped; we keep drawing until `SAMPLES_PER_DATASET` are
    valid (capped by the random pool size).

    Parameters
    ----------
    info_name : str
        Key into `SAMPLE_CONFIG`.
    cfg : dict[str, Any]
        The config entry for this dataset.
    assets_root : Path
        Parent directory under which a `<info_name>/` folder is created.

    Returns
    -------
    list[dict[str, Any]]
        One dict per successfully written sample, ready to be inserted
        into the `dataset_samples` DuckDB table.
    """
    import random

    from scripts.dashboard.render_samples import render_log_mel_png, transcode_mp3
    from scripts.dashboard.sample_config import (
        MIN_DURATION_S,
        SAMPLE_POOL_FACTOR,
        SAMPLE_SEED,
        SAMPLES_PER_DATASET,
    )

    cls = getattr(ds_module, cfg["class_name"])
    ds = cls(split=cfg["split"], sample_rate=cfg["target_sr"])
    n_total = len(ds)
    if n_total == 0:
        logger.warning("[%s] empty dataset, skipping", info_name)
        return []

    rng = random.Random(SAMPLE_SEED)
    pool_size = min(n_total, SAMPLES_PER_DATASET * SAMPLE_POOL_FACTOR)
    indices = rng.sample(range(n_total), pool_size)

    out_dir = assets_root / info_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear any prior artifacts so re-runs are reproducible.
    for old in out_dir.iterdir():
        old.unlink()

    written: list[dict[str, Any]] = []
    for raw_idx in indices:
        if len(written) >= SAMPLES_PER_DATASET:
            break
        try:
            row = ds[raw_idx]
        except Exception as exc:
            logger.warning("[%s] row %d read failed: %s", info_name, raw_idx, exc)
            continue

        audio = row.get("audio")
        sr = row.get("sample_rate")
        if audio is None or sr is None:
            continue

        try:
            clip = _process_one_sample(audio, sr, cfg["target_sr"], cfg)
        except Exception as exc:
            logger.warning("[%s] row %d processing failed: %s", info_name, raw_idx, exc)
            continue

        if len(clip) / cfg["target_sr"] < MIN_DURATION_S:
            continue

        sample_idx = len(written)
        mp3_path = out_dir / f"{sample_idx:02d}.mp3"
        png_path = out_dir / f"{sample_idx:02d}.png"
        try:
            transcode_mp3(clip, cfg["target_sr"], mp3_path)
            render_log_mel_png(
                clip,
                cfg["target_sr"],
                png_path,
                n_fft=cfg["mel_n_fft"],
                hop=cfg["mel_hop"],
                n_mels=cfg["mel_n_mels"],
                fmin=cfg["mel_fmin"],
                fmax=cfg["mel_fmax"],
            )
        except Exception as exc:
            logger.warning("[%s] render failed for row %d: %s", info_name, raw_idx, exc)
            mp3_path.unlink(missing_ok=True)
            png_path.unlink(missing_ok=True)
            continue

        label = row.get(cfg.get("label_column") or "") or ""
        license_text = row.get(cfg.get("license_column") or "") or ""
        url = row.get(cfg.get("url_column") or "") or "" if cfg.get("url_column") else ""
        written.append(
            {
                "info_name": info_name,
                "sample_idx": sample_idx,
                "label": str(label),
                "license": str(license_text),
                "duration_s": float(len(clip) / cfg["target_sr"]),
                "audio_rel": f"samples/{info_name}/{mp3_path.name}",
                "spec_rel": f"samples/{info_name}/{png_path.name}",
                "source_url": str(url),
            }
        )
        logger.info(
            "[%s] sample %02d: %s (%.1fs)",
            info_name,
            sample_idx,
            label or "?",
            len(clip) / cfg["target_sr"],
        )

    if len(written) < SAMPLES_PER_DATASET:
        logger.warning(
            "[%s] only %d/%d samples written", info_name, len(written), SAMPLES_PER_DATASET
        )
    return written


def cmd_build_samples(db_path: Path, only: list[str] | None) -> None:
    """Render audio + spectrogram assets for the curated datasets.

    Iterates `SAMPLE_CONFIG`, writes per-sample mp3/png pairs under
    ``dashboard/assets/samples/<info_name>/``, and persists a flat
    `dataset_samples` table to `db_path`.

    Parameters
    ----------
    db_path : Path
        DuckDB file path. Must already exist (built by `build-stats`).
    only : list[str] or None
        Optional whitelist of `info_name` keys from `SAMPLE_CONFIG`.

    Raises
    ------
    SystemExit
        If `only` is provided but matches no entries in `SAMPLE_CONFIG`,
        or if `db_path` does not exist.
    """
    from scripts.dashboard.sample_config import SAMPLE_CONFIG

    selected = SAMPLE_CONFIG
    if only:
        wanted = set(only)
        selected = {k: v for k, v in SAMPLE_CONFIG.items() if k in wanted}
        if not selected:
            raise SystemExit(f"No SAMPLE_CONFIG entries match --only {only}")

    if not db_path.exists():
        raise SystemExit(f"{db_path} does not exist. Run `build-stats` first to create it.")

    assets_root = db_path.parent / "samples"
    assets_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for info_name, cfg in selected.items():
        logger.info("=== building samples for %s ===", info_name)
        rows = _build_one_dataset_samples(info_name, cfg, assets_root)
        all_rows.extend(rows)

    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE OR REPLACE TABLE dataset_samples (
                info_name   VARCHAR,
                sample_idx  INTEGER,
                label       VARCHAR,
                license     VARCHAR,
                duration_s  DOUBLE,
                audio_rel   VARCHAR,
                spec_rel    VARCHAR,
                source_url  VARCHAR,
                PRIMARY KEY (info_name, sample_idx)
            )
            """
        )
        if all_rows:
            con.executemany(
                """
                INSERT INTO dataset_samples VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["info_name"],
                        r["sample_idx"],
                        r["label"],
                        r["license"],
                        r["duration_s"],
                        r["audio_rel"],
                        r["spec_rel"],
                        r["source_url"],
                    )
                    for r in all_rows
                ],
            )
        n = con.execute("SELECT COUNT(*) FROM dataset_samples").fetchone()[0]
    finally:
        con.close()
    logger.info("Wrote %d sample rows to %s", n, db_path)


def cmd_check_manifests(workers: int) -> None:
    """Probe every dataset's manifest in parallel and print a summary.

    Parameters
    ----------
    workers : int
        Maximum number of concurrent GCS reads.
    """
    rows = _probe_all(workers=workers)
    found = 0
    total_h = 0.0
    print(f"{'dataset':32s} {'version':10s} {'found':5s} {'#files':>10s} {'hours':>10s}")
    print("-" * 72)
    for r in rows:
        is_found = r["found"]
        if is_found:
            found += 1
            total_h += r.get("total_duration_hours") or 0.0
        files_str = f"{r['num_files']:>10}" if r["num_files"] is not None else "-" * 10
        hours_str = (
            f"{r['total_duration_hours']:>10.1f}"
            if r["total_duration_hours"] is not None
            else "-" * 10
        )
        print(f"{r['info_name']:32s} {r['version']:10s} {str(is_found):5s} {files_str} {hours_str}")
    print("-" * 72)
    print(f"{found}/{len(rows)} manifests found, {total_h:.1f} h total")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="Print the dataset inventory as JSON lines.")
    p_check = sub.add_parser("check-manifests", help="Probe each dataset's manifest.json on GCS.")
    p_check.add_argument("--workers", type=int, default=8)
    p_build = sub.add_parser(
        "build-stats",
        help="Probe manifests and write dataset_manifest_stats to DuckDB.",
    )
    p_build.add_argument("--workers", type=int, default=8)
    p_build.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p_cards = sub.add_parser(
        "build-cards",
        help=(
            "Scan each dataset CSV and write dataset_card_stats + "
            "dataset_taxonomy tables to DuckDB."
        ),
    )
    p_cards.add_argument("--workers", type=int, default=4)
    p_cards.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p_cards.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional info_name whitelist (e.g. inaturalist xeno-canto).",
    )
    p_samples = sub.add_parser(
        "build-samples",
        help=(
            "Render audio + spectrogram samples for the curated datasets "
            "(see scripts.dashboard.sample_config.SAMPLE_CONFIG)."
        ),
    )
    p_samples.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p_samples.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional info_name whitelist into SAMPLE_CONFIG.",
    )
    args = parser.parse_args()

    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "check-manifests":
        cmd_check_manifests(workers=args.workers)
    elif args.cmd == "build-stats":
        cmd_build_stats(workers=args.workers, db_path=args.db)
    elif args.cmd == "build-cards":
        cmd_build_cards(workers=args.workers, db_path=args.db, only=args.only)
    elif args.cmd == "build-samples":
        cmd_build_samples(db_path=args.db, only=args.only)


if __name__ == "__main__":
    main()
