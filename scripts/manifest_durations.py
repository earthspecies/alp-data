"""Collect per-dataset audio durations from migrated manifest.json files.

For every dataset in ``DATASET_REGISTRY``, locate each version's
``manifest.json`` under the migrated bucket (default
``gs://esp-data-274503``) and write a CSV with the total audio duration.

Usage
-----
    uv run python scripts/manifest_durations.py \\
        --new-bucket esp-data-274503 \\
        --new-protocol gs \\
        --output dataset_durations.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from migrate_datasets import swap_root  # noqa: E402
from migrate_registry import (  # noqa: E402
    DATASET_REGISTRY,
    derive_data_root,
    get_dataset_class,
    get_version_configs,
)

from esp_data.io import filesystem_from_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("manifest_durations")

MAX_WORKERS = 16


def _read_manifest(path: str) -> dict | None:
    """Read a manifest.json from local or cloud storage.

    Parameters
    ----------
    path : str
        URL of the manifest file.

    Returns
    -------
    dict or None
        Parsed manifest payload, or ``None`` if the file is missing or
        unreadable.
    """
    fs = filesystem_from_path(path)
    try:
        with fs.open(path, "rb") as f:
            return json.loads(f.read())
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("read %s failed: %s", path, exc)
        return None


def _collect_dataset(
    dataset_name: str,
    new_bucket: str,
    new_protocol: str,
) -> list[dict]:
    """Resolve every version's manifest URL and extract duration stats.

    Parameters
    ----------
    dataset_name : str
        Registered ESP dataset name.
    new_bucket : str
        Destination bucket name (no scheme).
    new_protocol : str
        Destination scheme (``gs``, ``s3``, or ``r2``).

    Returns
    -------
    list of dict
        One row per dataset-version with duration stats. Versions whose
        manifest cannot be located are still included with ``found=False``.
    """
    rows: list[dict] = []
    registry_entry = DATASET_REGISTRY[dataset_name]
    try:
        dataset_class = get_dataset_class(dataset_name)
    except KeyError as exc:
        logger.warning("skip %s: %s", dataset_name, exc)
        return rows

    version_configs = get_version_configs(dataset_class, registry_entry)

    for version, vcfg in version_configs.items():
        split_paths = vcfg["split_paths"]
        first_split = next(iter(split_paths.values()))
        src_data_root = derive_data_root(first_split, vcfg, registry_entry)
        dst_data_root = swap_root(src_data_root, new_bucket, new_protocol)
        if not dst_data_root.endswith("/"):
            dst_data_root += "/"
        manifest_path = dst_data_root + "manifest.json"

        manifest = _read_manifest(manifest_path)
        row: dict = {
            "dataset": dataset_name,
            "version": version,
            "manifest_path": manifest_path,
            "found": manifest is not None,
            "num_files": "",
            "total_duration_seconds": "",
            "total_duration_hours": "",
            "mean_duration_seconds": "",
        }
        if manifest is not None:
            stats = manifest.get("audio_stats", {})
            total_s = float(stats.get("total_duration_seconds", 0.0))
            row.update(
                num_files=stats.get("num_files", 0),
                total_duration_seconds=total_s,
                total_duration_hours=total_s / 3600.0,
                mean_duration_seconds=stats.get("mean_duration_seconds", 0.0),
            )
            logger.info(
                "[%s/%s] %.1f h (%d files)",
                dataset_name,
                version,
                total_s / 3600.0,
                stats.get("num_files", 0),
            )
        else:
            logger.warning("[%s/%s] manifest missing: %s", dataset_name, version, manifest_path)
        rows.append(row)
    return rows


def main() -> None:
    """Command-line entry point.

    Iterates ``DATASET_REGISTRY``, reads each version's manifest from the
    destination bucket in parallel, and writes a CSV summary.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-bucket", default="esp-data-274503")
    parser.add_argument("--new-protocol", default="gs", choices=("gs", "r2", "s3"))
    parser.add_argument("--output", required=True, help="CSV output path.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional subset of datasets to query (default: all in registry).",
    )
    args = parser.parse_args()

    names = args.datasets or list(DATASET_REGISTRY.keys())
    logger.info(
        "Querying %d dataset(s) under %s://%s",
        len(names),
        args.new_protocol,
        args.new_bucket,
    )

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_collect_dataset, name, args.new_bucket, args.new_protocol): name
            for name in names
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                all_rows.extend(fut.result())
            except Exception as exc:
                logger.error("[%s] failed: %s", name, exc, exc_info=True)

    all_rows.sort(key=lambda r: (r["dataset"], r["version"]))

    fieldnames = [
        "dataset",
        "version",
        "found",
        "num_files",
        "total_duration_seconds",
        "total_duration_hours",
        "mean_duration_seconds",
        "manifest_path",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    found = sum(1 for r in all_rows if r["found"])
    total_h = sum(float(r["total_duration_hours"] or 0) for r in all_rows)
    logger.info(
        "Wrote %s — %d/%d versions found, %.1f h total",
        args.output,
        found,
        len(all_rows),
        total_h,
    )


if __name__ == "__main__":
    main()
