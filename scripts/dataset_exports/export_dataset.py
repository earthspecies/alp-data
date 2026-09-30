"""Freeze a configured dataset with `alp_data.export` and verify the result.

Reads a dataset YAML (the same `dataset:` / `concat:` / `chain:` format that
`dataset_from_config` accepts), exports it as a pack or as Hugging Face
parquet, and optionally reloads the pack to compare random samples with the
live dataset, in the calling process and through spawned worker processes.
Timings and sizes go to a JSON summary so a run doubles as a first benchmark.

Examples
--------
uv run python scripts/dataset_exports/export_dataset.py \\
    --config scripts/dataset_exports/configs/beans_validation_16k.yaml \\
    --out gs://esp-ci-cd-tests/esp-data-tests/exports/beans/validation-16k \\
    --num-workers 32 --verify 200 --verify-workers 4
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import multiprocessing as mp
import pickle
import random
import statistics
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import yaml

from alp_data.dataset import ChainedDatasetConfig, config_from_yaml, dataset_from_config
from alp_data.export import PackedDataset, pack, to_hf
from alp_data.export.columns import SOURCE_INDEX_COL
from alp_data.io import anypath, filesystem_from_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("export_dataset")


# --- comparison --------------------------------------------------------------


def _same(a: Any, b: Any, audio_atol: float) -> bool:  # noqa: ANN401
    """Whether two sample values are equal, allowing FLAC quantisation on audio.

    Parameters
    ----------
    a, b : Any
        The two values.
    audio_atol : float
        Absolute tolerance for float arrays.

    Returns
    -------
    bool
        True when the values match.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if np.issubdtype(a.dtype, np.floating):
            return bool(np.allclose(a, b, atol=audio_atol, rtol=0))
        return bool(np.array_equal(a, b))
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(a, b)
            return True
        except AssertionError:
            return False
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    try:
        return bool(a == b)
    except ValueError:
        return False


def compare_items(
    packed: dict[str, Any], live: dict[str, Any], audio_key: str, audio_atol: float
) -> list[str]:
    """Return the keys on which a packed sample differs from the live one.

    Parameters
    ----------
    packed, live : dict[str, Any]
        The two samples.
    audio_key : str
        Key whose comparison uses `audio_atol`.
    audio_atol : float
        Absolute tolerance for the audio array.

    Returns
    -------
    list[str]
        Differing keys, empty when the samples match.
    """
    diffs = []
    if set(packed) != set(live):
        diffs.append(f"keys: packed={sorted(packed)} live={sorted(live)}")
    for key in packed.keys() & live.keys():
        atol = audio_atol if key == audio_key else 0.0
        if not _same(packed[key], live[key], atol):
            diffs.append(key)
    return diffs


# --- worker-side reads ---------------------------------------------------------


def _read_timed(args: tuple[bytes, list[int]]) -> list[tuple[int, float, dict[str, Any]]]:
    """Read packed items in a spawned process.

    Parameters
    ----------
    args : tuple[bytes, list[int]]
        A pickled `PackedDataset` and the row indices to read.

    Returns
    -------
    list[tuple[int, float, dict[str, Any]]]
        `(index, seconds, item)` per row.
    """
    ds = pickle.loads(args[0])
    out = []
    for i in args[1]:
        t0 = time.perf_counter()
        item = ds[i]
        out.append((i, time.perf_counter() - t0, item))
    return out


def _stats(seconds: list[float]) -> dict[str, float]:
    if not seconds:
        return {}
    ordered = sorted(seconds)
    return {
        "n": len(seconds),
        "mean_ms": 1000 * statistics.fmean(seconds),
        "p50_ms": 1000 * ordered[len(ordered) // 2],
        "p95_ms": 1000 * ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_ms": 1000 * ordered[-1],
    }


def verify_pack(
    config: Any,  # noqa: ANN401
    out: str,
    n: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    """Compare `n` random packed samples with the live dataset.

    Parameters
    ----------
    config : DatasetConfig | ConcatConfig
        The config that was exported.
    out : str
        The pack location.
    n : int
        Number of samples to compare.
    workers : int
        If positive, also read the same samples through this many spawned
        processes and compare again.
    seed : int
        Random seed for the sample choice.

    Returns
    -------
    dict[str, Any]
        Counts, mismatches, and read timings for packed and live reads.
    """
    packed = PackedDataset(out)
    live, _ = dataset_from_config(config)
    audio_atol = 1.5 / 32768 if packed.pack_config["audio_format"] == "flac" else 0.0
    audio_key = packed.audio_key

    n = min(n, len(packed))
    rng = random.Random(seed)
    rows = sorted(rng.sample(range(len(packed)), n))
    live_index = [int(packed._data[i][SOURCE_INDEX_COL]) for i in rows]

    packed_t, live_t, mismatches = [], [], []
    for i, src in zip(rows, live_index, strict=True):
        t0 = time.perf_counter()
        p_item = packed[i]
        packed_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        l_item = live[src]
        live_t.append(time.perf_counter() - t0)
        diffs = compare_items(p_item, l_item, audio_key, audio_atol)
        if diffs:
            entry: dict[str, Any] = {"packed_row": i, "source_index": src, "keys": diffs}
            if audio_key in diffs:
                # Distinguish clipping (live audio outside [-1, 1]) from any other cause.
                entry["live_peak"] = float(np.abs(l_item[audio_key]).max())
                entry["stored_format"] = packed._data[i].get("_audio_format")
            mismatches.append(entry)

    result: dict[str, Any] = {
        "packed_rows": len(packed),
        "live_rows": len(live),
        "num_skipped": packed.pack_config.get("num_skipped", 0),
        "compared": n,
        "mismatches": mismatches,
        "packed_read": _stats(packed_t),
        "live_read": _stats(live_t),
    }

    if workers > 0 and n:
        blob = pickle.dumps(packed)
        chunks = [rows[k::workers] for k in range(workers)]
        t0 = time.perf_counter()
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_read_timed, [(blob, c) for c in chunks if c])
        wall = time.perf_counter() - t0
        worker_t, worker_mismatch = [], []
        for chunk in results:
            for i, sec, item in chunk:
                worker_t.append(sec)
                src = live_index[rows.index(i)]
                if compare_items(item, live[src], audio_key, audio_atol):
                    worker_mismatch.append(i)
        result["worker_read"] = {**_stats(worker_t), "workers": workers, "wall_s": wall}
        result["worker_mismatches"] = worker_mismatch

    return result


# --- CLI -----------------------------------------------------------------------


@click.command()
@click.option(
    "--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--key", default=None, help="Key selecting one config inside a collection YAML")
@click.option("--out", required=True, help="Destination directory: local, gs://, s3://")
@click.option("--format", "fmt", type=click.Choice(["pack", "hf"]), default="pack")
@click.option("--samples-per-shard", default=1000, show_default=True)
@click.option("--audio-format", type=click.Choice(["flac", "wav"]), default="flac")
@click.option("--num-workers", default=1, show_default=True)
@click.option("--on-error", type=click.Choice(["raise", "skip"]), default="raise")
@click.option(
    "--verify", default=0, show_default=True, help="Samples to compare with the live dataset"
)
@click.option(
    "--verify-workers", default=0, show_default=True, help="Spawned readers for the check"
)
@click.option("--seed", default=0, show_default=True)
@click.option("--import-module", multiple=True, help="Import first, to register user datasets")
@click.option("--summary", type=click.Path(path_type=Path), default=None, help="JSON summary path")
def main(
    config_path: Path,
    key: str | None,
    out: str,
    fmt: str,
    samples_per_shard: int,
    audio_format: str,
    num_workers: int,
    on_error: str,
    verify: int,
    verify_workers: int,
    seed: int,
    import_module: tuple[str, ...],
    summary: Path | None,
) -> None:
    """Export a configured dataset and optionally verify the pack.

    Raises
    ------
    click.UsageError
        If the YAML key selects a collection rather than one config.
    SystemExit
        If verification finds mismatches.
    """
    for module in import_module:
        importlib.import_module(module)

    config = config_from_yaml(config_path, key=key)
    if isinstance(config, list):
        raise click.UsageError("The YAML key selects a collection; pass a single dataset config")
    if isinstance(config, ChainedDatasetConfig):
        logger.warning("Chained config: it will be exported as a concatenation")

    exporter = pack if fmt == "pack" else to_hf
    logger.info("Exporting %s to %s as %s with %d workers", config_path, out, fmt, num_workers)
    t0 = time.perf_counter()
    exporter(
        config,
        out,
        samples_per_shard=samples_per_shard,
        audio_format=audio_format,
        num_workers=num_workers,
        on_error=on_error,
    )
    export_s = time.perf_counter() - t0

    report: dict[str, Any] = {
        "config": str(config_path),
        "out": out,
        "format": fmt,
        "num_workers": num_workers,
        "samples_per_shard": samples_per_shard,
        "audio_format": audio_format,
        "export_seconds": export_s,
    }
    if fmt == "pack":
        fs = filesystem_from_path(anypath(out))
        with fs.open(str(anypath(out) / "config.yaml"), "r") as f:
            pack_cfg = yaml.safe_load(f)
        total_bytes = sum(s["size"] for s in pack_cfg["shards"])
        report.update(
            {
                "num_rows": pack_cfg["num_rows"],
                "num_skipped": pack_cfg["num_skipped"],
                "num_shards": len(pack_cfg["shards"]),
                "media_bytes": total_bytes,
                "rows_per_second": pack_cfg["num_rows"] / export_s if export_s else None,
                "media_mb_per_second": total_bytes / 1e6 / export_s if export_s else None,
            }
        )
        logger.info(
            "Packed %d rows into %d shards (%.2f GB) in %.1fs: %.1f rows/s",
            pack_cfg["num_rows"],
            len(pack_cfg["shards"]),
            total_bytes / 1e9,
            export_s,
            report["rows_per_second"] or 0.0,
        )
        if verify:
            logger.info("Verifying %d samples against the live dataset", verify)
            report["verify"] = verify_pack(config, out, verify, verify_workers, seed)
            v = report["verify"]
            logger.info(
                "Verified %d samples: %d mismatches; "
                "packed read p50 %.1f ms, live read p50 %.1f ms",
                v["compared"],
                len(v["mismatches"]),
                v["packed_read"].get("p50_ms", float("nan")),
                v["live_read"].get("p50_ms", float("nan")),
            )
            if "worker_read" in v:
                logger.info(
                    "Through %d spawned readers: p50 %.1f ms, %d mismatches",
                    verify_workers,
                    v["worker_read"].get("p50_ms", float("nan")),
                    len(v["worker_mismatches"]),
                )
    elif verify:
        logger.warning("--verify only applies to the pack format; skipping")

    if summary is not None:
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Summary written to %s", summary)

    if (
        fmt == "pack"
        and verify
        and (report["verify"]["mismatches"] or report["verify"].get("worker_mismatches"))
    ):
        raise SystemExit("Verification found mismatches; see the summary")


if __name__ == "__main__":
    main()
