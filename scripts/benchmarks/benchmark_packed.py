"""Benchmark packed datasets against the live datasets they were frozen from.

For every pack given, the live dataset is rebuilt from the pack's frozen
`source` config, so both sides serve exactly the same rows. Each side is then
measured the same way:

- `dataloader`: a PyTorch `DataLoader` with spawned workers, shuffled, with an
  identity collate so the number is IO plus decode and nothing else. Records
  samples per second over the measured batches, time to the first batch
  (worker start-up included), batch latency percentiles, and peak resident
  memory of the parent and of the workers.
- `sequential`: plain `ds[i]` over random indices in the calling process.

One CSV row per (pack, side, mode, workers, prefetch, batch size). Rows are
appended to `--out` and, with `--upload`, the CSV is copied to a bucket.

Examples
--------
uv run --group benchmark python scripts/benchmarks/benchmark_packed.py \\
    --pack gs://bucket/exports/beans/validation-native \\
    --pack gs://bucket/exports/fasd13/all-32k \\
    --workers 0,4,16,48 --prefetch 2,8 --batch-size 32 --measure-seconds 60
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import random
import socket
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import pandas as pd
import psutil

from alp_data.dataset import dataset_from_config, load_config
from alp_data.export import PackedDataset
from alp_data.io import anypath, filesystem_from_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("benchmark_packed")


def collate_identity(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the batch as is, so collation costs nothing.

    Parameters
    ----------
    batch : list[dict[str, Any]]
        Samples from the dataset.

    Returns
    -------
    list[dict[str, Any]]
        The same list.
    """
    return batch


class MemorySampler:
    """Sample resident memory of this process and its children in a thread.

    Parameters
    ----------
    interval : float
        Seconds between samples.
    """

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.parent_peak = 0
        self.children_sum_peak = 0
        self.children_max_peak = 0
        self.children_count_peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                self.parent_peak = max(self.parent_peak, proc.memory_info().rss)
                rss = []
                for child in proc.children(recursive=True):
                    try:
                        rss.append(child.memory_info().rss)
                    except psutil.Error:
                        continue
                if rss:
                    self.children_sum_peak = max(self.children_sum_peak, sum(rss))
                    self.children_max_peak = max(self.children_max_peak, max(rss))
                    self.children_count_peak = max(self.children_count_peak, len(rss))
            except psutil.Error:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> MemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def as_dict(self) -> dict[str, float]:
        """Peak values in megabytes.

        Returns
        -------
        dict[str, float]
            Parent peak, worker sum peak, largest single worker peak, and
            the most workers seen at once.
        """
        mb = 1 / 1024**2
        return {
            "parent_peak_mb": self.parent_peak * mb,
            "workers_sum_peak_mb": self.children_sum_peak * mb,
            "worker_max_peak_mb": self.children_max_peak * mb,
            "workers_seen": self.children_count_peak,
        }


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def run_dataloader(
    ds: Any,  # noqa: ANN401
    *,
    batch_size: int,
    workers: int,
    prefetch: int,
    warmup_batches: int,
    measure_seconds: float,
    max_batches: int,
    seed: int,
) -> dict[str, Any]:
    """Iterate a dataset through a `DataLoader` and time its steady state.

    With workers, the loader has up to `workers * prefetch` batches in flight
    by the time the first one arrives, because every worker starts prefetching
    at once. Timing from the first batch would mostly measure that backlog
    draining. So the warm-up is at least `workers * prefetch` batches, and the
    timed window is a fixed `measure_seconds` of wall clock after that.

    Parameters
    ----------
    ds : Dataset
        Map-style dataset.
    batch_size : int
        Samples per batch.
    workers : int
        `num_workers`; 0 loads in the calling process.
    prefetch : int
        `prefetch_factor`, only meaningful with workers.
    warmup_batches : int
        Minimum batches discarded before timing starts; raised to
        `workers * prefetch` when that is larger.
    measure_seconds : float
        Length of the timed window. The window also ends when the epoch does.
    max_batches : int
        Upper bound on timed batches; `0` means no bound.
    seed : int
        Shuffle seed.

    Returns
    -------
    dict[str, Any]
        Throughput, latency percentiles, time to first batch, and memory peaks.
    """
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, Any] = {}
    if workers > 0:
        kwargs.update(
            prefetch_factor=prefetch,
            multiprocessing_context="spawn",
            persistent_workers=False,
        )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_identity,
        generator=generator,
        **kwargs,
    )

    warmup = max(warmup_batches, workers * prefetch)
    latencies: list[float] = []
    samples = 0
    first_batch_s = float("nan")
    t_measure = float("nan")
    with MemorySampler() as mem:
        t_start = time.perf_counter()
        t_prev = t_start
        for i, batch in enumerate(loader):
            now = time.perf_counter()
            if i == 0:
                first_batch_s = now - t_start
            if i == warmup:
                t_measure = now
            if i >= warmup:
                latencies.append(now - t_prev)
                samples += len(batch)
                if now - t_measure >= measure_seconds or (
                    max_batches and len(latencies) >= max_batches
                ):
                    break
            t_prev = now
        t_end = time.perf_counter()
    measured_s = t_end - t_measure if latencies else float("nan")
    return {
        "warmup_batches": warmup,
        "batches": len(latencies),
        "samples": samples,
        "measured_s": measured_s,
        "samples_per_s": samples / measured_s if latencies else float("nan"),
        "first_batch_s": first_batch_s,
        "batch_p50_s": _pct(latencies, 0.5),
        "batch_p95_s": _pct(latencies, 0.95),
        "batch_mean_s": statistics.fmean(latencies) if latencies else float("nan"),
        **mem.as_dict(),
    }


def run_sequential(ds: Any, *, n: int, seed: int) -> dict[str, Any]:  # noqa: ANN401
    """Read `n` random samples one after another in this process.

    Parameters
    ----------
    ds : Dataset
        Map-style dataset.
    n : int
        Samples to read.
    seed : int
        Seed for the index choice.

    Returns
    -------
    dict[str, Any]
        Throughput and per-sample latency percentiles.
    """
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    latencies = []
    with MemorySampler() as mem:
        t0 = time.perf_counter()
        for i in idx:
            t = time.perf_counter()
            ds[i]
            latencies.append(time.perf_counter() - t)
        total = time.perf_counter() - t0
    return {
        "samples": len(idx),
        "measured_s": total,
        "samples_per_s": len(idx) / total if total else float("nan"),
        "first_batch_s": float("nan"),
        "batch_p50_s": _pct(latencies, 0.5),
        "batch_p95_s": _pct(latencies, 0.95),
        "batch_mean_s": statistics.fmean(latencies) if latencies else float("nan"),
        **mem.as_dict(),
    }


def load_pair(pack_path: str) -> tuple[PackedDataset, Any, dict[str, Any]]:
    """Open a pack and rebuild the live dataset it was frozen from.

    Parameters
    ----------
    pack_path : str
        Pack directory.

    Returns
    -------
    tuple[PackedDataset, Dataset, dict[str, Any]]
        The packed dataset, the live dataset, and the pack's `config.yaml`.
    """
    packed = PackedDataset(pack_path)
    live, _ = dataset_from_config(load_config(packed.pack_config["source"]))
    return packed, live, packed.pack_config


def _append_rows(out: Path, rows: list[dict[str, Any]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, mode="a", header=not out.exists(), index=False)


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


@click.command()
@click.option("--pack", "packs", multiple=True, required=True, help="Pack directory; repeatable")
@click.option("--workers", default="0,4,16", show_default=True, help="DataLoader worker counts")
@click.option("--prefetch", default="2", show_default=True, help="prefetch_factor values")
@click.option("--batch-size", default=32, show_default=True)
@click.option("--measure-seconds", default=60.0, show_default=True, help="Timed window per run")
@click.option("--max-batches", default=0, show_default=True, help="Cap on timed batches; 0 = none")
@click.option("--warmup-batches", default=4, show_default=True, help="Minimum warm-up batches")
@click.option("--sequential-samples", default=200, show_default=True)
@click.option("--modes", default="dataloader,sequential", show_default=True)
@click.option("--sides", default="packed,live", show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--import-module", multiple=True, help="Import first, to register user datasets")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="CSV to append to")
@click.option("--upload", default=None, help="Bucket directory to copy the CSV to afterwards")
@click.option("--tag", default="", help="Free text stored with every row")
def main(
    packs: tuple[str, ...],
    workers: str,
    prefetch: str,
    batch_size: int,
    measure_seconds: float,
    max_batches: int,
    warmup_batches: int,
    sequential_samples: int,
    modes: str,
    sides: str,
    seed: int,
    import_module: tuple[str, ...],
    out: Path,
    upload: str | None,
    tag: str,
) -> None:
    """Benchmark packed datasets against their live sources."""
    for module in import_module:
        importlib.import_module(module)

    worker_counts = _parse_ints(workers)
    prefetches = _parse_ints(prefetch)
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    side_list = [s.strip() for s in sides.split(",") if s.strip()]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname()

    for pack_path in packs:
        packed, live, cfg = load_pair(pack_path)
        datasets = {"packed": packed, "live": live}
        base = {
            "run_id": run_id,
            "host": host,
            "tag": tag,
            "pack": pack_path,
            "dataset": cfg.get("name"),
            "split": cfg.get("split"),
            "sample_rate": cfg.get("sample_rate"),
            "audio_format": cfg.get("audio_format"),
            "num_rows": len(packed),
            "media_gb": sum(s["size"] for s in cfg["shards"]) / 1e9,
            "cpu_count": os.cpu_count(),
        }
        logger.info("%s: %d rows, %.2f GB", cfg.get("name"), len(packed), base["media_gb"])

        rows: list[dict[str, Any]] = []
        if "sequential" in mode_list:
            for side in side_list:
                res = run_sequential(datasets[side], n=sequential_samples, seed=seed)
                logger.info(
                    "  sequential %-6s %.1f samples/s, p50 %.0f ms, p95 %.0f ms",
                    side,
                    res["samples_per_s"],
                    1000 * res["batch_p50_s"],
                    1000 * res["batch_p95_s"],
                )
                rows.append(
                    {
                        **base,
                        "mode": "sequential",
                        "side": side,
                        "workers": 0,
                        "prefetch": 0,
                        "batch_size": 1,
                        **res,
                    }
                )
        if "dataloader" in mode_list:
            for w in worker_counts:
                for pf in prefetches if w > 0 else [0]:
                    for side in side_list:
                        res = run_dataloader(
                            datasets[side],
                            batch_size=batch_size,
                            workers=w,
                            prefetch=pf,
                            warmup_batches=warmup_batches,
                            measure_seconds=measure_seconds,
                            max_batches=max_batches,
                            seed=seed,
                        )
                        logger.info(
                            "  dataloader %-6s workers=%-2d prefetch=%-2d %.1f samples/s, "
                            "first batch %.1fs, batch p50 %.2fs, worker peak %.0f MB",
                            side,
                            w,
                            pf,
                            res["samples_per_s"],
                            res["first_batch_s"],
                            res["batch_p50_s"],
                            res["worker_max_peak_mb"],
                        )
                        rows.append(
                            {
                                **base,
                                "mode": "dataloader",
                                "side": side,
                                "workers": w,
                                "prefetch": pf,
                                "batch_size": batch_size,
                                **res,
                            }
                        )
        _append_rows(out, rows)
        packed._store.close()

    logger.info("Results appended to %s", out)
    if upload:
        target = anypath(upload) / out.name
        fs = filesystem_from_path(target)
        fs.put(str(out), str(target))
        logger.info("Uploaded to %s", target)

    summary = pd.read_csv(out)
    summary = summary[summary["run_id"] == run_id]
    cols = [
        "dataset",
        "mode",
        "side",
        "workers",
        "prefetch",
        "samples_per_s",
        "batch_p50_s",
        "first_batch_s",
        "worker_max_peak_mb",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(json.dumps({"run_id": run_id, "rows": int(summary.shape[0])}))


if __name__ == "__main__":
    main()
