"""Iterate every row of the iNaturalist splits and record any that fail to load.

Temporary sanity check for issue #345: loads each row through
`INaturalist._process` (the method `__getitem__` / `__iter__` call, i.e. the
real `read_audio` path) for the requested versions and split, and checks that
the returned audio is non-empty, finite, float32 and at the requested rate.

Only the main process reads the split CSV, and only its path columns (the
dataset is opened with ``streaming=True`` so the column selection is pushed
into the CSV scan). Rows are cut into contiguous
chunks; each chunk's path columns are shipped to a pool of spawned worker
processes, which run `_process` on a thread pool (loading is dominated by GCS
I/O). Workers hold no CSV, so memory does not grow with ``--n-procs``.

Native decoder crashes (e.g. libmpg123 inside libsndfile on a corrupt mp3) kill
a worker process and cannot be caught in Python. When that happens the pool is
restarted, chunks that were running are split in half and retried, and this
repeats until the crash is narrowed to single rows. Those rows are then run
one at a time in a single-process pool: if a row still kills the process it is
recorded as ``WorkerCrash``; if it passes on its own, the crash was a
concurrency effect and the row counts as OK.

Outputs, per version, in ``--out-dir``:

- ``failures_<version>_<split>_sr<rate>[_shard<i>of<n>].csv``: one line per
  failing row (index, ``originals_path``, the path column actually read, error).
- ``summary_<...>.json``: row counts, failure counts by error type, sample
  rate histogram, worker crash/restart counts and wall time.
- ``done_<...>.txt``: indices of finished chunks, so a rerun with the same
  arguments resumes where it stopped.

Usage (on a GCP VM, same region as the bucket)
-----------------------------------------------
    uv run python scripts/iterate_inaturalist.py --out-dir ~/inat_iter \\
        --versions 0.1.0 0.2.0 --split all --n-procs 8 --n-threads 32

    # Native sample rates (default) exercise `originals_path`, which is what
    # #345 broke; --sample-rate 16000 / 32000 exercise the resampled columns.
    # Shard across machines with --num-shards N --shard-index i.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alp_data.datasets.inaturalist import INaturalist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("iterate_inaturalist")

PATH_COLUMNS = ["originals_path", "16khz_path", "32khz_path"]
MAX_IDLE_CRASHES = 3
# Test hook: comma-separated row indices whose loading hard-kills the worker.
_CRASH_AT = {int(i) for i in os.environ.get("ITER_INAT_CRASH_AT", "").split(",") if i}


class _RowsOnlyINaturalist(INaturalist):
    """`INaturalist` that skips reading the split CSV.

    Workers receive rows from the main process and only need `_process`, so
    loading the multi-GB CSV in every worker is wasted memory.
    """

    def _load(self) -> None:
        """Do nothing; rows are supplied by the caller."""


# Worker-process globals, set by `_init_worker`.
_DS: INaturalist | None = None
_N_THREADS = 1
_RUNNING_DIR: Path | None = None


def _init_worker(
    version: str, split: str, sample_rate: int | None, n_threads: int, running_dir: str
) -> None:
    """Set up a worker process.

    Parameters
    ----------
    version : str
        Dataset version.
    split : str
        Split name.
    sample_rate : int or None
        Requested sample rate, or None for native rates.
    n_threads : int
        Threads used per task.
    running_dir : str
        Directory for "task is running" marker files.
    """
    global _DS, _N_THREADS, _RUNNING_DIR
    _DS = _RowsOnlyINaturalist(split=split, version=version, sample_rate=sample_rate)
    _N_THREADS = n_threads
    _RUNNING_DIR = Path(running_dir)


def _validate(audio: np.ndarray, sr: int, requested: int | None) -> None:
    """Check that loaded audio is usable.

    Parameters
    ----------
    audio : np.ndarray
        Audio returned by the dataset.
    sr : int
        Sample rate returned by the dataset.
    requested : int or None
        Sample rate the dataset was asked for, or None for native rates.

    Raises
    ------
    ValueError
        If the audio is empty, not float32, contains non-finite samples, or
        is not at the requested sample rate.
    """
    if audio.size == 0:
        raise ValueError("empty audio")
    if audio.dtype != np.float32:
        raise ValueError(f"dtype {audio.dtype}")
    if not np.isfinite(audio).all():
        raise ValueError("non-finite samples")
    if requested is not None and sr != requested:
        raise ValueError(f"sample_rate {sr} != requested {requested}")


def _read_path_column(sample_rate: int | None) -> str:
    """Return the path column `_process` reads for a sample rate.

    Parameters
    ----------
    sample_rate : int or None
        Requested sample rate.

    Returns
    -------
    str
        ``16khz_path``/``32khz_path`` for the pre-resampled rates, otherwise
        ``originals_path``.
    """
    return INaturalist._sample_rate_paths.get(sample_rate, INaturalist._originals_path_column)


def _check_row(item: tuple[int, dict[str, Any]]) -> tuple[int, int | None, float, str | None]:
    """Load one row through `_process` and validate the audio.

    Parameters
    ----------
    item : tuple of (int, dict)
        ``(row index, row dict)``.

    Returns
    -------
    tuple of (int, int or None, float, str or None)
        ``(idx, sample_rate, duration_s, error)``; ``error`` is None on success.
    """
    idx, row = item
    if idx in _CRASH_AT:
        os._exit(1)
    try:
        out = _DS._process(dict(row))
        audio, sr = out["audio"], out["sample_rate"]
        _validate(audio, sr, _DS.sample_rate)
        return idx, int(sr), audio.shape[-1] / sr, None
    except Exception as exc:  # noqa: BLE001 - every failure is recorded
        return idx, None, 0.0, f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class Task:
    """A contiguous range of rows belonging to one chunk."""

    chunk_id: int
    start: int
    stop: int

    @property
    def key(self) -> str:
        """Return a file-name-safe identifier."""
        return f"{self.start}-{self.stop}"

    def halves(self) -> list[Task]:
        """Split the task in half.

        Returns
        -------
        list of Task
            Two tasks covering the same rows, or ``[self]`` for a single row.
        """
        if self.stop - self.start <= 1:
            return [self]
        mid = (self.start + self.stop) // 2
        return [Task(self.chunk_id, self.start, mid), Task(self.chunk_id, mid, self.stop)]


def _run_task(task: Task, rows: list[dict[str, Any]]) -> dict:
    """Run one task in a worker, leaving a marker file while it runs.

    Parameters
    ----------
    task : Task
        Row range.
    rows : list of dict
        Path columns of rows ``task.start`` .. ``task.stop - 1``.

    Returns
    -------
    dict
        ``{"ok": [(idx, sr, duration_s)], "failed": [(idx, error)]}``.
    """
    marker = _RUNNING_DIR / task.key
    marker.touch()
    ok, failed = [], []
    items = list(zip(range(task.start, task.stop), rows, strict=True))
    with ThreadPoolExecutor(max_workers=min(_N_THREADS, len(items))) as pool:
        for idx, sr, dur, err in pool.map(_check_row, items):
            if err is None:
                ok.append((idx, sr, dur))
            else:
                failed.append((idx, err))
    marker.unlink(missing_ok=True)
    return {"ok": ok, "failed": failed}


@dataclass
class Progress:
    """Mutable bookkeeping for one version's run."""

    rows: list[dict[str, Any]]
    read_column: str
    writer: Any
    fail_f: Any
    done_f: Any
    outstanding: Counter = field(default_factory=Counter)
    sr_counts: Counter = field(default_factory=Counter)
    n_done: int = 0
    n_fail: int = 0
    seconds: float = 0.0
    restarts: int = 0
    crashes: int = 0

    def record(self, task: Task, ok: list, failed: list) -> None:
        """Record a finished task and mark its chunk done when complete.

        Parameters
        ----------
        task : Task
            The finished task.
        ok : list
            ``(idx, sr, duration_s)`` per successful row.
        failed : list
            ``(idx, error)`` per failed row.
        """
        for _, sr, dur in ok:
            self.sr_counts[sr] += 1
            self.seconds += dur
        for idx, err in failed:
            row = self.rows[idx]
            self.writer.writerow([idx, row["originals_path"], row[self.read_column], err])
        self.fail_f.flush()
        self.n_done += task.stop - task.start
        self.n_fail += len(failed)
        self.outstanding[task.chunk_id] -= 1
        if self.outstanding[task.chunk_id] == 0:
            self.done_f.write(f"{task.chunk_id}\n")
            self.done_f.flush()


def _run_round(
    tasks: list[Task], prog: Progress, n_procs: int, init: tuple, total: int, tag: str
) -> tuple[list[Task], list[Task]]:
    """Run tasks in one pool until they all finish or the pool breaks.

    Parameters
    ----------
    tasks : list of Task
        Tasks to run.
    prog : Progress
        Bookkeeping, updated in place.
    n_procs : int
        Worker processes.
    init : tuple
        `_init_worker` arguments; the last one is the marker directory.
    total : int
        Rows to check in this run (for the progress log).
    tag : str
        Log prefix.

    Returns
    -------
    tuple of (list of Task, list of Task)
        ``(were_running, not_started)``: tasks lost to a pool crash, split by
        whether they had started (their marker file existed). Both are empty
        if the pool did not break.
    """
    running_dir = Path(init[-1])
    ctx = mp.get_context("spawn")  # gcsfs is not fork-safe
    lost: list[Task] = []
    t0 = time.time()
    n0 = prog.n_done
    with ProcessPoolExecutor(n_procs, mp_context=ctx, initializer=_init_worker,
                             initargs=init) as pool:  # fmt: skip
        futs = {pool.submit(_run_task, t, prog.rows[t.start : t.stop]): t for t in tasks}
        for fut in as_completed(futs):
            task = futs[fut]
            try:
                res = fut.result()
            except BrokenProcessPool:
                lost.append(task)
                continue
            prog.record(task, res["ok"], res["failed"])
            rate = (prog.n_done - n0) / max(time.time() - t0, 1e-9)
            logger.info("[%s] %d/%d rows (%.0f rows/s, ETA %.1f min), %d failures, %d restarts",
                        tag, prog.n_done, total, rate, (total - prog.n_done) / max(rate, 1e-9) / 60,
                        prog.n_fail, prog.restarts)  # fmt: skip
    running = {p.name for p in running_dir.iterdir()}
    for p in running_dir.iterdir():
        p.unlink()
    were_running = [t for t in lost if t.key in running]
    not_started = [t for t in lost if t.key not in running]
    return were_running, not_started


def _run_all(tasks: list[Task], prog: Progress, args: argparse.Namespace, init: tuple,
             tag: str) -> None:  # fmt: skip
    """Run tasks, bisecting crashing tasks down to single rows.

    Parameters
    ----------
    tasks : list of Task
        Tasks to run.
    prog : Progress
        Bookkeeping, updated in place.
    args : argparse.Namespace
        Parsed CLI arguments.
    init : tuple
        `_init_worker` arguments.
    tag : str
        Log prefix.

    Raises
    ------
    RuntimeError
        If the pool keeps crashing while no task is running (e.g. workers die
        on start-up), which bisection cannot fix.
    """
    total = sum(t.stop - t.start for t in tasks)
    pending, suspects = tasks, []
    idle_crashes = 0
    while pending:
        were_running, not_started = _run_round(pending, prog, args.n_procs, init, total, tag)
        if not were_running and not not_started:
            break
        prog.restarts += 1
        idle_crashes = 0 if were_running else idle_crashes + 1
        if idle_crashes >= MAX_IDLE_CRASHES:
            raise RuntimeError("worker pool keeps crashing with no task running")
        logger.warning("[%s] worker pool crashed (%d tasks were running); restarting", tag,
                       len(were_running))  # fmt: skip
        pending = list(not_started)
        for t in were_running:
            if t.stop - t.start == 1:
                suspects.append(t)
                continue
            pending.extend(t.halves())
            prog.outstanding[t.chunk_id] += 1  # one task became two

    # Rerun single-row suspects alone, so a crash can only be that row.
    single = (*init[:3], 1, init[4])
    for t in suspects:
        were_running, not_started = _run_round([t], prog, 1, single, total, tag)
        if were_running or not_started:
            prog.crashes += 1
            prog.record(t, [], [(t.start, "WorkerCrash: process died while loading this row")])


def run_version(args: argparse.Namespace, version: str) -> dict:
    """Iterate one version's split and write its failures and summary.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    version : str
        Dataset version.

    Returns
    -------
    dict
        The summary written to ``summary_<...>.json``.
    """
    tag = f"{version}_{args.split}_sr{args.sample_rate or 'native'}"
    if args.num_shards > 1:
        tag += f"_shard{args.shard_index}of{args.num_shards}"
    out = args.out_dir
    fail_path, done_path = out / f"failures_{tag}.csv", out / f"done_{tag}.txt"
    running_dir = out / f"running_{tag}"
    shutil.rmtree(running_dir, ignore_errors=True)
    running_dir.mkdir()

    # Streaming gives a LazyFrame, so selecting the path columns before
    # collecting means polars only parses those (~100 MB instead of ~3 GB).
    ds = INaturalist(split=args.split, version=version, streaming=True)
    lf = ds._data.unwrap.select(PATH_COLUMNS)
    if args.limit:
        lf = lf.head(args.limit)
    rows = lf.collect().to_dicts()
    del ds, lf

    done = set(map(int, done_path.read_text().split())) if done_path.exists() else set()
    chunks = [
        Task(i, start, min(start + args.chunk_size, len(rows)))
        for i, start in enumerate(range(0, len(rows), args.chunk_size))
        if i % args.num_shards == args.shard_index and i not in done
    ]
    logger.info("[%s] %d rows, %d chunks to do (%d already done)", tag, len(rows), len(chunks),
                len(done))  # fmt: skip

    new_file = not fail_path.exists()
    t0 = time.time()
    init = (version, args.split, args.sample_rate, args.n_threads, str(running_dir))
    with open(fail_path, "a", newline="") as fail_f, open(done_path, "a") as done_f:
        writer = csv.writer(fail_f)
        if new_file:
            writer.writerow(["idx", "originals_path", "read_path", "error"])
        prog = Progress(rows, _read_path_column(args.sample_rate), writer, fail_f, done_f)
        prog.outstanding.update({c.chunk_id: 1 for c in chunks})
        _run_all(chunks, prog, args, init, tag)
    shutil.rmtree(running_dir, ignore_errors=True)

    # Summarise from the failures file so resumed runs report the full picture.
    with open(fail_path) as f:
        errors = [r["error"] for r in csv.DictReader(f)]
    summary = {
        "version": version,
        "split": args.split,
        "sample_rate": args.sample_rate,
        "shard": [args.shard_index, args.num_shards],
        "rows_in_split": len(rows),
        "rows_checked_this_run": prog.n_done,
        "failures_total": len(errors),
        "failures_by_type": Counter(e.split(":", 1)[0] for e in errors).most_common(),
        "worker_pool_restarts_this_run": prog.restarts,
        "rows_that_crash_a_worker_this_run": prog.crashes,
        "sample_rate_counts_this_run": dict(sorted(prog.sr_counts.items())),
        "audio_hours_this_run": round(prog.seconds / 3600, 1),
        "wall_minutes_this_run": round((time.time() - t0) / 60, 1),
    }
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    logger.info("[%s] summary: %s", tag, json.dumps(summary))
    return summary


def main() -> None:
    """Command-line entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--versions", nargs="+", default=["0.1.0", "0.2.0"])
    p.add_argument("--split", default="all")
    p.add_argument("--sample-rate", type=int, default=None, help="Default: native rates.")
    p.add_argument("--n-procs", type=int, default=8)
    p.add_argument("--n-threads", type=int, default=32, help="Threads per process.")
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--limit", type=int, default=None, help="Only the first N rows (smoke test).")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()
    args.out_dir = args.out_dir.expanduser()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [run_version(args, v) for v in args.versions]
    for s in summaries:
        logger.info("%s %s: %d/%d failed (%d worker crashes)", s["version"], s["split"],
                    s["failures_total"], s["rows_in_split"],
                    s["rows_that_crash_a_worker_this_run"])  # fmt: skip


if __name__ == "__main__":
    main()
