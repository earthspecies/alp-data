"""Slurm-side load test for the ObservationOrg dataset.

Iterates the full `all` split at sample_rate=32000, loading each row's
audio via the dataset's `__getitem__`. Reports:
- success / failure counts (with error message buckets)
- sample-rate audit (must equal 32000 for every loaded row)
- schema completeness — which downstream prompt templates would work,
  by counting non-null/non-empty rates for key columns

Writes a JSON summary + a per-error CSV to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np


def _load_one(args: tuple[int, object]) -> tuple[int, bool, str, int, int, dict]:
    """Worker: load one row and return diagnostics.

    The dataset is shared across threads (read-only pandas backend
    access; audio reads are I/O-bound and release the GIL).

    Returns
    -------
    tuple
        ``(idx, success, error_class, sample_rate, n_samples, schema_flags)``
    """
    idx, ds = args
    try:
        row = ds._data[idx]
    except Exception as e:
        return idx, False, f"row_index:{type(e).__name__}", -1, 0, {}

    def _ok(v):
        if v is None: return False
        s = str(v).strip()
        return s and s.lower() != "nan"

    schema_flags = {
        col: _ok(row.get(col)) for col in [
            "canonical_name", "species_common", "genus", "family", "order",
            "class", "phylum", "lifeStage", "sex", "license", "media_license",
            "16khz_path", "32khz_path", "relative_path",
        ]
    }

    try:
        item = ds[idx]
    except Exception as e:
        # Trim message to a bucket-friendly key.
        msg = f"load:{type(e).__name__}"
        # Special-case the most common failure modes.
        em = str(e).lower()
        if "no such file" in em or "no urls matched" in em or "no objects" in em:
            msg = "load:missing_file"
        elif "format" in em:
            msg = "load:unsupported_format"
        elif "memory" in em or "killed" in em:
            msg = "load:oom"
        return idx, False, msg, -1, 0, schema_flags

    audio = item.get("audio")
    sr = item.get("sample_rate", -1)
    if not isinstance(audio, np.ndarray):
        return idx, False, "audio:not_ndarray", sr, 0, schema_flags
    if audio.dtype != np.float32:
        return idx, False, f"audio:wrong_dtype:{audio.dtype}", sr, audio.size, schema_flags
    if audio.size == 0:
        return idx, False, "audio:empty", sr, 0, schema_flags
    if np.any(np.isnan(audio)):
        return idx, False, "audio:nan", sr, audio.size, schema_flags
    if sr != 32000:
        return idx, False, f"audio:sr:{sr}", sr, audio.size, schema_flags
    return idx, True, "", sr, audio.size, schema_flags


def main() -> None:
    """Entry point — see module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=-1,
                        help="Limit for smoke-testing; -1 = all rows.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset once; share across worker threads (read-only).
    from esp_data.datasets import ObservationOrg
    ds = ObservationOrg(split="all", sample_rate=32000, backend="pandas")
    n_rows = len(ds)
    print(f"Dataset has {n_rows:,} rows; columns = {list(ds.columns)[:20]} ...",
          flush=True)
    print(f"available_sample_rates = {ds.available_sample_rates}", flush=True)

    if args.max_rows > 0:
        n_rows = min(n_rows, args.max_rows)
    indices: list[tuple[int, object]] = [(i, ds) for i in range(n_rows)]

    t0 = time.time()
    errors_by_type: Counter = Counter()
    schema_counter: Counter = Counter()
    n_total = 0
    n_ok = 0
    sr_counter: Counter = Counter()
    sample_durations_s: list[float] = []
    error_rows: list[tuple[int, str]] = []

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for idx, ok, err, sr, n_samples, schema in ex.map(_load_one, indices):
            n_total += 1
            sr_counter[sr] += 1
            for col, flag in schema.items():
                schema_counter[(col, flag)] += 1
            if ok:
                n_ok += 1
                if len(sample_durations_s) < 200 and sr > 0:
                    sample_durations_s.append(n_samples / sr)
            else:
                errors_by_type[err] += 1
                if len(error_rows) < 200:
                    error_rows.append((idx, err))
            if n_total % 200 == 0:
                elapsed = time.time() - t0
                rate = n_total / max(elapsed, 1e-3)
                eta = (len(indices) - n_total) / max(rate, 1e-3) / 60.0
                print(
                    f"  {n_total:,}/{len(indices):,}  ok={n_ok:,}  "
                    f"err={n_total - n_ok}  rate={rate:.1f}/s  eta={eta:.1f} min",
                    flush=True,
                )

    elapsed = time.time() - t0
    summary = {
        "n_rows_attempted": n_total,
        "n_rows_ok": n_ok,
        "success_rate": n_ok / max(n_total, 1),
        "elapsed_s": round(elapsed, 1),
        "errors_by_type": dict(errors_by_type),
        "sample_rate_counts": {str(k): v for k, v in sr_counter.items()},
        "duration_stats_s_sample": {
            "n": len(sample_durations_s),
            "mean": float(np.mean(sample_durations_s)) if sample_durations_s else None,
            "median": float(np.median(sample_durations_s)) if sample_durations_s else None,
            "min": float(np.min(sample_durations_s)) if sample_durations_s else None,
            "max": float(np.max(sample_durations_s)) if sample_durations_s else None,
        },
        "schema_completeness": {
            col: schema_counter[(col, True)] / max(n_total, 1)
            for col in {c for c, _ in schema_counter}
        },
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    (args.out_dir / "observation_org_load_test.json").write_text(
        json.dumps(summary, indent=2)
    )
    if error_rows:
        import csv
        with (args.out_dir / "observation_org_load_test_errors.csv").open("w") as f:
            w = csv.writer(f)
            w.writerow(["row_index", "error_class"])
            w.writerows(error_rows)


if __name__ == "__main__":
    main()
