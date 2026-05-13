"""Check that every split path of every registered dataset exists.

Iterates the global dataset registry, resolves each `split_paths` entry, and
verifies it exists on its backing filesystem. Exits with status 1 if any
split is missing.

Run with::

    uv run python scripts/check_dataset_splits_exist.py
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import esp_data.datasets  # noqa: F401  -- populates the dataset registry
from esp_data.dataset import _dataset_registry
from esp_data.io import exists


def check_split(dataset_name: str, split: str, path: str) -> tuple[str, str, str, bool, str | None]:
    """Check a single split path for existence.

    Parameters
    ----------
    dataset_name : str
        Registered dataset name.
    split : str
        Split name (e.g. `train`, `validation`).
    path : str
        Path to check.

    Returns
    -------
    tuple
        `(dataset_name, split, path, ok, error)` where `ok` is True iff the
        path exists and no exception was raised. `error` holds the exception
        string when the check failed unexpectedly.
    """
    try:
        ok = exists(path)
    except Exception as e:  # noqa: BLE001
        return dataset_name, split, path, False, repr(e)
    return dataset_name, split, path, ok, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Restrict to one or more registered dataset names. Repeatable.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel existence checks (default: 16).",
    )
    args = parser.parse_args()

    selected = set(args.dataset) if args.dataset else None

    jobs: list[tuple[str, str, str]] = []
    for name, cls in sorted(_dataset_registry.items()):
        if selected is not None and name not in selected:
            continue

        # Some datasets (e.g. AudioSet) populate `info.split_paths` at instance
        # init time from a class-level `VERSIONS` registry. Expand those here so
        # every version's splits get checked.
        versions = getattr(cls, "VERSIONS", None)
        if versions:
            per_version: list[tuple[str, str, str]] = []
            for version, cfg in versions.items():
                vsplits = (cfg or {}).get("split_paths") or {}
                for split, path in vsplits.items():
                    per_version.append((f"{name}@{version}", split, str(path)))
            if per_version:
                jobs.extend(per_version)
                continue

        split_paths = getattr(cls.info, "split_paths", None) or {}
        if not split_paths:
            print(f"[WARN] {name}: no split_paths defined")
            continue
        for split, path in split_paths.items():
            jobs.append((name, split, str(path)))

    if not jobs:
        print("No splits to check.")
        return 0

    print(f"Checking {len(jobs)} split paths across {len({j[0] for j in jobs})} datasets...\n")

    missing: list[tuple[str, str, str, str | None]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(check_split, *job) for job in jobs]
        for fut in as_completed(futures):
            name, split, path, ok, err = fut.result()
            status = "OK   " if ok else "MISS "
            suffix = f"  ({err})" if err else ""
            print(f"  {status} {name}/{split} -> {path}{suffix}")
            if not ok:
                missing.append((name, split, path, err))

    print()
    if missing:
        print(f"FAIL: {len(missing)} missing split(s):")
        for name, split, path, err in missing:
            err_s = f"  [{err}]" if err else ""
            print(f"  - {name}/{split}: {path}{err_s}")
        return 1

    print(f"OK: all {len(jobs)} splits exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
