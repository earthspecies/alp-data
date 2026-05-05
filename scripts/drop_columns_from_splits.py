"""Drop columns from each split annotation file of an ESP dataset.

For every split in `<Dataset>.info.split_paths`, the script loads the
annotation file, drops the given columns (if present), and writes the result
back to the same bucket prefix with a user-supplied stem suffix
(e.g. `_v2`). Supports CSV, JSONL, and Parquet files.

Example
-------
    uv run python scripts/drop_columns_from_splits.py \\
        AnimalSpeak \\
        --columns cluster_path Unnamed:0 \\
        --suffix _v2
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Iterable

import pandas as pd

from esp_data import dataset_class_from_name
from esp_data.io import anypath


def _split_ext(path: str) -> tuple[str, str]:
    """Split `path` into (stem-with-prefix, extension) on the final dot.

    Treats `.jsonl` as a single extension.

    Returns
    -------
    tuple[str, str]
        Stem-with-prefix and extension lowercased including the leading dot.
    """
    # slash = path.rfind("/")
    # lower = path.lower()
    # for ext in (".jsonl", ".parquet", ".csv", ".json"):
    #     if lower.endswith(ext) and (len(path) - len(ext)) > slash:
    #         return path[: -len(ext)], ext
    # dot = path.rfind(".")
    # if dot == -1 or dot < slash:
    #     raise ValueError(f"Path has no file extension: {path}")
    # return path[:dot], path[dot:].lower()
    pt = anypath(path)
    stem_with_prefix = anypath("/".join(pt.parts[:-1])) / pt.stem
    ext = pt.suffix.lower()
    return str(stem_with_prefix), ext


def _suffix_path(path: str, suffix: str) -> str:
    """Insert `suffix` before the file extension of `path`.

    Returns
    -------
    str
        Path with `suffix` inserted before the extension.
    """
    head, ext = _split_ext(path)
    # if suffix is like _vX, X is int, and head ends with _vY where Y is int,
    # replace Y with X instead of appending
    m = re.match(r"^(.*_v)(\d+)$", head)
    if m and suffix.startswith("_v") and suffix[2:].isdigit():
        head = m.group(1) + suffix[2:]
        return head + ext
    return head + suffix + ext


def _read(path: str) -> pd.DataFrame:
    """Read a CSV, JSONL, or Parquet annotation file.

    Returns
    -------
    pd.DataFrame
        Loaded annotation table.

    Raises
    ------
    ValueError
        If the path has an unsupported extension.
    """
    _, ext = _split_ext(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file type: {path}")


def _write(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV, JSONL, or Parquet based on the path extension.

    Raises
    ------
    ValueError
        If the path has an unsupported extension.
    """
    _, ext = _split_ext(path)
    if ext == ".parquet":
        df.to_parquet(path, index=False)
    elif ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".jsonl":
        df.to_json(path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def drop_columns_from_splits(
    dataset_name: str,
    columns: Iterable[str],
    suffix: str,
    dry_run: bool = False,
) -> None:
    """Drop `columns` from each split of `dataset_name` and save with `suffix`.

    Parameters
    ----------
    dataset_name : str
        Name of a registered ESP dataset (e.g. `"AnimalSpeak"`).
    columns : Iterable[str]
        Column names to drop. Missing columns are reported but do not fail.
    suffix : str
        Stem suffix appended to each split file (e.g. `"_v2"`).
    dry_run : bool, optional
        If True, do not write files; only print what would happen.
    """
    cls = dataset_class_from_name(dataset_name)
    split_paths = cls.info.split_paths
    cols_to_drop = list(columns)

    for split, src in split_paths.items():
        dst = _suffix_path(src, suffix)
        print(f"\n[{split}] {src} -> {dst}")

        df = _read(src)
        present = [c for c in cols_to_drop if c in df.columns]
        missing = [c for c in cols_to_drop if c not in df.columns]
        if missing:
            print(f"  not present (skipped): {missing}")
        if present:
            df = df.drop(columns=present)
            print(f"  dropped: {present}")
        else:
            print("  nothing to drop; writing copy unchanged")

        print(f"  final shape: {df.shape}")
        if dry_run:
            print("  [dry-run] not writing")
            continue
        _write(df, dst)
        print(f"  wrote {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="Registered dataset name, e.g. AnimalSpeak")
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="Column names to drop from each split",
    )
    parser.add_argument(
        "--suffix",
        required=True,
        help="Stem suffix for the new file, e.g. _v2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing any files",
    )
    args = parser.parse_args()

    drop_columns_from_splits(
        dataset_name=args.dataset,
        columns=args.columns,
        suffix=args.suffix,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
