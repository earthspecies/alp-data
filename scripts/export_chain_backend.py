"""Materialize a NatureLM chain manifest and export its full backend.

This is an export-only path for inspecting exactly what a chain dataset
materializes, without initializing a model or starting training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from esp_data import dataset_from_config
from esp_data.backends.polars_backend import PolarsBackend
from esp_data.chain import ChainedDataset
from esp_data.dataset import ChainedDatasetConfig, DatasetConfig
from esp_data.io import AnyPathT, filesystem_from_path, read_yaml

ROOT = Path(__file__).resolve().parent.parent
NATURELM_PROJECT = ROOT / "esp-research" / "projects" / "NatureLM-audio-v1.5"
DEFAULT_MANIFEST = (
    NATURELM_PROJECT / "configs" / "datasets" / "manifest_train_stage2_variations.yml"
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "exports" / "stage2_variations_backend"

# Register NatureLM transforms and make the project-local config modules importable.
sys.path.insert(0, str(NATURELM_PROJECT))
import data.transforms  # noqa: E402, F401


def _merge_chain_configs(*paths: str | Path | AnyPathT) -> dict:
    """Merge included chain config files.

    Returns
    -------
    dict
        Merged raw chain config dictionary.

    Raises
    ------
    ValueError
        If an included config is not a chain config.
    """
    all_datasets = []
    for path in paths:
        data = read_yaml(path)
        if "chain" not in data:
            raise ValueError(f"Config file {path} does not have a 'chain' key")
        all_datasets.extend(data["chain"]["datasets"])
    return {"chain": {"datasets": all_datasets}}


def _prepare_preserved_backend_transforms(raw_configs: list[dict]) -> tuple[int, int]:
    """Rewrite transforms for full-column backend materialization.

    Returns
    -------
    tuple[int, int]
        Counts of removed ``select_columns`` transforms and rewritten
        ``rename_columns`` transforms.
    """
    removed_select = 0
    rewritten_rename = 0
    for cfg in raw_configs:
        transformations = cfg.get("transformations")
        if transformations is None:
            continue
        preserved_transforms = []
        for transform in transformations:
            transform_type = transform.get("type")
            if transform_type == "select_columns":
                removed_select += 1
                continue
            if transform_type == "rename_columns":
                transform = {**transform, "type": "preserve_rename_columns"}
                rewritten_rename += 1
            preserved_transforms.append(transform)
        cfg["transformations"] = preserved_transforms
    return removed_select, rewritten_rename


def _load_chain_config(
    manifest_path: Path,
    *,
    sample_rate: int | None,
    preserve_backend_columns: bool,
) -> ChainedDatasetConfig:
    """Load a manifest into a validated chained dataset config.

    Returns
    -------
    ChainedDatasetConfig
        Validated chain config ready for ``dataset_from_config``.

    Raises
    ------
    ValueError
        If the manifest does not include chain config files.
    """
    data = read_yaml(manifest_path)
    if "include" not in data:
        raise ValueError(f"Manifest {manifest_path} does not have an 'include' key")

    include_paths = [manifest_path.parent / inc for inc in data["include"]]
    merged = _merge_chain_configs(*include_paths)
    raw_configs = merged["chain"]["datasets"]
    if sample_rate is not None:
        for cfg in raw_configs:
            cfg["sample_rate"] = sample_rate
    if preserve_backend_columns:
        removed_select, rewritten_rename = _prepare_preserved_backend_transforms(raw_configs)
        print(
            "Preserve-column preflight: "
            f"removed {removed_select} select_columns transforms; "
            f"rewrote {rewritten_rename} rename_columns transforms.",
            flush=True,
        )

    return ChainedDatasetConfig(
        datasets=[DatasetConfig.model_validate(ds) for ds in raw_configs]
    )


def _select_entry_shard(
    chain_config: ChainedDatasetConfig,
    *,
    shard_index: int | None,
    shard_count: int | None,
) -> ChainedDatasetConfig:
    """Select a contiguous shard of chain entries.

    Parameters
    ----------
    chain_config
        Full chain configuration.
    shard_index
        Zero-based shard index, or ``None`` to keep all entries.
    shard_count
        Total number of entry shards, or ``None`` to keep all entries.

    Returns
    -------
    ChainedDatasetConfig
        Chain configuration containing only the selected entries.

    Raises
    ------
    ValueError
        If only one sharding argument is provided, or if the shard index is
        outside ``[0, shard_count)``.
    """
    if shard_index is None and shard_count is None:
        return chain_config
    if shard_index is None or shard_count is None:
        raise ValueError("Both --entry-shard-index and --entry-shard-count are required.")
    if shard_count <= 0:
        raise ValueError("--entry-shard-count must be positive.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--entry-shard-index must satisfy 0 <= index < count.")

    total_entries = len(chain_config.datasets)
    start = total_entries * shard_index // shard_count
    stop = total_entries * (shard_index + 1) // shard_count
    selected = chain_config.datasets[start:stop]
    for original_idx, cfg in zip(range(start, stop), selected, strict=True):
        transforms = list(cfg.transformations or [])
        transforms.append(
            {
                "type": "set_columns",
                "columns": {
                    "_original_chain_idx": original_idx,
                    "_entry_shard_index": shard_index,
                    "_entry_shard_count": shard_count,
                },
            }
        )
        cfg.transformations = transforms

    print(
        "Entry-shard selection: "
        f"shard {shard_index + 1}/{shard_count}, "
        f"entries [{start}, {stop}) of {total_entries}.",
        flush=True,
    )
    return ChainedDatasetConfig(datasets=selected)


def _source_metadata(dataset: ChainedDataset) -> pl.DataFrame:
    """Return chain index metadata for source dataset provenance.

    Returns
    -------
    pl.DataFrame
        Mapping from chain index to readable source dataset fields.
    """
    return pl.DataFrame(
        {
            "_chain_idx": list(range(len(dataset._source_datasets))),
            "_source_dataset_name": [
                getattr(ds.info, "name", "unknown") for ds in dataset._source_datasets
            ],
            "_source_split": [
                str(getattr(ds, "split", "unknown")) for ds in dataset._source_datasets
            ],
        }
    ).with_columns(pl.col("_chain_idx").cast(pl.Int32))


def _with_source_metadata(dataset: ChainedDataset, df: pl.DataFrame) -> pl.DataFrame:
    """Attach readable source dataset provenance columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with source metadata columns when ``_chain_idx`` is present.
    """
    if "_chain_idx" not in df.columns:
        return df
    return df.join(_source_metadata(dataset), on="_chain_idx", how="left")


def _task_label(df: pl.DataFrame) -> str:
    """Return a readable task label for a backend group.

    Returns
    -------
    str
        Task label, or ``unknown`` when the group has no task column.
    """
    if "task" not in df.columns or len(df) == 0:
        return "unknown"
    return str(df.row(0, named=True).get("task", "unknown"))


def _save_chain_backend(
    dataset: ChainedDataset,
    output_dir: Path,
    *,
    max_rows: int | None = None,
    save_entries: bool = True,
    shard_full_chain: bool = False,
) -> None:
    """Save a chained dataset's full backend and optional per-entry backends.

    Parameters
    ----------
    dataset
        Materialized chained dataset to export.
    output_dir
        Directory where JSONL files will be written.
    max_rows
        Optional cap on rows written to ``full_chain.jsonl`` for smoke tests.
    save_entries
        Whether to also write per-entry JSONLs under ``entries/``.
    shard_full_chain
        Write ``full_chain`` as one shard per chain entry. This avoids
        allocating a second full-size DataFrame during export.

    Raises
    ------
    ValueError
        If the dataset has no consolidated backend.
    """
    if dataset._data is None:
        raise ValueError(
            "Expected a consolidated ChainedDataset backend, but dataset._data is None"
        )

    out_str = str(output_dir)
    fs = filesystem_from_path(out_str)
    fs.mkdirs(out_str, exist_ok=True)

    df = dataset._data._df
    data_cols = [c for c in df.columns if c != "_chain_idx"]
    source_meta = {
        idx: {
            "_source_dataset_name": getattr(ds.info, "name", "unknown"),
            "_source_split": str(getattr(ds, "split", "unknown")),
        }
        for idx, ds in enumerate(dataset._source_datasets)
    }

    if shard_full_chain:
        shards_str = f"{out_str}/full_chain_shards"
        fs.mkdirs(shards_str, exist_ok=True)
        remaining = max_rows
        written = 0
        for chain_idx, group_df in df.group_by("_chain_idx", maintain_order=True):
            idx = int(chain_idx[0]) if isinstance(chain_idx, tuple) else int(chain_idx)
            shard_df = group_df
            if remaining is not None:
                if remaining <= 0:
                    break
                shard_df = shard_df.head(remaining)
                remaining -= len(shard_df)
            meta = source_meta[idx]
            shard_df = shard_df.with_columns(
                pl.lit(meta["_source_dataset_name"]).alias("_source_dataset_name"),
                pl.lit(meta["_source_split"]).alias("_source_split"),
            )
            source = dataset._source_datasets[idx]
            ds_name = getattr(source.info, "name", "unknown").replace("/", "_")
            label = f"{ds_name}__{_task_label(shard_df.select(data_cols))}"
            path = f"{shards_str}/{idx:03d}_{label}.jsonl"
            PolarsBackend(shard_df).to_jsonl(path)
            written += len(shard_df)
            print(f"Saved full-chain shard [{idx}] {label} ({len(shard_df)} rows)", flush=True)
        print(f"Saved sharded full chain backend ({written} rows) -> {shards_str}", flush=True)
    else:
        full_df = _with_source_metadata(dataset, df)
        if max_rows is not None:
            full_df = full_df.head(max_rows)

        full_path = f"{out_str}/full_chain.jsonl"
        PolarsBackend(full_df).to_jsonl(full_path)
        print(f"Saved full chain backend ({len(full_df)} rows) -> {full_path}", flush=True)

    if not save_entries:
        return

    entries_str = f"{out_str}/entries"
    fs.mkdirs(entries_str, exist_ok=True)
    for chain_idx, group_df in df.group_by("_chain_idx", maintain_order=True):
        idx = int(chain_idx[0]) if isinstance(chain_idx, tuple) else int(chain_idx)
        sub_df = group_df.select(data_cols)
        source = dataset._source_datasets[idx]
        ds_name = getattr(source.info, "name", "unknown").replace("/", "_")
        label = f"{ds_name}__{_task_label(sub_df)}"
        path = f"{entries_str}/{idx:03d}_{label}.jsonl"
        PolarsBackend(sub_df).to_jsonl(path)
        print(f"Saved entry [{idx}] {label} ({len(sub_df)} rows) -> {path}", flush=True)


def _print_columns(dataset: ChainedDataset) -> None:
    """Print exported backend columns for quick inspection."""
    if dataset._data is None:
        return
    columns = dataset._data.columns
    print(f"Columns ({len(columns)}): {', '.join(columns)}", flush=True)


def main() -> None:
    """CLI entry point.

    Raises
    ------
    TypeError
        If the loaded manifest does not materialize to a ``ChainedDataset``.
    """
    parser = argparse.ArgumentParser(description="Export a full NatureLM chain backend.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Chain manifest YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write exported backend JSONL files.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample-rate override to match training config materialization.",
    )
    parser.add_argument(
        "--preserve-backend-columns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove select_columns transforms before materialization.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for rows written to full_chain.jsonl.",
    )
    parser.add_argument(
        "--save-entries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write per-chain-entry backend JSONLs.",
    )
    parser.add_argument(
        "--shard-full-chain",
        action="store_true",
        help="Write full_chain as per-entry shards to reduce export memory.",
    )
    parser.add_argument(
        "--entry-shard-index",
        type=int,
        default=None,
        help="Zero-based contiguous chain-entry shard to materialize.",
    )
    parser.add_argument(
        "--entry-shard-count",
        type=int,
        default=None,
        help="Total number of contiguous chain-entry shards.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate config rewrites and exit before materializing the dataset.",
    )
    args = parser.parse_args()

    print(f"Manifest: {args.manifest}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Preserve backend columns: {args.preserve_backend_columns}", flush=True)

    chain_config = _load_chain_config(
        args.manifest,
        sample_rate=args.sample_rate,
        preserve_backend_columns=args.preserve_backend_columns,
    )
    chain_config = _select_entry_shard(
        chain_config,
        shard_index=args.entry_shard_index,
        shard_count=args.entry_shard_count,
    )
    if args.preflight_only:
        print(
            f"Preflight OK: validated {len(chain_config.datasets)} chain entries.",
            flush=True,
        )
        return

    train_ds, _ = dataset_from_config(chain_config)
    if not isinstance(train_ds, ChainedDataset):
        raise TypeError(f"Expected ChainedDataset, got {type(train_ds).__name__}")

    print(f"Materialized rows: {len(train_ds)}", flush=True)
    _print_columns(train_ds)
    _save_chain_backend(
        train_ds,
        args.output_dir,
        max_rows=args.max_rows,
        save_entries=args.save_entries,
        shard_full_chain=args.shard_full_chain,
    )


if __name__ == "__main__":
    main()
