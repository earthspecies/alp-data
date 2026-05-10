"""Iterate over all registered datasets and splits via a single ChainedDataset.

Builds a `ChainedDatasetConfig` covering every (dataset, split) pair in the
registry, dumps it to a YAML config, then iterates the chained dataset with a
PyTorch DataLoader. Per-sample failures are caught, attributed to the
originating (dataset, split), and written to a JSON report at the end so a
single broken sample does not abort the run.
"""

import json
import logging
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click
import torch
import yaml

import esp_data.datasets  # noqa: F401  -- populate the dataset registry
from esp_data import DatasetConfig, dataset_from_config
from esp_data.chain import ChainedDataset
from esp_data.dataset import ChainedDatasetConfig, Dataset, _dataset_registry

# Sentinel keys attached to a sample dict when iteration of an underlying
# dataset raises. The wrappers convert exceptions into these sentinel dicts so
# DataLoader workers don't need shared state to report failures back.
_FAILED_KEY = "_failed"
_DATASET_KEY = "_dataset"
_SPLIT_KEY = "_split"
_INDEX_KEY = "_index"
_ERROR_KEY = "_error"


def _attribution(ds: Dataset) -> tuple[str, str]:
    """Return `(dataset_name, split)` for a source dataset.

    Parameters
    ----------
    ds : Dataset
        A `Dataset` instance from `ChainedDataset._source_datasets`.

    Returns
    -------
    tuple[str, str]
        The dataset name (from `info.name`) and split (from `self.split` if
        present, else the literal string ``"<unknown>"``).
    """
    name = getattr(getattr(ds, "info", None), "name", "<unknown>")
    split = getattr(ds, "split", "<unknown>")
    return name, split


def _failure_record(name: str, split: str, index: int, exc: BaseException) -> dict[str, Any]:
    """Build a sentinel dict representing a per-sample iteration failure.

    Parameters
    ----------
    name : str
        Dataset name.
    split : str
        Split name.
    index : int
        Local index within the (dataset, split) where the failure occurred.
        For streaming datasets this is the count of samples consumed so far.
    exc : BaseException
        The exception raised by the underlying dataset.

    Returns
    -------
    dict[str, Any]
        A dict with the sentinel `_failed` key set, plus attribution and the
        stringified error.
    """
    return {
        _FAILED_KEY: True,
        _DATASET_KEY: name,
        _SPLIT_KEY: split,
        _INDEX_KEY: index,
        _ERROR_KEY: f"{type(exc).__name__}: {exc}",
    }


class _RobustStreamingChainWrapper(torch.utils.data.IterableDataset):
    """Stream the chained sub-datasets while catching per-sample failures.

    Sharding is per sub-dataset (round-robin by sub-dataset index) rather than
    per sample, so each worker iterates whole sub-datasets to completion. This
    differs from the previous round-robin-per-sample sharding but is
    appropriate for an iteration-check script and lets us cheaply attribute
    each yielded item (or sentinel failure) to a single (dataset, split).

    Parameters
    ----------
    chained : ChainedDataset
        A `ChainedDataset` instantiated in streaming mode.
    """

    def __init__(self, chained: ChainedDataset) -> None:
        super().__init__()
        self._sub_datasets = list(chained._source_datasets)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        for ds_idx, ds in enumerate(self._sub_datasets):
            if ds_idx % num_workers != worker_id:
                continue
            name, split = _attribution(ds)
            local_idx = 0
            try:
                it = iter(ds)
            except Exception as exc:
                # Failure constructing the iterator itself — report once and
                # move on to the next sub-dataset.
                yield _failure_record(name, split, -1, exc)
                continue
            while True:
                try:
                    item = next(it)
                except StopIteration:
                    break
                except Exception as exc:
                    yield _failure_record(name, split, local_idx, exc)
                    local_idx += 1
                    continue
                yield item
                local_idx += 1


class _RobustMapChainWrapper(torch.utils.data.Dataset):
    """Map-style wrapper around a non-streaming `ChainedDataset`.

    `__getitem__` catches exceptions raised by the underlying sub-datasets and
    returns a sentinel failure dict instead of propagating, so a single bad
    sample does not abort the DataLoader iteration.

    Parameters
    ----------
    chained : ChainedDataset
        A `ChainedDataset` instantiated in non-streaming mode.
    """

    def __init__(self, chained: ChainedDataset) -> None:
        self._sub_datasets = list(chained._source_datasets)
        self._lengths = [len(ds) for ds in self._sub_datasets]
        self._total = sum(self._lengths)

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        cum = 0
        for ds, length in zip(self._sub_datasets, self._lengths, strict=True):
            if idx < cum + length:
                local = idx - cum
                name, split = _attribution(ds)
                try:
                    return ds[local]
                except Exception as exc:
                    return _failure_record(name, split, local, exc)
            cum += length
        # Unreachable: bounds checked above.
        raise IndexError(idx)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("iteration_check")

# Datasets that are not "real" (virtual containers) or that have no splits.
_SKIP_DATASETS = {"chained_dataset", "concatenated_dataset"}


def collate_fn(batch: list[dict]) -> dict:
    """Collate samples by padding/truncating audio to a fixed length.

    Sentinel failure dicts (those with `_failed=True`) are separated out and
    returned under the `_failures` key so the main loop can record them
    without aborting iteration.

    Parameters
    ----------
    batch : list[dict]
        List of samples; each successful sample must contain an `audio`
        ndarray. Failure sentinels are passed through under `_failures`.

    Returns
    -------
    dict
        Dictionary with the stacked `audio` tensor of shape (B, T) (possibly
        empty if every sample failed or lacked `audio`) and a `_failures`
        list of sentinel dicts.
    """
    max_length = 10 * 16000
    failures = [s for s in batch if s.get(_FAILED_KEY)]
    good = [s for s in batch if not s.get(_FAILED_KEY) and "audio" in s]
    if good:
        audios = torch.stack(
            [
                torch.nn.functional.pad(
                    torch.from_numpy(sample["audio"][:max_length]),
                    (0, max(0, max_length - sample["audio"].shape[0])),
                )
                for sample in good
            ]
        )
    else:
        audios = torch.empty(0, max_length)
    return {"audio": audios, "_failures": failures}


def build_chain_config(
    sample_rate: int | None,
    data_root: str | None,
    streaming: bool,
) -> ChainedDatasetConfig:
    """Build a `ChainedDatasetConfig` over every (dataset, split) in the registry.

    Parameters
    ----------
    sample_rate : int | None
        Target sample rate applied to every dataset.
    data_root : str | None
        Optional `data_root` override applied to every dataset.
    streaming : bool
        Whether each per-dataset config should use streaming.

    Returns
    -------
    ChainedDatasetConfig
        Config chaining every registered dataset across all of its splits.
    """
    dataset_configs: list[DatasetConfig] = []
    for name, cls in sorted(_dataset_registry.items()):
        if name in _SKIP_DATASETS:
            continue
        splits = list(cls.info.split_paths.keys())
        if not splits:
            logger.warning("Skipping %s (no splits registered)", name)
            continue
        for split in splits:
            dataset_configs.append(
                DatasetConfig(
                    dataset_name=name,
                    split=split,
                    sample_rate=sample_rate,
                    data_root=data_root,
                    streaming=streaming,
                )
            )
    return ChainedDatasetConfig(datasets=dataset_configs)


def dump_config(chain_cfg: ChainedDatasetConfig, path: Path) -> None:
    """Dump a `ChainedDatasetConfig` to YAML under the canonical `chain:` key.

    Parameters
    ----------
    chain_cfg : ChainedDatasetConfig
        Config to serialize.
    path : Path
        Output YAML path.
    """
    payload = {"chain": chain_cfg.model_dump(exclude_none=True)}
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    logger.info("Wrote chained config to %s", path)


def write_failures_report(
    failures: dict[tuple[str, str], list[dict[str, Any]]],
    fatal: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write the per-(dataset, split) failure report to JSON.

    Parameters
    ----------
    failures : dict[tuple[str, str], list[dict[str, Any]]]
        Mapping from (dataset, split) to a list of per-sample failure
        records. Each record contains the local `index` and the stringified
        `error`.
    fatal : list[dict[str, Any]]
        Loader-level errors that aborted iteration (if any). Each entry
        carries a stringified `error` and the `batches_consumed` count at the
        time of failure.
    path : Path
        Output JSON path.
    """
    by_dataset: dict[str, dict[str, list[dict[str, Any]]]] = {}
    total = 0
    for (name, split), records in sorted(failures.items()):
        by_dataset.setdefault(name, {})[split] = records
        total += len(records)
    payload = {
        "total_failures": total,
        "fatal": fatal,
        "by_dataset": by_dataset,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    logger.info(
        "Wrote failure report to %s (%d per-sample failures, %d fatal)",
        path,
        total,
        len(fatal),
    )


@click.command()
@click.option(
    "--config-in",
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    default=None,
    help=(
        "Path to an existing chained config YAML (with a top-level `chain:` "
        "key). If omitted, a config covering every registered (dataset, split) "
        "is built instead."
    ),
)
@click.option(
    "--config-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("chained_all_datasets.yaml"),
    show_default=True,
    help=(
        "Where to write the generated chained config YAML. Ignored when `--config-in` is provided."
    ),
)
@click.option(
    "--failures-report",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("iteration_check_failures.json"),
    show_default=True,
    help="Where to write the per-(dataset, split) failure report.",
)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--num-workers", type=int, default=4, show_default=True)
@click.option(
    "--max-batches",
    type=int,
    default=-1,
    show_default=True,
    help="Max batches to iterate (-1 = full iteration).",
)
@click.option(
    "--sample-rate",
    type=int,
    default=16000,
    show_default=True,
    help="Target sample rate applied to every dataset.",
)
@click.option(
    "--data-root",
    type=str,
    default=None,
    help="Optional data_root override applied to every dataset.",
)
@click.option("--streaming", is_flag=True, help="Use streaming mode for all datasets.")
@click.option(
    "--log-interval",
    type=int,
    default=10,
    show_default=True,
    help="Log stats every N batches.",
)
@click.option(
    "--dump-only",
    is_flag=True,
    help="Only generate the YAML config; do not iterate.",
)
def main(
    config_in: Path | None,
    config_out: Path,
    failures_report: Path,
    batch_size: int,
    num_workers: int,
    max_batches: int,
    sample_rate: int,
    data_root: str | None,
    streaming: bool,
    log_interval: int,
    dump_only: bool,
) -> None:
    """Iterate a chained config, either supplied via `--config-in` or built from the registry."""
    chain_source: Path | ChainedDatasetConfig
    if config_in is not None:
        chain_source = config_in
        logger.info("Using chained config from %s", config_in)
    else:
        chain_source = build_chain_config(
            sample_rate=sample_rate,
            data_root=data_root,
            streaming=streaming,
        )
        logger.info(
            "Built chained config with %d (dataset, split) entries",
            len(chain_source.datasets),
        )
        dump_config(chain_source, config_out)

    if dump_only:
        return

    t0 = time.time()
    chained: ChainedDataset
    chained, _ = dataset_from_config(chain_source)
    logger.info("Instantiated ChainedDataset in %.2fs", time.time() - t0)

    loader_dataset: torch.utils.data.Dataset | torch.utils.data.IterableDataset
    if chained.streaming:
        loader_dataset = _RobustStreamingChainWrapper(chained)
    else:
        loader_dataset = _RobustMapChainWrapper(chained)

    loader = torch.utils.data.DataLoader(
        loader_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    failures: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    fatal: list[dict[str, Any]] = []
    n_batches = 0
    n_samples = 0
    n_failed = 0
    start = time.time()
    loader_iter = iter(loader)
    while True:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        except Exception as exc:
            # A loader-level error (e.g. worker crash, collate bug) is not
            # attributable to a single sample — record as fatal and stop.
            logger.exception("Loader-level failure; aborting iteration.")
            fatal.append(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "batches_consumed": n_batches,
                }
            )
            break

        for record in batch.get("_failures", []):
            key = (record[_DATASET_KEY], record[_SPLIT_KEY])
            failures[key].append({"index": record[_INDEX_KEY], "error": record[_ERROR_KEY]})
            n_failed += 1

        n_batches += 1
        n_samples += batch["audio"].shape[0]
        if n_batches % log_interval == 0:
            elapsed = time.time() - start
            logger.info(
                "batch %d | %d samples (%d failed) in %.2fs (%.2f samples/s)",
                n_batches,
                n_samples,
                n_failed,
                elapsed,
                n_samples / elapsed if elapsed > 0 else 0.0,
            )
        if max_batches > 0 and n_batches >= max_batches:
            logger.info("Reached max-batches=%d, stopping early.", max_batches)
            break

    elapsed = time.time() - start
    logger.info(
        "FINAL: %d batches, %d samples (%d failed) in %.2fs (%.2f samples/s)",
        n_batches,
        n_samples,
        n_failed,
        elapsed,
        n_samples / elapsed if elapsed > 0 else 0.0,
    )
    write_failures_report(failures, fatal, failures_report)


if __name__ == "__main__":
    sys.exit(main())
