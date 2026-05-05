"""Iterate over all registered datasets and splits via a single ChainedDataset.

Builds a `ChainedDatasetConfig` covering every (dataset, split) pair in the
registry, dumps it to a YAML config, then iterates the chained dataset with a
PyTorch DataLoader. Any failures are reported per (dataset, split).
"""

import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import click
import torch
import yaml

import esp_data.datasets  # noqa: F401  -- populate the dataset registry
from esp_data import DatasetConfig, dataset_from_config
from esp_data.chain import ChainedDataset
from esp_data.dataset import ChainedDatasetConfig, _dataset_registry


class _StreamingChainWrapper(torch.utils.data.IterableDataset):
    """Adapt an esp_data streaming `ChainedDataset` to torch's `IterableDataset`.

    `esp_data.Dataset` does not inherit from `torch.utils.data.IterableDataset`,
    so a `DataLoader` treats a streaming `ChainedDataset` as map-style and
    fails to use `__iter__`. This wrapper exposes the chain via the
    `IterableDataset` protocol, sharding samples round-robin across DataLoader
    workers so each item is yielded by exactly one worker.

    Parameters
    ----------
    chained : ChainedDataset
        A `ChainedDataset` instantiated in streaming mode.
    """

    def __init__(self, chained: ChainedDataset) -> None:
        super().__init__()
        self._chained = chained

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            yield from self._chained
            return
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        for i, item in enumerate(self._chained):
            if i % num_workers == worker_id:
                yield item


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

    Parameters
    ----------
    batch : list[dict]
        List of samples; each must contain an `audio` ndarray.

    Returns
    -------
    dict
        Dictionary with the stacked `audio` tensor of shape (B, T).
    """
    max_length = 10 * 16000
    audios = torch.stack(
        [
            torch.nn.functional.pad(
                torch.from_numpy(sample["audio"][:max_length]),
                (0, max(0, max_length - sample["audio"].shape[0])),
            )
            for sample in batch
            if "audio" in sample
        ]
    )
    return {"audio": audios}


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
        loader_dataset = _StreamingChainWrapper(chained)
    else:
        loader_dataset = chained

    loader = torch.utils.data.DataLoader(
        loader_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    n_batches = 0
    n_samples = 0
    start = time.time()
    for batch in loader:
        n_batches += 1
        n_samples += batch["audio"].shape[0]
        if n_batches % log_interval == 0:
            elapsed = time.time() - start
            logger.info(
                "batch %d | %d samples in %.2fs (%.2f samples/s)",
                n_batches,
                n_samples,
                elapsed,
                n_samples / elapsed,
            )
        if max_batches > 0 and n_batches >= max_batches:
            logger.info("Reached max-batches=%d, stopping early.", max_batches)
            break

    elapsed = time.time() - start
    logger.info(
        "FINAL: %d batches, %d samples in %.2fs (%.2f samples/s)",
        n_batches,
        n_samples,
        elapsed,
        n_samples / elapsed if elapsed > 0 else 0.0,
    )


if __name__ == "__main__":
    sys.exit(main())
