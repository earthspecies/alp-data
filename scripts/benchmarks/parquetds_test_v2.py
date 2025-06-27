# /// script
# dependencies = [
#   "webdataset>=0.2.111",
# ]
# ///
"""
Usage:
uv run --with torch --with torchaudio --with tqdm \
 --with pyarrow python scripts/benchmarks/parquetds_test.py \
  --use_beans \
  --num_workers 8 \
  --batch_size 256 \
  --max_iters 100
"""

import argparse
import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import torch
import torch.nn.functional as nnf
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from esp_data.io import anypath, filesystem_from_path


def _get_shard_files(shard_path: str, file_pattern: str = "shard*parquet") -> list[str]:
    """Get list of shard files from the given path.

    Parameters
    ----------
    shard_path: str
        Path to the directory containing shard files.
    file_pattern: str
        Pattern to match shard files (default: "shard*parquet").

    Returns
    -------
    list[str]
        List of sorted shard file paths matching the pattern.
    """
    path = anypath(shard_path)
    shard_files = list([str(s) for s in path.glob(file_pattern)])
    return sorted(shard_files)  # Sort for consistent ordering


def _make_parquet_dataset_from_files(shard_files: list[str]) -> ds.Dataset:
    """Create PyArrow dataset from specific shard files.

    Parameters
    ----------
    shard_files: list[str]
        List of shard file paths to include in the dataset.

    Returns
    -------
    ds.Dataset
        PyArrow dataset created from the specified shard files.
    """
    fs = filesystem_from_path("gs://")
    dataset = ds.dataset(shard_files, format="parquet", filesystem=fs)
    return dataset


class ShardedParquetDataset(torch.utils.data.IterableDataset):
    def __init__(self, path: str, worker_id: int = 0, num_workers: int = 1) -> None:
        self.path = path
        self.worker_id = worker_id
        self.num_workers = num_workers
        self._ds = None
        self._worker_shard_files = None

    def _get_worker_shards(self) -> list[str]:
        """Get shard files assigned to this worker.

        Returns
        -------
        list[str]
            List of shard file paths assigned to this worker based on worker ID.
        """
        if self._worker_shard_files is None:
            all_shard_files = _get_shard_files(self.path)
            print(f"Total shard files found: {len(all_shard_files)}")

            # Distribute shards among workers
            worker_shards = []
            for i, shard_file in enumerate(all_shard_files):
                if i % self.num_workers == self.worker_id:
                    worker_shards.append(shard_file)

            self._worker_shard_files = worker_shards
            print(f"Worker {self.worker_id} assigned {len(worker_shards)} shards")

        return self._worker_shard_files

    def _get_dataset(self) -> ds.Dataset | None:
        """Lazy initialization of dataset with worker-specific shards.

        Returns
        -------
        ds.Dataset or None
            PyArrow dataset created from the worker's assigned shard files,
            or None if no shards are assigned.
        """
        if self._ds is None:
            worker_shards = self._get_worker_shards()
            if worker_shards:
                self._ds = _make_parquet_dataset_from_files(worker_shards)
            else:
                # Return empty dataset if no shards assigned
                self._ds = None
        return self._ds

    def __iter__(self) -> pa.RecordBatch:
        dataset = self._get_dataset()
        if dataset is not None:
            for batch in dataset.scanner(batch_size=1).to_batches():
                yield batch


class WorkerAwareParquetDataset(torch.utils.data.IterableDataset):
    """Dataset that automatically detects worker info and distributes shards."""

    def __init__(self, path: str, distribute_shards: bool = True) -> None:
        self.path = path
        self.distribute_shards = distribute_shards
        self._ds = None
        self._all_shard_files = None

    def _get_all_shard_files(self) -> list[str]:
        """Cache all shard files.

        Returns
        -------
        list[str]
            List of all shard file paths in the dataset directory.
        """
        if self._all_shard_files is None:
            self._all_shard_files = _get_shard_files(self.path)
            print(f"Found {len(self._all_shard_files)} total shard files")
        return self._all_shard_files

    def _get_worker_info(self) -> tuple[int, int]:
        """Get current worker information.

        Returns
        -------
        tuple[int, int]
            Worker ID and total number of workers.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-threaded
            return 0, 1
        else:
            return worker_info.id, worker_info.num_workers

    def _get_dataset(self) -> ds.Dataset | None:
        """Initialize dataset with worker-specific shards.

        Returns
        -------
        ds.Dataset or None
            PyArrow dataset created from the worker's assigned shard files,
        """
        if self._ds is None:
            worker_id, num_workers = self._get_worker_info()
            all_shards = self._get_all_shard_files()

            if self.distribute_shards:
                # Distribute shards to this worker
                worker_shards = [
                    shard for i, shard in enumerate(all_shards) if i % num_workers == worker_id
                ]
                print(
                    f"Worker {worker_id}/{num_workers} processing"
                    f"{len(worker_shards)} shards (distributed)"
                )
            else:
                # All workers process all shards
                worker_shards = all_shards
                print(
                    f"Worker {worker_id}/{num_workers} processing"
                    f"{len(worker_shards)} shards (shared)"
                )

            if worker_shards:
                self._ds = _make_parquet_dataset_from_files(worker_shards)
            else:
                self._ds = None

        return self._ds

    def __iter__(self) -> pa.RecordBatch:
        dataset = self._get_dataset()
        if dataset is not None:
            for batch in dataset.scanner(batch_size=1).to_batches():
                yield batch


def worker_init_fn(worker_id: int) -> None:
    print(f"Initializing worker {worker_id}")


def torch_mel_spec_webds(batch: list) -> torch.Tensor:
    """Create mel spectrograms on the fly

    Parameters
    ----------
    batch: list
        List of PyArrow RecordBatch objects or dicts containing audio data.

    Returns
    -------
    torch.Tensor
        Mel spectrogram tensor batch.
    """
    output = []
    for item in batch:
        # Handle PyArrow RecordBatch
        if isinstance(item, pa.RecordBatch):
            # Extract first (and only) row from the batch
            audio_array = item["audio"][0].as_py()
            metadata_str = item["metadata"][0].as_py()
            audio = torch.from_numpy(np.array(audio_array)).to(torch.float32)
            md = json.loads(metadata_str)
        else:
            # Handle dict format (fallback)
            audio = torch.from_numpy(item["audio"]).to(torch.float32)
            md = json.loads(item["metadata"])

        sr = md.get("sample_rate", 16000)
        if sr != 16000:
            resampler = T.Resample(sr, new_freq=16000)
            audio = resampler(audio)

        # Clip to max len 10 sec
        maxlen = 10 * 16000
        audio = audio[:maxlen]
        # Pad to 10 sec if needed
        if len(audio) < maxlen:
            audio = nnf.pad(audio, (0, maxlen - len(audio)), "constant", 0)

        transform = T.MelSpectrogram(sample_rate=16000, n_fft=256, n_mels=32)
        mel_specgram = transform(audio)
        output.append(mel_specgram)

    if output:
        output = torch.stack(output, dim=0)
        return output
    else:
        # Return empty tensor if no data
        return torch.empty(0, 32, 626)  # Empty tensor with correct shape


def run_iterable_dl(dl: DataLoader, max_iters: int) -> tuple[float, float]:
    """Iterate over a DataLoader and collect stats.

    Parameters
    ----------
    dl: DataLoader
        The DataLoader to iterate over.
    max_iters: int
        Maximum number of iterations to run.

    Returns
    -------
    tuple[float, float]
        Average and standard deviation of run times per sample.
    """
    run_times = []
    t0 = time.time()
    for i, batch in tqdm(enumerate(dl)):
        run_times.append(time.time() - t0)
        if batch.numel() > 0:  # Only count non-empty batches
            t0 = time.time()
        if (i + 1) == max_iters:
            break

    return np.mean(run_times), np.std(run_times)


def main() -> None:
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark Parquet dataset loading performance.")

    parser.add_argument(
        "--use_beans",
        action="store_true",
        help="Use the Beans dataset for benchmarking. If False, uses NatureLM dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of workers for the DataLoader.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for the DataLoader.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="Maximum number of iterations to run for the benchmark.",
    )
    parser.add_argument(
        "--shard_strategy",
        choices=["shared", "distributed"],
        default="distributed",
        help="Shard distribution strategy: 'shared' (all workers read all shards) or "
        "'distributed' (workers get separate shards)",
    )
    args = parser.parse_args()

    # Dataset paths
    PATH_TO_BEANS_DATASET = "gs://esp-ml-datasets/beans0/processed/v0.1.0/parquet/"
    PATH_TO_NATLM_DATASET = "gs://esp-ml-datasets/naturelm/processed/v0.1.0/tar/train/"

    if args.use_beans:
        PATH_TO_DATASET = PATH_TO_BEANS_DATASET
    else:
        PATH_TO_DATASET = PATH_TO_NATLM_DATASET

    # Choose dataset class based on strategy
    if args.shard_strategy == "distributed":
        print("Using distributed shard strategy - each worker gets different shards")
        pq = WorkerAwareParquetDataset(PATH_TO_DATASET, distribute_shards=True)
    else:
        print("Using shared shard strategy - all workers read from all shards")
        pq = WorkerAwareParquetDataset(PATH_TO_DATASET, distribute_shards=False)

    torch.manual_seed(42)

    print(f"Using {args.num_workers} workers with batch size {args.batch_size}")
    dl = DataLoader(
        pq,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=torch_mel_spec_webds,
        worker_init_fn=worker_init_fn,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=False,
    )

    # Run dataloader
    print("Running DataLoader...")
    avg_run_time_dl, std_run_time_dl = run_iterable_dl(dl, args.max_iters)

    print(f"Average time per batch: {avg_run_time_dl:.4f} seconds")
    print(f"Standard deviation: {std_run_time_dl:.4f} seconds")
    print(f"Effective samples per second: {args.batch_size / avg_run_time_dl:.2f}")


if __name__ == "__main__":
    main()
