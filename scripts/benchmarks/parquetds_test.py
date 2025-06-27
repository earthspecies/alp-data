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

fs = filesystem_from_path("gs://")


def _make_parquet_dataset(shard_path: str, file_pattern: str = "shard*parquet") -> ds.Dataset:
    path = anypath(shard_path)
    shard_files = list([str(s) for s in path.glob(file_pattern)])
    # Log what we found for debugging
    print(f"Found {len(shard_files)} shard files in {path}")
    dataset = ds.dataset(shard_files, format="parquet", filesystem=fs)
    return dataset


class ParquetDataset(torch.utils.data.IterableDataset):
    def __init__(self, path: str) -> None:
        self._ds = _make_parquet_dataset(path)

    def __iter__(self) -> pa.RecordBatch:
        for batch in self._ds.scanner().to_batches(batch_size=1):
            yield batch


def torch_mel_spec_webds(batch: list[dict]) -> torch.Tensor:
    """Create mel spectrograms on the fly

    Parameters
    ----------
    data: dict
        Dictionary containing the audio data and metadata.

    Returns
    -------
    torch.Tensor
        Mel spectrogram of the audio data.
    """
    output = []
    for item in batch:
        audio = torch.from_numpy(item["audio"]).to(torch.float32)
        md = json.loads(item["metadata"])
        sr = md.get("sample_rate", 16000)
        if sr != 16000:
            resampler = T.Resample(sr, new_freq=16000)
            audio = resampler(audio)

        # clip to max len 10 sec
        maxlen = 10 * 16000
        audio = audio[:maxlen]
        # or pad to 10 sec
        if len(audio) < maxlen:
            audio = nnf.pad(audio, (1, maxlen - len(audio) - 1), "constant", 0)

        transform = T.MelSpectrogram(sample_rate=16000, n_fft=256, n_mels=32)
        mel_specgram = transform(audio)
        output.append(mel_specgram)

    output = torch.stack(output, dim=0)

    return output


def run_iterable_ds(pq: ds.Dataset, max_iters: int) -> tuple[float, float]:
    """Iterate over a dataset and collect stats.

    Parameters
    ----------
    ds: wds.WebDataset
        The dataset to iterate over.
    max_iters: int
        Maximum number of iterations to run.

    Returns
    -------
    tuple[float, float]
        Average and standard deviation of run times per sample.
    """
    run_times = []
    t0 = time.time()
    for i, _ in tqdm(enumerate(pq)):
        run_times.append(time.time() - t0)
        t0 = time.time()
        if (i + 1) == max_iters:
            break

    return np.mean(run_times), np.std(run_times)


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
    for i, _ in tqdm(enumerate(dl)):
        run_times.append(time.time() - t0)
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
    args = parser.parse_args()

    # EXAMPLE USAGE
    PATH_TO_BEANS_DATASET = "gs://esp-ml-datasets/beans0/processed/v0.1.0/parquet/"
    PATH_TO_NATLM_DATASET = "gs://esp-ml-datasets/naturelm/processed/v0.1.0/tar/train/"

    if args.use_beans:
        PATH_TO_DATASET = PATH_TO_BEANS_DATASET
    else:
        PATH_TO_DATASET = PATH_TO_NATLM_DATASET

    # this will shuffle across shards
    # and within shards in a buffer of 1000 samples
    pq = ParquetDataset(PATH_TO_DATASET)

    torch.manual_seed(42)
    dl = DataLoader(
        pq,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=torch_mel_spec_webds,
    )

    # Run dataloader
    print("Running DataLoader...")
    avg_run_time_dl, std_run_time_dl = run_iterable_dl(dl, args.max_iters)

    print(f"Average time per sample: {avg_run_time_dl:.2f} seconds")
    print(f"Standard deviation: {std_run_time_dl:.2f} seconds")


if __name__ == "__main__":
    main()
