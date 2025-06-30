# /// script
# dependencies = [
#   "webdataset>=0.2.111",
# ]
# ///
"""
Usage:
uv run --with torch --with torchaudio --with tqdm \
 --with webdataset python scripts/benchmarks/webds_test.py \
  --use_beans \
  --num_workers 8 \
  --batch_size 256 \
  --max_iters 100
"""

import argparse
import io
import json
import time
from typing import Callable, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as nnf
import torchaudio.transforms as T
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm

from esp_data.io import AnyPathT, anypath


def read_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as audio_buffer:
        data, samplerate = sf.read(audio_buffer)

    return data, samplerate


def data_processor(data: dict) -> dict:
    """Process the data from the dataset.
    Returns
    -------
    dict
        A dictionary containing the audio data and metadata.
    """
    audio, _ = read_audio_bytes(data["audio.wav"])
    metadata = json.loads(data["metadata.json"])
    return {"audio": audio, **metadata}


def torch_mel_spec_webds(data: dict) -> torch.Tensor:
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
    data = data_processor(data)
    audio = torch.from_numpy(data["audio"]).to(torch.float32)

    sr = data.get("sample_rate", 16000)
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
    return mel_specgram


def load_dataset(
    path: Union[str, AnyPathT],
    file_pattern: str = "shard*tar",
    data_processor: Callable = None,
    shuffle_size: int | None = None,
    batch_size: int | None = None,
    shard_shuffle: bool = False,
    shard_shuffle_size: int = 1000,
    split_by_worker: bool = False,
    batch_collate_fn: Callable = None,
    seed: int | bool | None = 42,
) -> wds.WebDataset:
    """Create a pipeline for loading the dataset

    Arguments
    ---------
    path: str | AnyPath
            Path to the directory where the sharded dataset will be stored or
            is already stored.
    file_pattern: str, optional
        Pattern to match the shard files.
    data_processor: Callable, optional
        Function to process the data.
    shuffle_size: int, optional
        Size of the shuffle buffer.
    batch_size: int, optional
        Batch size for processing audio files.
    shard_shuffle: bool, optional
        Whether to shuffle the shards.
    shard_shuffle_size: int, optional
        Size of the shuffle buffer for shards.
    split_by_worker: bool, optional
        Whether to split the dataset by worker.
    batch_collate_fn: Callable, optional
        Function to collate the batch.
    seed Union[int, bool, None]:
        Seed for shuffling. Defaults to True, random seed. If None, means no shuffling!

    Returns
    -------
        wds.WebDataset: WebDataset object

    Raises
    ------
    FileNotFoundError
        If no shard files are found in the specified path.
    """
    path = anypath(path)
    shard_files = list([str(s) for s in path.glob(file_pattern)])

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {path}")

    # Log what we found for debugging
    print(f"Found {len(shard_files)} shard files in {path}")

    webds_kwargs = {"shardshuffle": shard_shuffle_size if shard_shuffle else False}
    if shard_shuffle and seed is not None:
        webds_kwargs["seed"] = seed
    if split_by_worker:
        webds_kwargs["workersplitter"] = wds.split_by_worker

    webds = wds.WebDataset(shard_files, **webds_kwargs)

    if shuffle_size:
        webds = webds.shuffle(shuffle_size, seed=seed if seed is not None else 42)
    if data_processor:
        webds = webds.map(data_processor)
    if batch_size is not None:
        webds = webds.batched(batch_size, collation_fn=batch_collate_fn)

    return webds


def run_iterable_ds(ds: wds.WebDataset, max_iters: int) -> tuple[float, float]:
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
    for i, _ in tqdm(enumerate(ds)):
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
    parser = argparse.ArgumentParser(description="Benchmark WebDataset loading performance.")

    parser.add_argument(
        "--use_beans",
        action="store_true",
        help="Use the Beans dataset for benchmarking. If False, uses NatureLM dataset.",
    )
    parser.add_argument(
        "--path_to_beans",
        type=str,
        default="gs://esp-ml-datasets/beans0/processed/v0.1.0/tar/",
        help="Path to the Beans dataset tar files.",
    )
    parser.add_argument(
        "--path_to_naturelm",
        type=str,
        default="gs://esp-ml-datasets/naturelm/processed/v0.1.0/tar/train/",
        help="Path to the NatureLM dataset tar files.",
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
        "--shard_shuffle",
        action="store_true",
        help="Shuffle the shards of the dataset.",
    )
    parser.add_argument(
        "--shard_shuffle_size",
        type=int,
        default=100,
        help="Size of the shuffle buffer for the shards. Defaults to 1000.",
    )
    parser.add_argument(
        "--shuffle_size",
        type=int,
        default=None,
        help="Size of the shuffle buffer for the dataset. If None, no shuffling is applied.",
    )
    parser.add_argument(
        "--shuffle_dataloader",
        action="store_true",
        help="Shuffle the DataLoader.",
    )
    args = parser.parse_args()

    if args.use_beans:
        PATH_TO_DATASET = args.path_to_beans
    else:
        PATH_TO_DATASET = args.path_to_naturelm

    # this will shuffle across shards
    # and within shards in a buffer of 1000 samples
    webds1 = load_dataset(
        path=PATH_TO_DATASET,
        data_processor=torch_mel_spec_webds,
        shuffle_size=args.shuffle_size,
        shard_shuffle=args.shard_shuffle,
        shard_shuffle_size=args.shard_shuffle_size,
        batch_collate_fn=torch.stack,
        batch_size=1,
    )

    torch.manual_seed(42)
    webdl1 = DataLoader(
        webds1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=torch.cat,
        shuffle=args.shuffle_dataloader,
    )

    # Run dataloader
    print("Running DataLoader...")
    avg_run_time_dl, std_run_time_dl = run_iterable_dl(webdl1, args.max_iters)

    print(f"Average time per sample: {avg_run_time_dl:.2f} seconds")
    print(f"Standard deviation: {std_run_time_dl:.2f} seconds")


if __name__ == "__main__":
    main()
