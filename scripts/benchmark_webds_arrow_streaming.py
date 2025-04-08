"""Run a latency benchmark for webdataset and HF arrow streaming datasets."""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as nnf
import torchaudio.transforms as T
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from esp_data.dataset.esp_dataset import load_esp_dataset as lds
from esp_data.io.parsers import read_audio_bytes

from .temp_utils import make_simple_logger

logger = make_simple_logger("benchmark_webds_arrow_streaming", add_file_handler=True)


def data_processor(data: dict):
    audio, _ = read_audio_bytes(data["audio.wav"], "wav")
    metadata = json.loads(data["metadata.json"])
    return {"audio": audio, **metadata}


def torch_mel_spec_webds(data: dict) -> torch.Tensor:
    """Create mel spectrograms on the fly"""
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


def processor(data) -> dict:
    audio = torch.Tensor(data["audio"]).to(torch.float32)

    sr = json.loads(data["metadata"]).get("sample_rate", 16000)
    if sr != 16000:
        resampler = T.Resample(sr, new_freq=16000)
        audio = resampler(audio)

    # clip to max len 10 sec
    maxlen = 10 * 16000
    audio = audio[:maxlen]
    # or pad to 10 sec
    if len(audio) < maxlen:
        audio = nnf.pad(audio, (1, maxlen - len(audio) - 1), "constant", 0)

    return {"audio": audio}


def run_iterable_ds(ds, max_iters: int) -> tuple[float, float]:
    run_times = []
    t0 = time.time()
    for i, _ in tqdm(enumerate(ds)):
        run_times.append(time.time() - t0)
        t0 = time.time()
        if (i + 1) == max_iters:
            break

    return np.mean(run_times), np.std(run_times)


def main():
    parser = argparse.ArgumentParser(
        description="Run a latency benchmark for webdataset and HF arrow streaming datasets."
    )
    parser.add_argument("--path_to_arrow_dataset", type=str, default="gs://esp-ml-datasets/beans0/processed/v0.1.0/")
    parser.add_argument("--path_to_webdataset", type=str, default="gs://esp-ml-datasets/beans0/processed/v0.1.1/")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--shuffle_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    args = parser.parse_args()

    t0 = time.time()
    arrow_ds = lds(
        "bucket_hf", path=args.path_to_arrow_dataset, streaming=True, file_pattern="shard*arrow", split="train"
    )
    tend = time.time()
    logger.info(f"Time to load HF arrow streaming dataset: {tend - t0:.2f} seconds")

    # Load a webdataset
    t0 = time.time()
    webds = lds(
        "webdataset",
        path=args.path_to_webdataset,
        file_pattern="shard*tar",
        load_metadata=False,  # This will try and load a metadata.xx (csv, parquet or json) file, which contains an index
        data_processor=data_processor,
        shuffle_size=args.shuffle_size,  # will shuffle within shards in a buffer of 1000 samples
        shard_shuffle=True,  # shuffle the order of the shards
        shard_shuffle_size=100,  # shuffle the order of the shards
    )
    tend = time.time()
    logger.info(f"Time to load webdataset: {tend - t0:.2f} seconds")

    logger.info("Benchmarking webdataset...")
    avg_time, std_time = run_iterable_ds(webds, args.num_samples)
    logger.info(f"Average time per sample for webdataset: {avg_time:.2f} seconds, std: {std_time:.2f}")

    logger.info("Benchmarking HF arrow streaming dataset...")
    avg_time, std_time = run_iterable_ds(arrow_ds, args.num_samples)
    logger.info(f"Average time per sample for HF arrow streaming dataset: {avg_time:.2f} seconds, std: {std_time:.2f}")

    # now we do batched processing
    t0 = time.time()
    webds = lds(
        "webdataset",
        path=args.path_to_webdataset,
        file_pattern="shard*tar",
        load_metadata=False,
        data_processor=torch_mel_spec_webds,
        shuffle_size=args.shuffle_size,
        shard_shuffle=True,
        shard_shuffle_size=100,
        batch_size=args.batch_size,
        batch_collate_fn=torch.stack,
    )
    tend = time.time()

    logger.info(f"Time to load webdataset with batch processing: {tend - t0:.2f} seconds")

    logger.info("Benchmarking webdataset with batch processing...")
    avg_time, std_time = run_iterable_ds(webds, args.num_batches)
    logger.info(f"Average time per batch for webdataset: {avg_time:.2f} seconds with std: {std_time:.2f}")

    # now we do batched processing with HF arrow streaming dataset
    class IterDS(IterableDataset):
        def __init__(self, ds):
            self.ds = ds

        def __iter__(self):
            return iter(self.ds)

    ds = IterDS(arrow_ds.map(processor))

    td = DataLoader(ds, batch_size=args.batch_size, num_workers=args.dataloader_num_workers)

    logger.info("Benchmarking HF arrow streaming and torch dataloader...")
    avg_time, std_time = run_iterable_ds(td, args.num_batches)

    logger.info(
        f"Average time per batch for HF arrow streaming and torch dataloader on dataset: {avg_time:.2f} seconds with std: {std_time:.2f}"
    )


if __name__ == "__main__":
    main()
