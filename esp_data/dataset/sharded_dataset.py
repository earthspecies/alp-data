import asyncio
import io
import queue
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import aiofiles
import numpy as np
import pandas as pd

from esp_data.db.caches import BaseCache, LRUCache
from esp_data.file_io.functional import read_bytes, write_bytes
from esp_data.paths import AnyPath


@dataclass
class ShardInfo:
    shard_id: str
    start_idx: int
    end_idx: int
    sample_ids: List[str]
    file_size: int


class ShardedDataset:
    def __init__(
        self,
        metadata_path: str,
        shard_root: str,
        shard_size: int = 1000,  # samples per shard
        cache_size: int = 5,  # number of shards to cache
    ):
        self.metadata = pd.read_csv(metadata_path)
        self.shard_root = AnyPath(shard_root)
        self.shard_size = shard_size
        self.cache_size = cache_size
        self.shard_cache = {}  # Maps shard_id -> shard content
        self.shard_info = {}  # Maps shard_id -> ShardInfo

    def create_shards(self, file_paths: List[str], compression: bool = True) -> None:
        """Create shards from individual files."""
        for shard_idx in range(0, len(file_paths), self.shard_size):
            batch_paths = file_paths[shard_idx : shard_idx + self.shard_size]
            shard_id = f"shard_{shard_idx // self.shard_size:05d}"

            # Create tar file in memory
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode="w:gz" if compression else "w") as tar:
                for path in batch_paths:
                    file_data = read_bytes(path)
                    file_info = tarfile.TarInfo(name=path.split("/")[-1])
                    file_info.size = len(file_data)

                    # Add file to tar
                    tar.addfile(file_info, io.BytesIO(file_data))

            # Write shard
            shard_path = self.shard_root / f"{shard_id}.tar.gz"
            write_bytes(str(shard_path), tar_buffer.getvalue())

            # Store shard info
            self.shard_info[shard_id] = ShardInfo(
                shard_id=shard_id,
                start_idx=shard_idx,
                end_idx=shard_idx + len(batch_paths),
                sample_ids=list(range(shard_idx, shard_idx + len(batch_paths))),
                file_size=len(tar_buffer.getvalue()),
            )

    def _load_shard(self, shard_id: str) -> Dict[str, Any]:
        """Load a shard into memory cache if not present."""
        if shard_id not in self.shard_cache:
            # Implement LRU cache eviction if needed
            if len(self.shard_cache) >= self.cache_size:
                # Remove least recently used shard
                self.shard_cache.pop(next(iter(self.shard_cache)))

            shard_path = self.shard_root / f"{shard_id}.tar.gz"
            shard_data = read_bytes(str(shard_path))
            self.shard_cache[shard_id] = shard_data

        return self.shard_cache[shard_id]

    def _get_shard_id(self, idx: int) -> str:
        """Get shard ID containing the sample at idx."""
        shard_idx = idx // self.shard_size
        return f"shard_{shard_idx:05d}"

    def _get_sample(self, idx):
        shard_id = self._get_shard_id(idx)
        shard_data = self._load_shard(shard_id)

        # Extract specific file from shard
        tar_buffer = io.BytesIO(shard_data)
        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            # Get relative index within shard
            relative_idx = idx % self.shard_size
            member = tar.getmembers()[relative_idx]
            file_data = tar.extractfile(member).read()

        # Combine with metadata
        metadata = self.metadata.iloc[idx].to_dict()
        return {"data": file_data, "metadata": metadata}

    def __getitem__(self, idx: int, callback=None) -> Dict[str, Any]:
        """Get a single sample."""
        data = self._get_sample(idx)
        if callback is not None:
            data = callback(data)

        return data

    def __len__(self):
        return len(self.metadata)


class AsyncShardedDataset:
    def __init__(
        self,
        metadata_path: str,
        shard_root: str,
        shard_size: int = 1000,
        cache: Optional["BaseCache"] = None,
        prefetch_size: int = 2,
        num_workers: int = 4,
    ):
        self.metadata = pd.read_csv(metadata_path)
        self.shard_root = AnyPath(shard_root)
        self.shard_size = shard_size
        self.cache = cache or LRUCache(capacity=5)
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers
        self.shard_info = {}

        # Prefetch queue and thread
        self.prefetch_queue = queue.Queue(maxsize=prefetch_size)
        self.prefetch_thread = None
        self._stop_prefetch = threading.Event()

        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)

    async def _read_shard(self, shard_id: str) -> bytes:
        """Asynchronously read a shard file."""
        shard_path = self.shard_root / f"{shard_id}.tar.gz"

        # Check cache first
        cached_data = self.cache.get(shard_id)
        if cached_data is not None:
            return cached_data

        # Read from storage
        async with aiofiles.open(str(shard_path), mode="rb") as f:
            shard_data = await f.read()
            self.cache.put(shard_id, shard_data)
            return shard_data

    def _extract_sample_from_shard(self, shard_data: bytes, relative_idx: int) -> Tuple[bytes, Dict[str, Any]]:
        """Extract a single sample from a shard."""
        tar_buffer = io.BytesIO(shard_data)
        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            member = tar.getmembers()[relative_idx]
            file_data = tar.extractfile(member).read()
            return file_data

    async def _get_samples(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Get multiple samples asynchronously."""
        # Group indices by shard
        shard_groups = {}
        for idx in indices:
            shard_id = f"shard_{idx // self.shard_size:05d}"
            if shard_id not in shard_groups:
                shard_groups[shard_id] = []
            shard_groups[shard_id].append(idx)

        # Load shards concurrently
        shard_futures = []
        for shard_id, shard_indices in shard_groups.items():
            shard_futures.append(self._read_shard(shard_id))

        shard_data = await asyncio.gather(*shard_futures)

        # Process samples using thread pool
        results = []
        futures = []

        for shard_data, (shard_id, shard_indices) in zip(shard_data, shard_groups.items()):
            for idx in shard_indices:
                relative_idx = idx % self.shard_size
                future = self.thread_pool.submit(self._extract_sample_from_shard, shard_data, relative_idx)
                futures.append((idx, future))

        # Collect results in original order
        results = [None] * len(indices)
        for idx, future in futures:
            file_data = future.result()
            metadata = self.metadata.iloc[idx].to_dict()
            results[indices.index(idx)] = {"data": file_data, "metadata": metadata}

        return results

    def _start_prefetch(self, start_idx: int) -> None:
        """Start prefetch thread."""

        def prefetch_worker():
            current_idx = start_idx
            while not self._stop_prefetch.is_set():
                if self.prefetch_queue.full():
                    continue

                # Get next batch of indices
                batch_size = min(self.shard_size, len(self) - current_idx)
                if batch_size <= 0:
                    break

                indices = list(range(current_idx, current_idx + batch_size))

                # Run async fetch in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                batch_data = loop.run_until_complete(self._get_samples(indices))
                loop.close()

                # Add to queue
                self.prefetch_queue.put(batch_data)
                current_idx += batch_size

        self.prefetch_thread = threading.Thread(target=prefetch_worker)
        self.prefetch_thread.start()

    def _stop_prefetch_thread(self) -> None:
        """Stop prefetch thread."""
        if self.prefetch_thread is not None:
            self._stop_prefetch.set()
            self.prefetch_thread.join()
            self.prefetch_thread = None
            self._stop_prefetch.clear()

    async def get_batch(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Get a batch of samples asynchronously."""
        return await self._get_samples(indices)

    def __len__(self) -> int:
        return len(self.metadata)

    async def stream(self, batch_size: int, shuffle: bool = False) -> AsyncIterator[List[Dict[str, Any]]]:
        """Stream batches of data asynchronously."""
        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)

        # Start prefetching
        self._start_prefetch(0)

        try:
            for start_idx in range(0, len(self), batch_size):
                end_idx = min(start_idx + batch_size, len(self))
                batch_indices = indices[start_idx:end_idx].tolist()

                # Try to get from prefetch queue first
                try:
                    batch_data = self.prefetch_queue.get_nowait()
                except queue.Empty:
                    # If queue is empty, fetch directly
                    batch_data = await self._get_samples(batch_indices)

                yield batch_data

        finally:
            self._stop_prefetch_thread()


# Example usage:
async def main():
    dataset = AsyncShardedDataset(
        metadata_path="metadata.csv", shard_root="gs://my-bucket/shards", shard_size=1000, prefetch_size=2
    )

    # Get a batch
    batch = await dataset.get_batch([0, 1, 2, 3])

    # Stream batches
    async for batch in dataset.stream(batch_size=32, shuffle=True):
        # Process batch
        pass


# if __name__ == "__main__":
#     asyncio.run(main())


# Example usage:
if __name__ == "__main__":
    # Initialize dataset
    dataset = ShardedDataset(metadata_path="metadata.csv", shard_root="gs://my-bucket/shards", shard_size=1000)

    # Create shards from individual files
    file_paths = ["path/to/file1.wav", "path/to/file2.wav", ...]
    dataset.create_shards(file_paths)

    # Access data
    sample = dataset[0]  # Returns dict with data and metadata
