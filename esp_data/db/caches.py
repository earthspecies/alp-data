import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypeVar

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0


class BaseCache(ABC):
    """Abstract base class for cache implementations."""

    def __init__(self):
        self.stats = CacheStats()

    @abstractmethod
    def get(self, key: K) -> Optional[V]:
        """Get item from cache."""
        pass

    @abstractmethod
    def put(self, key: K, value: V) -> None:
        """Put item in cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all items from cache."""
        pass

    def get_stats(self) -> CacheStats:
        return self.stats


class LRUCache(BaseCache):
    """Least Recently Used (LRU) cache implementation."""

    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: K) -> Optional[V]:
        if key not in self.cache:
            self.stats.misses += 1
            return None

        self.stats.hits += 1
        # Move to end to mark as most recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: K, value: V) -> None:
        if key in self.cache:
            # Move to end
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove least recently used item
                self.cache.popitem(last=False)
                self.stats.evictions += 1
        self.cache[key] = value

    def clear(self) -> None:
        self.cache.clear()
        self.stats = CacheStats()


class TTLCache(BaseCache):
    """Time-To-Live (TTL) cache implementation."""

    def __init__(self, ttl_seconds: int):
        super().__init__()
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[K, Dict[str, Any]] = {}

    def get(self, key: K) -> Optional[V]:
        if key not in self.cache:
            self.stats.misses += 1
            return None

        entry = self.cache[key]
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            # Entry has expired
            del self.cache[key]
            self.stats.evictions += 1
            self.stats.misses += 1
            return None

        self.stats.hits += 1
        return entry["value"]

    def put(self, key: K, value: V) -> None:
        import time

        self.cache[key] = {"value": value, "timestamp": time.time()}

    def clear(self) -> None:
        self.cache.clear()
        self.stats = CacheStats()


class TwoLevelCache(BaseCache):
    """Two-level cache implementation (e.g., memory + disk)."""

    def __init__(self, l1_cache: BaseCache, l2_cache: BaseCache):
        super().__init__()
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache

    def get(self, key: K) -> Optional[V]:
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            self.stats.hits += 1
            return value

        # Try L2 cache
        value = self.l2_cache.get(key)
        if value is not None:
            # Found in L2, promote to L1
            self.l1_cache.put(key, value)
            self.stats.hits += 1
            return value

        self.stats.misses += 1
        return None

    def put(self, key: K, value: V) -> None:
        # Write-through caching
        self.l1_cache.put(key, value)
        self.l2_cache.put(key, value)

    def clear(self) -> None:
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.stats = CacheStats()
