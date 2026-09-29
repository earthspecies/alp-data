"""Freeze a configured dataset into tar shards plus a parquet table, and read it back."""

from alp_data.packing.pack import pack
from alp_data.packing.packed_dataset import PackedDataset, PackedDatasetConfig

__all__ = ["pack", "PackedDataset", "PackedDatasetConfig"]
