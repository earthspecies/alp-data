"""Freeze a configured dataset into tar shards plus a parquet table."""

from alp_data.packing.pack import pack

__all__ = ["pack"]
