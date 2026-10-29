"""Names of the bookkeeping columns an export adds to its table.

The export index is called `_export_index`, not `_source_index`, because
`ConcatenatedDataset` already emits a `_source_index` column (the row within
the child dataset) that has to survive packing.
"""

SOURCE_INDEX_COL = "_export_index"
SHARD_COL = "_shard"
OFFSET_COL = "_offset"
SIZE_COL = "_size"
SHA256_COL = "_sha256"
AUDIO_FORMAT_COL = "_audio_format"

BOOKKEEPING_COLS = (SOURCE_INDEX_COL, SHARD_COL, OFFSET_COL, SIZE_COL, SHA256_COL, AUDIO_FORMAT_COL)
