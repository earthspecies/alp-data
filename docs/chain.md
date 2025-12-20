# `esp_data.chain` Module

## What is Dataset chaining?

The `chain` module provides a lightweight way to **iterate over multiple ESP datasets** as if they were a single dataset. Unlike `ConcatenatedDataset`, chaining does not merge DataFrames or support transformations on the combined data—it simply yields items from each source dataset in sequence.

Use `ChainedDataset` when you:

- Need to iterate over multiple datasets without applying joint transformations
- Want to support streaming mode across multiple datasets
- Prefer a lightweight approach that doesn't create a merged DataFrame

For combining datasets with transformation support, see [concatenate.md](concatenate.md).

## How can I chain datasets?

### Basic Usage

```python
from esp_data.datasets import InsectSet459, BirdSet
from esp_data.chain import ChainedDataset

# Load individual datasets
dataset1 = InsectSet459(split="validation")
dataset2 = BirdSet(split="HSN-test")

print(f"Dataset 1 length: {len(dataset1)}")
print(f"Dataset 2 length: {len(dataset2)}")

# Chain datasets for iteration
chained = ChainedDataset([dataset1, dataset2])

# Length is the sum of all source datasets
print(f"Chained dataset length: {len(chained)}")

# Iterate over all items
for item in chained:
    print(item.keys())
    break  # Just show first item
```

### Indexing

`ChainedDataset` supports indexing by mapping the global index to the appropriate source dataset:

```python
# Access items by global index
first_item = chained[0]  # From dataset1
last_item = chained[-1]  # Not supported - raises IndexError

# Index maps across datasets
# If dataset1 has 100 items and dataset2 has 200 items:
# - chained[0] returns dataset1[0]
# - chained[99] returns dataset1[99]
# - chained[100] returns dataset2[0]
# - chained[299] returns dataset2[199]
```

!!! warning
    Negative indexing is not supported in `ChainedDataset`. Attempting to use negative indices will raise an `IndexError`.

### Streaming Mode

Unlike `ConcatenatedDataset`, `ChainedDataset` supports streaming mode. All source datasets must have the same streaming mode:

```python
# Streaming mode - all datasets must be streaming
streaming_ds1 = SomeDataset(split="train", streaming=True)
streaming_ds2 = AnotherDataset(split="train", streaming=True)

chained_streaming = ChainedDataset([streaming_ds1, streaming_ds2])

# In streaming mode, len() raises RuntimeError
# Iterate instead:
for item in chained_streaming:
    process(item)
```

### Creating from Configuration

You can create a `ChainedDataset` from a YAML configuration file using the `chain` keyword:

```yaml
chain:
  datasets:
    - dataset_name: insectset459
      split: validation
    - dataset_name: birdset
      split: HSN-test
```

Load the configuration in Python:

```python
from esp_data import dataset_from_config

chained_dataset, metadata = dataset_from_config("path/to/chain_config.yaml")
```

## Key Differences from ConcatenatedDataset

| Aspect | ChainedDataset | ConcatenatedDataset |
|--------|----------------|---------------------|
| **DataFrame handling** | Delegates to source datasets | Merges into single DataFrame |
| **Transformations** | Not supported | Supported via `apply_transformations` |
| **Column handling** | Union of all columns reported | Merge strategies (hard/overlap/soft) |
| **Streaming** | Supported | Not supported |
| **Memory footprint** | Lightweight | Holds merged DataFrame |
| **Metadata merging** | Basic | Full merge (names, owners, versions, etc.) |

## Understanding the ChainedDataset Class

### Key Properties

```python
# Available columns (union of all source dataset columns)
print(chained.columns)

# Available splits (always ["chained"])
print(chained.available_splits)

# Length (sum of source dataset lengths)
print(len(chained))  # Raises RuntimeError in streaming mode
```

### Iteration Behavior

When iterating, items are yielded from each source dataset in order:

```python
# Items come from datasets in order
chained = ChainedDataset([dataset1, dataset2, dataset3])

# Iteration yields:
# - All items from dataset1
# - Then all items from dataset2
# - Then all items from dataset3
for item in chained:
    # item comes from whichever dataset it belongs to
    pass
```

## Limitations

1. **No transformations**: Cannot apply transforms to the chained dataset as a whole
2. **No negative indexing**: Only non-negative integer indices are supported
3. **Streaming mode consistency**: All source datasets must have the same streaming mode
4. **No column merging**: Columns are not aligned or merged; each item has whatever columns its source dataset provides

## Function Reference

::: esp_data.chain.ChainedDataset
    handler: python
    options:
        show_source: true

::: esp_data.chain.ChainException
    handler: python
    options:
        show_source: true
