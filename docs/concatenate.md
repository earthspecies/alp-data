# `esp_data.concat` Module

## What is Dataset concatenation?

The `concat` module provides utilities for **combining multiple ESP datasets** into a single unified dataset. This is particularly useful when you want to train models on data from multiple sources or combine different splits of related datasets while maintaining proper data handling and metadata.

More technically, dataset concatenation:

- Combines multiple `Dataset` objects into a single `ConcatenatedDataset`
- Preserves original dataset functionality through source dataset references
- Handles column mismatches through configurable merge strategies
- Maintains proper metadata and configuration merging
- Tracks data provenance for debugging and analysis through a merged `DatasetInfo`

### A note on dataset merging

In v1 of this approach, we have are three [merge strategies](#merge-strategies). Currently, because we use `pandas` DataFrames under the hood, the merge strategies are:

1. **Soft Merge**: Keeps all columns from all datasets, filling missing values with NaN.
2. **Overlap Merge**: Keeps only columns that exist in all datasets.
3. **Hard Merge**: Requires all datasets to have identical columns, raising an error if they differ.

!!! warning
    When merging two datasets, be aware that if a column with the same name appears in multiple datasets being concatenated, and the data type **dtype** of the column is not the same across datasets, the resulting column in the concatenated dataset will be of type `object` in pandas, which may lead to unexpected behavior in downstream processing because this can be a mixed data type!

## How can I concatenate datasets?

Datasets can be concatenated using the `ConcatenatedDataset` class with different merge strategies:

### Basic Usage

```python
from esp_data.datasets import AnimalSpeak, BarkleyCanyon
from esp_data.concat import ConcatanatedDataset

# Load individual datasets
dataset1 = AnimalSpeak(split="validation")
dataset2 = BarkleyCanyon(split="train")

print(f"Dataset 1 length: {len(dataset1)}")
# 2033
print(f"Dataset 2 length: {len(dataset2)}")
# 9770

# Concatenate with default soft merge
combined_dataset = ConcatenatedDataset(
    datasets=[dataset1, dataset2],
    merge_level="soft"  # Options: "soft", "overlap", "hard"
)

# Access the combined data
print(f"Combined dataset length: {len(combined_dataset)}")
# 11803
sample = combined_dataset[0]  # Get first sample (should be AnimalSpeak)
print(f"First sample: {sample.keys()}")

# get last sample (should be BarkleyCanyon)
print(f"Last sample: {combined_dataset[-1].keys()}")
```

#### Create a combined dataset from a yaml config
You can also create a concatenated dataset from a YAML configuration file. Here is an example of how to do this:
Note the `concat` keyword at the top level of the config, and `datasets` list inside it (required).

```yaml
concat:
  datasets:
    - dataset_name: beans
      split: dogs_test
      output_take_and_give: null
    - dataset_name: beans
      split: esc50_validation
      output_take_and_give: null
  merge_level: soft
  transformations:
    - type: label_from_feature
      feature: label
      output_feature: label
      override: true
    - type: deduplicate
      subset: ["file_name", "label"]
      keep_first: true
```
Here, we're concatenating two splits of the `beans` dataset and applying some transformations to the combined dataset.
The `concat` keyword is a *SPECIAL* keyword, which tells the `dataset_from_config` function to create a `ConcatenatedDataset` instead of a regular dataset. Here's the python code for loading this config:

```python
from esp_data import dataset_from_config
combined_dataset = dataset_from_config("path/to/concat_config.yaml")
```

#### Apply transformations before / after concatenation

You can apply transformations to the individual datasets *before* concatenation as shown in [transforms.md](transforms.md). This allows you to treat the data as needed, but you can also apply transformations *after* concatenation if you want to operate on the combined dataset as a whole. Here is an example of applying a filter transformation *after* concatenation:

```python
from esp_data.datasets import AnimalSpeak, BarkleyCanyon
from esp_data.transforms import FilterConfig
from esp_data.concat import ConcatenatedDataset

# Load individual datasets
dataset1 = AnimalSpeak(split="validation")
dataset2 = BarkleyCanyon(split="train")
# Concatenate datasets
combined_dataset = ConcatenatedDataset([dataset1, dataset2])
# Define a filter transformation
filter_config = FilterConfig(
    type="filter",
    property="species_common",
    values=["American Robin", "Bottle-nosed Dolphin"],
    mode="include"
)

# Run the transformation on the combined dataset
transform_metadata = combined_dataset.apply_transformations([filter_config])
```
!!! warning
    If the `merge_level` was set to "soft" in `ConcatenatedDataset`, running a filter transformation like this
    will end up dropping all rows from datasets that do not have the `species_common` column, since those rows will be
    `NaN` for those datasets.


As mentioned, ycou can also apply transforms to individual datasets before concatenation:

```python
# Create and transform individual datasets
animal_dataset = AnimalSpeak(split="train")
animal_filter = FilterConfig(
    type="filter",
    property="source",
    values=["xeno-canto"],
    mode="include"
)
animal_dataset.apply_transformations([animal_filter])

insect_dataset = InsectSet459(split="train")
insect_filter = FilterConfig(
    type="filter",
    property="family",
    values=["Cicadidae", "Gryllidae"],
    mode="include"
)
insect_dataset.apply_transformations([insect_filter])

# Concatenate the transformed datasets
combined_dataset = ConcatenatedDataset(
    [animal_dataset, insect_dataset],
    merge_level="overlap"
)
```


### Merge Strategies

The `merge_level` parameter controls how datasets with different columns are handled:

#### 1. Soft Merge (Default)
Keeps all columns from all datasets, filling missing values with NaN:

```python
# Soft merge - most permissive
combined_dataset = ConcatenatedDataset(
    [dataset1, dataset2],
    merge_level="soft"
)
```

#### 2. Overlap Merge
Keeps only columns that exist in all datasets:

```python
# Overlap merge - keeps common columns only
combined_dataset = ConcatenatedDataset(
    [dataset1, dataset2],
    merge_level="overlap"
)
```

#### 3. Hard Merge
Requires all datasets to have identical columns:

```python
# Hard merge - strictest option
combined_dataset = ConcatenatedDataset(
    [dataset1, dataset2],
    merge_level="hard"
)
```

## Understanding the ConcatenatedDataset Class

The `ConcatenatedDataset` class is the result of dataset concatenation and provides several important features:

### Key Properties

```python
# Access dataset information
print(combined_dataset.info.name)  # Combined dataset name
print(combined_dataset.info.description)  # Merged description
print(combined_dataset.columns)  # Available columns (excludes internal tracking)
print(combined_dataset.available_splits)  # Always ["concatenated"]

# Sample rate handling
print(combined_dataset.sample_rate)  # Unified sample rate if compatible
```

### Data Access

The concatenated dataset maintains full functionality of individual datasets:

```python
# Standard dataset operations
for i, sample in enumerate(combined_dataset):
    if i >= 5:  # Just show first 5
        break
    print(f"Sample {i}: {sample.keys()}")

# Direct indexing
specific_sample = combined_dataset[42]
```

### Source Dataset Tracking

Each sample maintains information about its original source:

```python
# The internal tracking is handled automatically
# You get the properly loaded data from the original source dataset
sample = combined_dataset[0]
# This sample was loaded using the appropriate source dataset's __getitem__ method
```

## Configuration and Metadata Merging

### DatasetInfo Merging

When datasets are concatenated, their metadata is merged like so:

- **Names**: Combined with "+" separator (e.g., "animalspeak+barkleycanyon")
- **Owners**: Deduplicated and joined with ";" separator
- **Versions**: Highest version is selected using semantic versioning
- **Descriptions**: Numbered list of original descriptions
- **Sources**: Deduplicated list of all sources
- **Licenses**: Unique licenses joined with ";" separator

### Sample Rate Validation

Sample rates must be compatible across datasets:

```python
# This will work if both datasets have the same sample rate
combined_dataset = ConcatenatedDataset([dataset1, dataset2])

# This will raise MergeException if sample rates differ
try:
    incompatible_dataset = ConcatenatedDataset([audio_16k, audio_44k])
except MergeException as e:
    print(f"Sample rate mismatch: {e}")
```

### Output Column Mapping

The `output_take_and_give` mappings are merged and validated:

```python
from esp_data.datasets import AnimalSpeak

# Create datasets with compatible column mappings
# i.e., either completely different mappings valid to each dataset individually,
# or overlapping, but not conflicting mappings
dataset1 = AnimalSpeak(
    split="validation",
    output_take_and_give={"canonical_name": "species"}
)
dataset2 = AnimalSpeak(
    split="train",
    output_take_and_give={"local_path": "path"}
)

# These will be merged successfully
combined_dataset = ConcatenatedDataset([dataset1, dataset2])
# Access the merged output mappings
print(combined_dataset.output_take_and_give)
# Output: {'canonical_name': 'species', 'local_path': 'path'}

# Conflicting mappings will raise MergeException
dataset3 = AnimalSpeak(
    split="validation",
    output_take_and_give={"canonical_name": "different_name"}  # Conflict!
)

try:
    bad_combined = ConcatenatedDataset([dataset1, dataset3])
except MergeException as e:
    print(f"Mapping conflict: {e}")
```

## Best Practices

### 1. Choose the Right Merge Strategy

- Use **soft merge** when datasets have different but complementary columns
- Use **overlap merge** when you only need common features across datasets
- Use **hard merge** when datasets should have identical schemas

### 2. Validate Before Concatenation
It might make sense to a perform a sanity check that multiple datasets *can* be
concatenated without issues. Incompatible datasets (for e.g. merge strategy "hard"
but different columns, or different sample rates) will raise a `MergeException`
when you try to concatenate them.

For example, this can be achieved using a `check_compatibility` function:

```python
# Check dataset compatibility
def check_compatibility(datasets):
    sample_rates = [getattr(ds, 'sample_rate', None) for ds in datasets]
    if len(set(sr for sr in sample_rates if sr is not None)) > 1:
        print("Warning: Different sample rates detected")

    columns = [set(ds._data.columns) for ds in datasets]
    common_cols = set.intersection(*columns)
    print(f"Common columns: {len(common_cols)}")

check_compatibility([dataset1, dataset2])
```

## Limitations and Considerations

### Current Limitations

1. **No Configuration Loading**: `ConcatenatedDataset` cannot be created from `DatasetConfig`
2. **Memory Usage**: All source datasets remain in memory
3. **Single Split**: Concatenated datasets only support the "concatenated" split

### Performance Considerations

- Concatenation creates a new DataFrame, which uses additional memory
- Source dataset references are maintained, so original datasets aren't garbage collected
- Index lookups require mapping back to source datasets

## Function Reference

::: esp_data.concat.ConcatenatedDataset
    handler: python
    options:
        show_source: true

::: esp_data.concat.MergeException
    handler: python
    options:
        show_source: true