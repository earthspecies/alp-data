# `alp_data.backends` Module

## What are data backends?

A `DataBackend` is a Python `Protocol` that defines an interface for any library that is used to read data and perform common operations on data. It's an abstraction that allows `alp-data` to support multiple data libraries without being tightly coupled to any specific one.

An example of such a library is `pandas` and so the corresponding backend class is `alp_data.backends.PandasBackend`. All the `Dataset` and `Transform` classes in `alp-data` up to version 1.3.0 used `pandas` to load the underlying annotation csv / jsonl files. This required implementing functions like `pd.read_csv` and pandas based dataframe manipulation directly within class methods. We want to reduce this dependence on `pandas` and allow ourselves the freedom to use other libraries like `polars`, `duckdb`, `pyarrow`, `webdataset` etc. to load and manipulate data because each library has its own strengths and weaknesses for different ML use-cases.

## Available Backends

Currently, `alp-data` provides two backend implementations:

| Backend | Class | Description |
|---------|-------|-------------|
| `pandas` | `PandasBackend` | Uses pandas DataFrames for data operations. Supports streaming via chunked reading. |
| `polars` | `PolarsBackend` | Uses polars DataFrames/LazyFrames. Supports streaming via LazyFrame for memory-efficient processing. |

## How to Use Backends

### Specifying a Backend in Dataset Configuration

When loading a dataset, you can specify which backend to use via the `backend` parameter:

```python
from alp_data.datasets import BirdSet

# Use polars backend (default)
dataset = BirdSet(split="HSN-train", backend="polars")

# Use pandas backend
dataset = BirdSet(split="HSN-train", backend="pandas")
```

Or via YAML configuration:

```yaml
dataset:
  dataset_name: birdset
  split: HSN-train
  backend: polars  # or "pandas"
  streaming: false
```

### Direct Backend Usage

You can also use backends directly for standalone data operations:

```python
from alp_data.backends import PandasBackend, PolarsBackend

# Load data with PandasBackend
backend = PandasBackend.from_csv("path/to/data.csv")
print(len(backend))  # Number of rows
print(backend.columns)  # Column names

# Load data with PolarsBackend
backend = PolarsBackend.from_parquet("path/to/data.parquet")
row = backend[0]  # Get first row as dict
```

### Streaming Mode

Both backends support streaming mode for memory-efficient processing of large datasets:

```python
from alp_data.datasets import BirdSet

# Enable streaming mode
dataset = BirdSet(split="HSN-train", backend="polars", streaming=True)

# Iterate over rows without loading entire dataset into memory
for sample in dataset:
    process(sample)
```

!!! warning "Streaming Limitations"
    In streaming mode, `__getitem__` indexing is disabled. Use iteration instead. Additionally, `len()` is not available until the stream is consumed.

### Accessing the Underlying Data Object

If you need to perform library-specific operations, use the `unwrap` property to access the underlying data object:

```python
from alp_data.backends import PandasBackend

backend = PandasBackend.from_csv("data.csv")
df = backend.unwrap  # Returns pd.DataFrame
```

For more details, see [Accessing the Underlying Data](#accessing-the-underlying-data).

## Pandas vs Polars: When to Use Which

| Use Case | Recommended Backend | Reason |
|----------|-------------------|--------|
| Small to medium datasets | Either | Both perform well |
| Large datasets (> 1GB) | `polars` | Better memory efficiency and performance |
| Streaming/lazy evaluation | `polars` | Native LazyFrame support |
| Compatibility with existing pandas code | `pandas` | Direct access to `pd.DataFrame` via `unwrap` |
| Parquet file streaming | `polars` | Pandas doesn't support streaming parquet |

## Accessing the Underlying Data

If you need to perform operations not covered by the backend interface, use the `unwrap` property:

```python
from alp_data.backends import PandasBackend, PolarsBackend

# Pandas backend
pandas_backend = PandasBackend.from_csv("data.csv")
df = pandas_backend.unwrap  # Returns pd.DataFrame
# Now use pandas-specific operations
df.describe()

# Polars backend
polars_backend = PolarsBackend.from_csv("data.csv")
df = polars_backend.unwrap  # Returns pl.DataFrame or pl.LazyFrame
# Now use polars-specific operations
df.select(pl.col("species").value_counts())
```

!!! tip
    When using `unwrap`, be aware that you lose the backend abstraction. Operations on the unwrapped object won't automatically work with other backends.

## The DataBackend Protocol

The `DataBackend` protocol defines a common interface that all backend implementations must follow. This enables `alp-data` to work uniformly with different data libraries.

### Core Interface

The protocol defines these key operations:

#### Data Loading (Class Methods)

| Method | Description |
|--------|-------------|
| `from_csv(path, streaming=False)` | Load data from a CSV file |
| `from_json(path, lines=False, streaming=False)` | Load data from a JSON file (supports JSON lines format) |
| `from_parquet(path, streaming=False)` | Load data from a Parquet file |

#### Data Access

| Method | Description |
|--------|-------------|
| `__getitem__(key)` | Get row(s) by index (int returns dict, list/slice returns new backend) |
| `__len__()` | Get number of rows |
| `__iter__()` | Iterate over rows as dictionaries |
| `columns` | Property returning list of column names |
| `column_exists(column)` | Check if a column exists |
| `unwrap` | Property returning the underlying data object (e.g., `pd.DataFrame`, `pl.DataFrame`) |

#### Data Manipulation

| Method | Description |
|--------|-------------|
| `filter_isin(column, values, negate=False)` | Filter rows by column values |
| `drop_duplicates(subset=None, keep="first")` | Remove duplicate rows |
| `dropna(subset=None)` | Remove rows with missing values |
| `get_unique(column)` | Get sorted unique values from a column |
| `map_column(column, mapping, output_column)` | Create new column by mapping values |
| `rename_columns(mapping)` | Rename columns |
| `add_column(column, values)` | Add a new column |
| `select_columns(columns)` | Select subset of columns |
| `concat(backends, ignore_index=True)` | Concatenate multiple backends vertically |

#### Sampling

| Method | Description |
|--------|-------------|
| `sample_rows(n, seed=42, replace=False)` | Randomly sample n rows |
| `subsample_by_column(column, ratios, seed=42)` | Subsample by column values with specified ratios |

#### Advanced Operations

| Method | Description |
|--------|-------------|
| `copy()` | Create a copy of the backend |
| `apply_fn(fn, fn_kwargs, apply_kwargs)` | Apply a custom function to the data |
| `multilabel_from_features(input_features, output_feature, ...)` | Create multilabel column from multiple features |

## Backend Integration in alp-data

### Integration with Datasets

All dataset classes use backends internally to manage their data. The backend is selected at instantiation time:

```python
# alp_data/dataset.py
class Dataset(ABC):
    def __init__(
        self,
        output_take_and_give: dict[str, str] = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        self._backend_class = get_backend(backend)
```

Datasets then use the backend class to load data:

```python
# Example from BirdSet._load()
self._data = self._backend_class.from_json(
    location, lines=True, streaming=self._streaming
)
```

### Integration with Transforms

Transforms operate directly on backend instances rather than raw DataFrames. This makes transforms backend-agnostic:

```python
from alp_data.transforms import Filter
from alp_data.backends import PandasBackend

# Create backend
backend = PandasBackend.from_csv("data.csv")

# Apply transform - works with any backend
filter_transform = Filter(property="species", values=["cat", "dog"], mode="include")
filtered_backend, metadata = filter_transform(backend)
```

Transforms use the backend's methods rather than library-specific operations:

```python
# alp_data/transforms/filter.py
class Filter:
    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        # Uses backend.filter_isin() instead of pandas-specific code
        negate = self.mode == "exclude"
        filtered_backend = backend.filter_isin(self.property, self.values, negate=negate)
        return filtered_backend, {}
```

### Integration with Dataset Concatenation

The `ConcatenatedDataset` class uses backend operations to merge multiple datasets:

```python
from alp_data.datasets import InsectSet459, BirdSet
from alp_data.concat import ConcatenatedDataset

dataset1 = InsectSet459(split="validation", backend="polars")
dataset2 = BirdSet(split="HSN-test", backend="polars")

# All datasets must use the same backend type
concat_ds = ConcatenatedDataset([dataset1, dataset2], merge_level="soft")
```

The concatenation uses the backend's `concat` class method internally.

## API Reference

::: alp_data.backends.protocol.DataBackend
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members_order: source

::: alp_data.backends.PandasBackend
    handler: python
    options:
        show_root_heading: true
        show_source: false

::: alp_data.backends.PolarsBackend
    handler: python
    options:
        show_root_heading: true
        show_source: false

::: alp_data.backends.get_backend
    handler: python
    options:
        show_root_heading: true
        show_source: true