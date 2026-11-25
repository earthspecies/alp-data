# Using Your Own Dataset

If you’d like to contribute your own dataset, please consider: is this new dataset relatively stable and potentially useful to others? If yes, you **should [submit this form](https://docs.google.com/forms/d/e/1FAIpQLScfkafLGUY-4s8nP2U39oueCAZxkMLKl5spnhqU66CCBGKOgQ/viewform) to ESP's engineering team** to add it as an official ALP Dataset. If not, just follow the next steps!

To create a new dataset, you need to subclass the base `Dataset` class and implement several key components. Here's a step-by-step guide:

## 1. Basic Structure

```python
from alp_data import Dataset, DatasetInfo, register_dataset
from alp_data.io import anypath, AnyPathT
from typing import Any, Dict, Optional
import pandas as pd

@register_dataset
class MyCustomDataset(Dataset):
    """My custom dataset description.

    Parameters
    ----------
    split : str
        The split to load. One of info.split_paths keys.
    output_take_and_give : dict[str, str], optional
        A dictionary mapping the original column names to the new column names.
    data_root : str | AnyPathT, optional
        Custom data root directory.
    """

    # Define dataset metadata
    info = DatasetInfo(
        name="my_custom_dataset",
        owner="your_name",
        split_paths={
            "train": "path/to/train.csv",
            "validation": "path/to/validation.csv",
        },
        version="0.1.0",
        description="Description of your dataset",
        sources=["Source 1", "Source 2"],
        license="Your License",
    )

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: Optional[dict[str, str]] = None,
        data_root: Optional[str | AnyPathT] = None,
    ) -> None:
        """Initialize the dataset."""
        super().__init__(output_take_and_give)
        self.split = split
        self._data = None
        self._load()
        self.data_root = data_root

    def _load(self) -> None:
        """Load the dataset data."""
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. "
                f"Expected one of {list(self.info.split_paths.keys())}"
            )

        # Implement your data loading logic here
        location = self.info.split_paths[self.split]
        # Example: Load CSV data
        self._data = pd.read_csv(anypath(location))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._data is None:
            raise RuntimeError("No split has been loaded yet.")
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific sample from the dataset."""
        if idx < 0 or idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds.")

        # Implement your sample loading logic here
        row = self._data.iloc[idx].to_dict()

        # Example: Load and process data
        if self.data_root:
            data_path = anypath(self.data_root) / row["path"]
        else:
            data_path = anypath(row["path"])

        # Load your data (e.g., image, audio, text)
        data = # your code goes here

        # Apply output_take_and_give if specified
        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
        else:
            item = row

        return item

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> "MyCustomDataset":
        """Create a Dataset instance from a configuration."""
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})

        split = cfg.get("split", None)
        if not split or split not in cls.info.split_paths:
            raise LookupError(
                f"Invalid split '{split}'. "
                f"Available splits: {', '.join(cls.info.split_paths.keys())}"
            )

        return cls(
            split=split,
            output_take_and_give=cfg.get("output_take_and_give", None),
            data_root=cfg.get("data_root"),
        )
```

## 2. Key Components to Implement

1. **DatasetInfo**:
   - `name`: Unique identifier for your dataset
   - `owner`: Dataset maintainer
   - `split_paths`: Dictionary mapping split names to data paths
   - `version`: Dataset version
   - `description`: Brief description
   - `sources`: List of data sources
   - `license`: Dataset license

2. **Required Methods**:
   - `__init__`: Initialize the dataset with split and configuration
   - `_load`: Load the dataset data
   - `__len__`: Return dataset size
   - `__getitem__`: Get a specific sample
   - `from_config`: Create dataset from configuration

3. **Optional Methods**:
   - `__iter__`: Iterate over samples
   - `__str__`: String representation

## 3. Registration

Use the `@register_dataset` decorator to register your dataset:

```python
@register_dataset
class MyCustomDataset(Dataset):
    # Your implementation
    pass
```

## 4. Example Usage

```python
# Create dataset instance
dataset = MyCustomDataset(
    split="train",
    output_take_and_give={"original_col": "new_col"}
)

# Access data
sample = dataset[0]
print(len(dataset))

# Use with transforms
from alp_data.transforms import Filter
filter_transform = Filter(property="category", values=["A", "B"])
dataset.apply_transformations([filter_transform])
```
