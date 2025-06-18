# `esp_data.transforms` module

## What are Transforms?

Transforms are operations that can be applied to an [ESP dataset](datasets.md) to modify, filter, or enhance the data in various ways. In short, Transforms are callable objects that take a pandas DataFrame as input and return a tuple containing:

1. The transformed DataFrame
2. A dictionary of metadata about the transformation

Each transform is defined by two main components:

- A configuration class (inheriting from `pydantic.BaseModel`)
- A transform class that implements the actual transformation logic

## How to Use Transforms

### Basic Usage

Transforms can be used in two ways:

1. Direct instantiation:
```python
from esp_data.transforms import Filter

# Create a filter transform
filter_transform = Filter(
    property="category",
    values=["A", "B"],
    mode="include"
)

# Apply the transform
transformed_data, metadata = filter_transform(data)
```

2. Using configuration:
```python
from esp_data.transforms import FilterConfig, transform_from_config

# Create a configuration
config = FilterConfig(
    type="filter",
    property="category",
    values=["A", "B"],
    mode="include"
)

# Assume a dataframe called 'data' is already defined
transform = transform_from_config(config)
transformed_data, metadata = transform(data)
```

### Transform Configuration

Each transform has its own configuration class that defines its parameters. For example, the `FilterConfig` has:
- `type`: The type of transform ("filter")
- `mode`: Either "include" or "exclude"
- `property`: The property to filter on
- `values`: List of values to filter by

### Creating Custom Transforms

To create a custom transform:

1. Create a configuration class:
```python
from pydantic import BaseModel
from typing import Literal

class MyTransformConfig(BaseModel):
    type: Literal["my_transform"]
    # Add your configuration parameters here
```

2. Create the transform class:
```python
class MyTransform:
    def __init__(self, **kwargs):
        # Initialize your transform
        pass

    @classmethod
    def from_config(cls, cfg: MyTransformConfig) -> "MyTransform":
        return cls(**cfg.model_dump(exclude=("type",)))

    def __call__(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        # Implement your transformation logic
        transformed_data = data  # Your transformation here
        return transformed_data, {}
```

3. Register your transform:
```python
from esp_data.transforms import register_transform

register_transform(MyTransformConfig, MyTransform)
```

## Available Transforms

The transforms system uses a registry pattern to manage available transforms. The registry ensures that each transform type is unique and properly configured before use.
The module provides several built-in transforms to handle common data transformation tasks. Here's an overview of each transform and its functionality:

### Filter Transform
The `Filter` transform allows you to selectively include or exclude rows from your dataset based on specific property values.

::: esp_data.transforms.Filter
    handler: python
    options:
        show_root_heading: true
        show_source: true

### LabelFromFeature Transform
The `LabelFromFeature` transform converts categorical features into numerical labels. Example use case: Converting a 'species' column with values like 'dog', 'cat', 'bird' into numerical labels 0, 1, 2.

::: esp_data.transforms.LabelFromFeature
    handler: python
    options:
        show_root_heading: true
        show_source: true

### MultiLabelFromFeatures Transform
The `MultiLabelFromFeatures` transform extends the functionality of `LabelFromFeature` to handle multiple features simultaneously. Example use case: Creating labels from multiple categorical columns like 'species', 'breed', and 'color' in a single operation.

::: esp_data.transforms.MultiLabelFromFeatures
    handler: python
    options:
        show_root_heading: true
        show_source: true
        allow_missing_labels: true

### Subsample Transform
The `Subsample` transform reduces the size of your dataset by sampling a subset of the data.  Example use case: Creating a 10% random sample of a large dataset for initial testing.

::: esp_data.transforms.Subsample
    handler: python
    options:
        show_root_heading: true
        show_source: true

### UniformSample Transform
The `UniformSample` transform performs uniform sampling of the data, ensuring balanced representation across different categories.

::: esp_data.transforms.UniformSample
    handler: python
    options:
        show_root_heading: true
        show_source: true