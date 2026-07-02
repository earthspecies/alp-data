# Getting Started

## Installation

`alp-data` is available on [PyPI](https://pypi.org/project/alp-data/).

```sh
uv add alp-data
```

or with pip:

```sh
pip install alp-data
```

```python
from alp_data import Beans

# Load 'train' split of BEANS dataset at 16kHz sample rate
# Resampling is done on the fly with librosa.resample
beans = Beans(split="train", sample_rate=16000)

print(len(beans))

# Iterate over all samples
for sample in beans:
    print(sample["audio"].shape)
    break

# Fetch a single sample
sample = beans[0]
print(sample["audio"].shape)

# Streaming only
beans_streaming = Beans(split="train", streaming=True)  # loads faster
print(len(beans_streaming))  # Throws an error since length is unknown in streaming mode

# Iterate over all samples in streaming mode
for sample in beans_streaming:
    print(sample["audio"].shape)
    break
```

!!! warning "PyTorch DataLoader with `num_workers > 0` requires `spawn`"
    When wrapping an alp-data `Dataset` in a PyTorch `DataLoader` with
    `num_workers > 0`, you must use the `"spawn"` multiprocessing start
    method instead of the default `"fork"` on Linux. alp-data datasets hold
    fsspec / `gcsfs` / `s3fs` handles that are not
    safe to inherit across a `fork` — workers may deadlock or return
    corrupted audio. Either call
    `torch.multiprocessing.set_start_method("spawn", force=True)` once at
    program start, or pass `multiprocessing_context=mp.get_context("spawn")`
    to the `DataLoader`.

Check out the datasets documentation for more details [here](./datasets.md).

