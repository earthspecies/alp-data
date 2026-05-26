# Getting Started

## What is esp-data?

`esp-data` is a Python package that helps with all data-related tasks at ESP. It aims to make working with datasets easier, regardless of where they are stored and in which format.

Key features:
- Unified dataset interface: Access datasets stored locally, on cloud storage (e.g., Google Cloud Storage), or in various formats (e.g., CSV, JSON, Parquet) through a consistent API.
- Iterate or random access: Easily iterate over dataset samples or access them randomly using indexing.
- Streaming support: Work with large datasets that don't fit into memory by streaming data on-the-fly.
- On-the-fly data transformations: Apply transformations such as filtering rows / columns, create label
- (beta) Concatenation and merging of multiple datasets: Combine datasets into a single unified dataset for training or evaluation.


## Getting started
```python
from esp_data import Beans

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
    When wrapping an esp-data `Dataset` in a PyTorch `DataLoader` with
    `num_workers > 0`, you must use the `"spawn"` multiprocessing start
    method instead of the default `"fork"` on Linux. esp-data datasets hold
    fsspec / `gcsfs` / `s3fs` handles that are not
    safe to inherit across a `fork` — workers may deadlock or return
    corrupted audio. Either call
    `torch.multiprocessing.set_start_method("spawn", force=True)` once at
    program start, or pass `multiprocessing_context=mp.get_context("spawn")`
    to the `DataLoader`.

Check out the datasets documentation for more details [here](./datasets.md).

## Installation
```sh
git clone https://github.com/earthspecies/esp-data.git
cd esp-data
pip install -e .  # or uv sync
```

### EarthSpeciesProject Internal: Installation via pip / uv

`esp-data` is currently a private package, hosted on ESP's internal Python package repository. Because it isn't available on the public PyPI index, you'll need to configure your project to use ESP's private package index in order to install and update `esp-data`:

### 1. Install `keyring` (one-time setup)

To authenticate and interact with Python repositories hosted on Artifact Registry, you'll need to install the `keyring` library system-wide (not inside a virtual environment), along with the Google Artifact Registry backend. This step is required only once per system, typically when setting up your VM or laptop (not needed on Slurm compute nodes):

```sh
uv tool install keyring --with keyrings.google-artifactregistry-auth
```

!!! success "Slurm"
    This step is **NOT** required for Slurm jobs. All nodes on the cluster already have this package installed.

!!! info
    You only need to do this step once on your system.

!!! tip
    `uv tool` allows you to install Python packages that provide command-line interfaces for system-wide use. The dependencies are installed in an isolated virtual environment, separate from your current project.

### 2. Set up Google Cloud to access `esp-data` package

This step is required if you haven't set up Google Cloud on your device yet. If Google Cloud isn't properly set up the following steps will fail.

- Install the Google Cloud SDK by following the steps on https://cloud.google.com/sdk/docs/install

- Initialize Google Cloud :
    ```sh
    gcloud init
    ```

    You will be prompted to sign in. Type `Y` to open a browser window for authentication and log in with your account.

    Select the project to use; follow the instructions to choose project `okapi-274503`.

    Configure a default Compute region and zone. It is recommended to use the same region and zone as your VM.

    Then run :
    ```sh
    gcloud auth application-default login
    ```
    This will open a browser for authentication again.

- Verify your active account :
    ```sh
    gcloud auth list
    ```

    Example output :
    ```sh
            Credentialed Accounts
    ACTIVE  ACCOUNT
    *       youremailaddress@example.com
    ```

- Confirm your active project :
    ```sh
    gcloud config list
    ```
    Example output :
    ```sh
        [core]
    account = youremailaddress@example.com
    disable_usage_reporting = True
    project = okapi274503
    ```


### 3. Configure your project to use the private index

Next, add the following to your `pyproject.toml` to configure your project to use the private package index:

```toml
[[tool.uv.index]]
name = "esp-pypi"
url = "https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/"
explicit = true

[tool.uv.sources]
esp-data = { index = "esp-pypi" }

[tool.uv]
keyring-provider = "subprocess"
```

### 4. Add `esp-data` as a dependency

You can now add `esp-data` to your project by running:

```sh
uv add esp-data
```

Alternatively, you can manually update the `dependencies` section of your `pyproject.toml` and then run:

```sh
uv sync
```
