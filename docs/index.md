# Getting Started

## What is esp-data?

`esp-data` is an internal Python package to help with all things data related at ESP. It aims to make working with datasets easier, regardless of where they are stored. In the future, it also plans to become the main interface for accessing and exploring datasets commonly used at ESP (this feature is on the roadmap and not yet implemented).

`esp-data.io`: a utility that lets you do file and bucket level operations on google cloud storage (GCS) and Cloudflare R2
buckets. Info [here](io.md)

## Installation

`esp-data` is currently a private package, hosted on ESP's internal Python package repository. Because it isn't available on the public PyPI index, you'll need to configure your project to use ESP's private package index in order to install and update `esp-data`:

### 1. Install `keyring` (One-Time Setup)

To authenticate and interact with Python repositories hosted on Artifact Registry, you'll need to install the `keyring` library system-wide (not inside a virtual environment), along with the Google Artifact Registry backend. This step is required only once per system, typically when setting up your VM or laptop (not needed on Slurm compute nodes):

```sh
uv tool install keyring --with keyrings.google-artifactregistry-auth
```

!!! info
    You only need to perform this installation once on your system.


!!! tip
    `uv tool` allows you to install Python packages that provide command-line interfaces for system-wide use. The dependencies are installed in an isolated virtual environment, separate from your current project.

### 2. Configure your project to use the private index

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

### 3. Add `esp-data` as a dependency

You can now add `esp-data` to your project by running:

```sh
uv add esp-data
```

Alternatively, you can manually update the `dependencies` section of your `pyproject.toml` and then run:

```sh
uv sync
```
