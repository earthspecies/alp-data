# Getting Started


## What is esp-data?


## Installation

`esp-data` is currently not public and therefore hosted on ESP's private Python package repository.

Install `keyring` library system-wide (i.e. not inside a virtual environment) with the additional Google Artifact Registry backend. needed only once and is needed to to authenticate and interact with Python repositories hosted on Artifact Registry:

```
uv tool install keyring --with keyrings.google-artifactregistry-auth
```

This step is not needed for running code on Slurm compute nodes but you need this if you're setting up your VM or laptop. 

!!! info
    You only need to this once on your system.

!!! tip
    `uv tool` allows installing Python packages that provide command-line interfaces for system-wide use. The dependencies of the package are installed in a temporary virtual environment isolated from the current project.


You then need to tell your project about the index by adding the following section to your `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "esp-pypi"
url = "https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/"
explicit = true

[tool.uv.sources]
esp-data = { index = "esp-pypi" }

[tool.uv]
keyring-provider = "subprocess"
```

You can now add `esp-data` to your project by doing `uv add esp-data` or manually updating the the `dependencies` section of your `pyproject.toml` and issuing a `uv sync`.
