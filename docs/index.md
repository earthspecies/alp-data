# Getting Started

## What is esp-data?

`esp-data` is an internal Python package that helps with all data-related tasks at ESP. It aims to make working with datasets easier, regardless of where they are stored.

In the future, it will also become the main interface for accessing and exploring datasets commonly used at ESP (this feature is on the roadmap and not yet implemented).

In the first version, the only available module is `esp_data.io`, which makes working with Google Cloud Storage (GCS) and Cloudflare R2 buckets more seamless. It provides utilities for file and bucket level operations. More information is available [here](io.md).

## Installation

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
