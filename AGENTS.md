# Code style
- Use NumPy style for docstrings. If any exceptions are raised make sure that they're documented in the "Raises" section.
- When you make changes to the logic of a function or class make sure its docstring is still valid. If required, update the docstrong to match the changes.
- Run `ruff check --fix` to make sure your code is formatter correctly.
- Use single backticks for inline code references in docstrings
- Don't use conditional `if TYPE_CHECKING`
- Always add a `__all__` in `__init__.py` with strings of units that are exported
- Make sure that only objects that are needed outside a module are exposed in the `__init__.py`. Tests don't count as external usage; they should import directly from the submodule if something isn't in `__init__.py`.

# Key Directories

## `alp_data` - main python library

Top-level modules:

- `dataset.py` - the `Dataset` abstract base class plus the pydantic config models
  (`DatasetConfig`, `ConcatConfig`, `ChainedDatasetConfig`), `DatasetInfo`, and the dataset
  registry (`register_dataset`, `register_config`, `dataset_from_config`,
  `dataset_class_from_name`, `list_registered_datasets`). Start here to understand the
  dataset contract.
- `concat.py` - `ConcatenatedDataset`, which stacks the rows of several datasets into one
  and reconciles their differing columns via a `hard`/`overlap`/`soft` merge level.
- `chain.py` - `ChainedDataset`, which iterates and indexes several datasets as if they
  were one.
- `utils.py` - general-purpose helpers used across the library (hashing, UUIDs, time,
  GCP secret access).

Subpackages:

- `io` - path handling and file/audio I/O.
    - `paths.py` - pure path classes for cloud URIs (`PureGSPath`, `PureS3Path`,
      `PureR2Path`), `anypath`, `AnyPathT`, and `DATA_HOME`.
    - `filesystem.py` - `fsspec` filesystem construction for local and cloud paths.
    - `file_utils.py` - path-level operations (`exists`, `rm`, `read_json`, `read_yaml`,
      `read_text`).
    - `read_utils.py` - audio reading and inspection (`read_audio`, `get_audio_info`,
      `audio_stereo_to_mono`), including the ffmpeg range-read path.
    - `auth.py` - downscoped GCS access tokens for authenticating HTTP/REST reads.
- `datasets` - one module per officially supported dataset (`beans.py`, `birdset.py`,
  `xeno_canto.py`, ...), each defining a `Dataset` subclass and its config, registered via
  `register_dataset`. Adding a dataset means adding a module here and exporting it from
  `datasets/__init__.py`.
- `transforms` - official transforms, one per module (`filter.py`, `subsample.py`,
  `balanced_sample.py`, `deduplicate.py`, `long_tail_upsample.py`, `select_columns.py`,
  `label_from_feature.py`, `multilabel_from_features.py`). Each module defines a transform
  and its pydantic config. `registry.py` owns `register_transform`,
  `transform_from_config`, and the `RegisteredTransformConfigs` union, and must be imported
  first because the transform modules depend on it.
- `backends` - the dataframe abstraction layer. `protocol.py` defines the `DataBackend` and
  `StreamingDataBackend` protocols every backend must implement, `pandas_backend.py` and
  `polars_backend.py` are the implementations, and `backends.py` is the registry
  (`BackendType`, `get_backend`).
- `discover` - dataset discovery and taxonomy enrichment. `gbif_taxonomy.py` provides
  `GBIFConverter` and the `AddTaxonomy` transform, which resolve species synonyms against
  the GBIF backbone taxonomy.

## Other directories

- `tests` - unit tests for the library.
    - `tests/*.py` - tests for the core modules and transforms.
    - `tests/io` - tests for the path classes.
    - `tests/unittests/datasets_tests` - one test module per dataset.
    - `tests/samples` - fixture audio, text, and YAML configs.
- `docs` - the mkdocs site (`mkdocs.yml` at the repo root). Guide pages live at the top
  level (`getting-started.md`, `datasets.md`, `io.md`, `transforms.md`, `backends.md`,
  ...), executable tutorials in `docs/tutorials`, and static files in `docs/assets` and
  `docs/stylesheets`. New pages must be added to the `nav` in `mkdocs.yml`.
- `_hooks` - mkdocs build hooks registered in `mkdocs.yml`. `dataset_info_hook.py` injects
  each dataset's `DatasetInfo` metadata into the rendered docs pages, and
  `hide_toc_hook.py` hides the table of contents on the notebook tutorial pages.
- `scripts` - one-off and operational scripts, not part of the shipped library.
    - `scripts/benchmarks` - loading-time and latency benchmarks with their configs.
    - `scripts/data_preprocessing_scripts` - per-dataset preparation and export scripts.
    - `scripts/dataset_prep_notebooks` - exploratory notebooks used while preparing datasets.
    - `scripts/configs` - YAML configs for the resampling scripts.
- `jobs` - shell entrypoints that run the benchmarks in `scripts/benchmarks`, including
  array jobs.
- `notebooks` - scratch and exploratory notebooks.

# Environment Setup
- Make sure you use `uv run` for running python commands
