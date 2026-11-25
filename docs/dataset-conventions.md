# Dataset Conventions

This page sets out what an official ALP audio dataset is expected to look like. It covers three types of dataset:

| Type | One row is | Output | Required metadata | Reference implementation |
|------|------------|--------|-------------------|--------------------------|
| [Classification](#classification-datasets) | One audio clip (or one segment of a recording) | `audio`, `sample_rate` and label columns | At least one label column; taxonomy columns when the source gives species | [`Watkins`][watkins] |
| [Detection](#detection-datasets) | One recording plus all its annotated events | `audio`, `sample_rate`, `selection_table` | Selection table with event times and labels | [`SapsuckerWoods`][sapsucker] |
| [Soundscape](#soundscape-datasets) | One long, unlabeled recording (minutes to hours) | `audio`, `sample_rate` | None | None currently, but see [`FASD13`][fasd13] for windowed reads |

Read the [common requirements](#common-requirements) first, then the section for your dataset's type. If you are adding a dataset for your own use only, see [Using Your Own Dataset](datasets-custom.md); these conventions are still a good guide.

*Note*
    Many existing datasets predate this page and don't follow all of it. The reference implementations are the closest matches, and each section lists where they still differ from the spec. New datasets should follow the spec, not copy those differences.

## Common requirements

Everything in this section applies to all three types.

### File layout and registration

- Put the dataset in its own module, `alp_data/datasets/<dataset_name>.py`, with one `Dataset` subclass named in CamelCase (e.g. `SapsuckerWoods` in `sapsucker_woods.py`).
- Decorate the class with `@register_dataset` so it can be loaded by name from a config.
- Import the class in `alp_data/datasets/__init__.py` **and** add it to `__all__` there. The [Available Datasets](datasets-available.md) page is generated from `alp_data.datasets`, so a dataset missing from `__all__` won't appear in the documentation. Similarly, add it to `alp_data/__init__.py`.
- Store the split CSVs and audio under `{DATA_HOME}/<dataset_name>/v<version>/`, and define the root as a module-level constant (e.g. `_RAW_ROOT`).

### `DatasetInfo`

Fill in every field with real values. Placeholders such as `description="[MISSING]"` are not accepted.

| Field | Type | Expectation |
|-------|------|-------------|
| `name` | `str` | snake_case, the same as the module name (e.g. `"sapsucker_woods"`) |
| `owner` | `str` | ESP team member(s) responsible for the dataset, separated by `;` (e.g. `"david; benjamin"`) |
| `split_paths` | `dict[str, str]` | Maps each split name to the path of its CSV, e.g. `{"all": f"{_RAW_ROOT}/all.csv"}`. Paths may be local or cloud (`gs://...`). See [Split CSVs](#split-csvs) for the file format |
| `version` | `str` | `MAJOR.MINOR.PATCH` (e.g. `"0.1.0"`), matching the `v<version>` folder in storage |
| `description` | `str` | One or two sentences with the essential numbers: how many recordings or clips, how long, how many classes, where from |
| `sources` | `str` or `list[str]` | Where the data comes from (e.g. `"Zenodo"`, or a DOI/URL) |
| `license` | `str` | An [SPDX identifier](https://spdx.org/licenses/) such as `"CC-BY-4.0"`. If licenses differ between rows, say so here and add a per-row `license` column |
| `changelog` | `str` | *Optional.* Changes from the previous version |

### Split CSVs

Each split is one CSV file:

- Comma-separated, UTF-8, with a header row of column names.
- One row per item: a clip for classification, a recording for detection and soundscape datasets.
- Columns follow the [metadata columns](#metadata-columns) rules and the tables for each type. An empty field means a missing value.
- Detection datasets store each recording's selection table in the `selection_table` column, as a tab-separated string (see [Detection datasets](#detection-datasets)).

### Constructor and methods

Every dataset takes the same constructor arguments, so it can be built from a [`DatasetConfig`](datasets.md#dataset-configuration):

```python
def __init__(
    self,
    split: str = "all",
    output_take_and_give: dict[str, str] | None = None,
    sample_rate: int | None = None,
    data_root: str | AnyPathT | None = None,
    backend: BackendType = "polars",
    streaming: bool = False,
) -> None:
```

Add extra arguments only when the dataset needs them, and register a matching `DatasetConfig` subclass with `@register_config` (see `INaturalist` for an example).

Implement the following methods and properties:

| Method / property | Behavior | Raises |
|-------------------|----------|--------|
| `__init__` | Stores the arguments, calls `_load()`, and sets `data_root` to the parent folder of the split CSV when none is given | — |
| `_load` | Reads the split CSV with `self._backend_class.from_csv(location, streaming=self._streaming)` | `LookupError` if the split is not in `info.split_paths` |
| `columns` | Column names of the loaded split | — |
| `available_splits` | `list(self.info.split_paths.keys())` | — |
| `available_sample_rates` | Pre-resampled rates present in the data (only if the dataset ships [pre-resampled audio](#audio-output)) | — |
| `__len__` | Number of rows in the split | `RuntimeError` if no split is loaded; `NotImplementedError` in streaming mode |
| `_process` | Turns one row into an output item: reads audio, resamples, parses the selection table (detection), applies `output_take_and_give` | — |
| `__getitem__` / `__iter__` | Call `_process` on one row / on each row | — |
| `from_config` | Builds the dataset from a `DatasetConfig`, applies any transformations, and returns `(dataset, metadata)` (metadata is `{}` when there are no transformations) | — |
| `__str__` | Name, version, sources, license and available splits | — |
| `get_available_labels` | **Classification and detection only.** See [Labels](#labels) | — |

Every method that raises must list the exceptions in a NumPy-style `Raises` section.

### Audio output

- `audio` is a 1-D `np.float32` array. Convert multi-channel audio to mono with `audio_stereo_to_mono(audio, mono_method="average")`. This guide does not address audio that is required to be multi-channel.
- `sample_rate` is an `int`: the rate the returned audio is actually at.
- When `sample_rate` is `None`, return audio at its native rate.
- When `sample_rate` is set and differs from the file's rate, resample with `librosa.resample(..., res_type="kaiser_best")`.
- State the native sample rate(s) and the default `sample_rate` in the docstring.

**Pre-resampled audio.** For large datasets, unless there is a good reason not to, store mono copies of each file at 16 kHz and 32 kHz next to the originals, and put their paths in the `16khz_path` and `32khz_path` columns. Declare which column holds which rate in a class attribute:

```python
_sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
```

In `_process`, if the requested `sample_rate` has a pre-resampled copy, read that copy directly. Otherwise, read the original file and resample it on the fly.

If the source audio is natively below 16 kHz or 32 kHz, say in the docstring that the copies are upsampled and carry no extra bandwidth.

**Windowed reads.** When a row carries `window_start_sec` and `window_end_sec`, read only that segment with `read_audio(path, start_time=..., end_time=...)`. This streams the segment from cloud storage instead of downloading the whole file. It is required for clips that are segments of longer files, and expected for detection and soundscape recordings longer than a few minutes.

*Warning*
    Resampling an hour-long recording with `kaiser_best` takes minutes. For long recordings, either ship pre-resampled audio or keep the default `sample_rate` at the native rate.

### Metadata columns

Each type below lists **required** and **optional** columns with their dtypes. The rules for all of them:

- Column names are snake_case. The only exceptions are the selection-table columns that follow Raven's format, such as `Begin Time (s)`.
- Required columns must be present in every split and have no missing values.
- Optional columns use exactly the listed name and dtype when the concept is present. Don't rename them.
- Additional columns are allowed, as long as they don't duplicate a concept already covered by a required or optional column. For example, don't add `lat` next to `latitude`, `duration` next to `duration_s`, or a `species` column holding the same GBIF names as `species_gbif_2023`.
- Name label columns after what they hold (`call_type`, `individual_id`), not `label`. Users choose the training target with `output_take_and_give` or the [`label_from_feature`](transforms.md) transform.
- When one row has several values for a column, separate them with `;` (e.g. `"click;whistle"`).

Dtypes are given as `str`, `int` or `float`. Paths are `str` and relative to `data_root`.

**Optional columns shared by all types:**

| Column | Dtype | Meaning |
|--------|-------|---------|
| `16khz_path`, `32khz_path` | `str` | Pre-resampled mono audio, relative to `data_root` |
| `duration_s` | `float` | Duration of the audio file, in seconds |
| `window_start_sec`, `window_end_sec` | `float` | Segment of `audio_path` to read, in seconds from the start of the file |
| `latitude`, `longitude` | `float` | Recording location, WGS84 decimal degrees |
| `event_date` | `str` | Recording date, ISO 8601 (`YYYY-MM-DD`) |
| `event_time` | `str` | Recording start time (`HH:MM:SS`); give the time zone in the docstring |
| `site` | `str` | Name or code of the recording site |
| `recorder_id` | `str` | Identifier of the recording device |
| `recordist` | `str` | Person or organization that made the recording |
| `license` | `str` | Per-row license, when licenses differ within the dataset |
| `citation` | `str` | Per-row citation, when sources require different citations |

### Taxonomy

Species names are converted to the GBIF backbone taxonomy, so datasets can be combined and filtered by taxon. Use `alp_data.discover.GBIFConverter` to resolve names. Wherever species names are converted, keep the name the source used too. GBIF-derived columns end in `_gbif_2023`, naming the GBIF backbone release they come from.

| Column | Dtype | Meaning |
|--------|-------|---------|
| `species_gbif_2023` | `str` | Accepted GBIF scientific name (e.g. `"Orcinus orca"`) |
| `species_original` | `str` | The species name exactly as the source gives it, before conversion (a scientific name, common name or code such as an eBird code) |
| `species_common_gbif_2023` | `str` | *Optional.* Common English name |
| `genus_gbif_2023`, `family_gbif_2023`, `order_gbif_2023`, `class_gbif_2023`, `phylum_gbif_2023`, `kingdom_gbif_2023` | `str` | *Optional.* Taxonomic hierarchy from GBIF |

- If a name doesn't resolve, add a manual correction to the converter instead of leaving the row unresolved. If it can't be fixed, drop the row and record how many were dropped in the docstring's `Remarks` section.

*Note*
    These column names are new. Existing datasets use `canonical_name`, `species_common` and unsuffixed higher-taxon columns (`genus`, `family`, ...), and none of them keep the source's name in `species_original`. New datasets should use the names above.

### Labels

Classification and detection datasets list their label columns and implement `get_available_labels`. Soundscape datasets have no labels and implement neither.

- `self.annotation_columns` (`list[str]`): the label columns. For classification these are row columns (e.g. `["species_gbif_2023", "call_type"]`); for detection they are selection-table columns.
- `get_available_labels(anno_column)` (returns `list[str]`): the sorted unique values of `anno_column` across the whole split, taken from the rows for classification and from every selection table for detection. Values separated by `;` count as separate labels. `anno_column` defaults to the first entry of `annotation_columns`.
- Label sounds that annotators could not identify as `"Unknown"`.

Many existing classification datasets don't implement `get_available_labels` yet; new ones must.

### Class docstring

The class docstring is what users read on the [Available Datasets](datasets-available.md) page, so it must stand alone. Use NumPy style, with these sections in this order (`INaturalist` is a good model):

1. A one-line summary.
2. **Description**: a short paragraph on what the data is, where it was recorded, by whom, and how much there is. Quoting the source's description is fine, but add what is specific to this version (re-encoding, label mapping, dropped rows).
3. **Available Metadata Fields**: the columns, grouped by kind with bold sub-headings (`**Taxonomic Information:**`, `**Audio File Paths:**`, `**Location:**`, ...). Describe at least every required column and every column users are expected to train or filter on. Give the fill rate of columns that are often empty (e.g. "populated for ~84% of rows").
4. **Labels**: what the label columns contain, how labels were mapped, and how unknown or ambiguous events are handled.
5. **Available Splits**: every split, and how it was made (random seed, held-out taxa, official source split).
6. **Pre-resampled Audio** (if applicable): which rates exist and when they are used.
7. **Remarks** (if applicable): caveats such as lossy formats, dropped rows or upsampled audio.
8. **References**: links to the paper, data repository and project page.
9. **Examples**: doctest examples that load a split and show the main options.

A skeleton:

```python
@register_dataset
class MyDataset(Dataset):
    """My Dataset: hour-long soundscapes from <place>, annotated for <N> bird species.

    Description
    -----------
    <What the data is, who recorded it, where and when, and how much there is.>

    Available Metadata Fields
    -------------------------
    **Taxonomic Information:**
        - ``species_gbif_2023``: Accepted GBIF scientific name
        - ``species_original``: Species name as given by the source

    **Audio File Paths:**
        - ``audio_path``: Original audio relative to data_root (48 kHz FLAC)
        - ``16khz_path``, ``32khz_path``: Pre-resampled mono WAV

    **Recording Metadata:**
        - ``duration_s``: Recording duration in seconds
        - ``event_date``: Recording date (~90%)

    Available Splits
    ----------------
    - ``all``: All recordings

    References
    ----------
    [Zenodo](https://zenodo.org/records/<id>)

    Examples
    --------
    >>> from alp_data.datasets import MyDataset
    >>> dataset = MyDataset(split="all", sample_rate=16000)
    """
```

### Tests

Every dataset has a test module at `tests/unittests/datasets_tests/test_<dataset_name>.py`, with **at least one test per method and property** in the [table above](#constructor-and-methods). Also include:

- **Reference stability:** a SHA-256 hash of the split manifest and of the first item's audio, computed with `alp_data.utils.create_hash`. Keep the code that generated the hashes in a comment at the top of the file, so they can be recomputed after an intentional data change.
- **Schema:** every required column is present, has the documented dtype, and has no missing values.
- **Audio:** returned audio is 1-D `float32` and the returned `sample_rate` is correct, for the native rate and for each pre-resampled rate.
- **Invalid split:** constructing with an unknown split raises `LookupError`.
- **Both backends and streaming:** the `polars` and `pandas` backends return the same items, and iteration works in streaming mode.
- **Config and output mapping:** `from_config` works with and without transformations, and `output_take_and_give` selects and renames columns.
- **Labels** (classification and detection): `get_available_labels` returns the expected labels for each entry of `annotation_columns`.
- **Windowed reads** (if supported): only the window is returned, and for detection, the selection table is relative to the window.

Reading long recordings is slow, so use module-scoped fixtures, read a few fixed items once and share them, and read audio at the native or a pre-resampled rate. [`test_sapsucker_woods.py`][test_sapsucker] and [`test_fasd13.py`][test_fasd13] are good models.

## Classification datasets

A classification dataset has one row per audio clip, each with one or more labels for the whole clip: species, call type, individual, and so on. A clip is either a whole file or a segment of a longer file, given by `window_start_sec` and `window_end_sec`.

**Required columns:**

| Column | Dtype | Required | Meaning |
|--------|-------|----------|---------|
| `audio_path` | `str` | yes | Original audio file, relative to `data_root` |
| `species_gbif_2023` | `str` | when the source gives species | Accepted GBIF scientific name of the vocalizing taxon (see [Taxonomy](#taxonomy)) |
| `species_original` | `str` | when the source gives species | Species name as given by the source |

Every classification dataset has at least one label column: a taxonomy column or one of the optional label columns below. List them in `annotation_columns` (see [Labels](#labels)).

**Optional columns**, in addition to the [shared optional columns](#metadata-columns) and [taxonomy columns](#taxonomy):

| Column | Dtype | Meaning |
|--------|-------|---------|
| `call_type` | `str` | Fine-grained vocalization type, according to the source |
| `coarse_call_type` | `str` | Coarser vocalization category (e.g. `call`, `whistle`, `click`, `burst_pulse`) |
| `individual_id` | `str` | Identifier of the vocalizing individual |
| `sex` | `str` | Sex of the vocalizing animal |
| `life_stage` | `str` | Life stage (e.g. `adult`, `juvenile`) |

**Output item:** every column of the row, plus `audio` (`np.ndarray`, `float32`, 1-D) and `sample_rate` (`int`).

### Classification references

- [`Watkins`][watkins]: the reference. It has a complete metadata docstring with pre-resampled audio with fallback resampling, and a test for every option.
    - Differs from the spec: the original species label is in `species`, not `species_original`; the taxonomy columns use the old names (`canonical_name`, `species_common`, `genus`, ...), not the `_gbif_2023` names; it has no `annotation_columns` or `get_available_labels`; and some dunder methods have no docstrings.
- [`InfantMarmosetsVox`][marmosets]: clips that are segments of longer recordings, with call-type and caller-ID labels.
    - Differs from the spec: `info.name` is `"InfantMarmosetsVox"`; the audio column is `path`, not `audio_path`; segments use `start` and `end`, not `window_start_sec` and `window_end_sec`; labels are integer codes in camelCase columns (`calltypeID`, `callerID`), not `call_type` and `individual_id`; it has no taxonomy columns; and it has no `annotation_columns` or `get_available_labels`.

## Detection datasets

A detection dataset has one row per recording, with a selection table listing every annotated event in it: when it starts and ends, optionally its frequency range, and its label(s).

**Required columns:**

| Column | Dtype | Meaning |
|--------|-------|---------|
| `audio_path` | `str` | Original audio file, relative to `data_root` |
| `selection_table` | `str` | The recording's events, as a tab-separated table in [Raven](https://www.ravensoundsoftware.com/) format (see below) |

**Optional columns:** the [shared optional columns](#metadata-columns). `duration_s` is strongly recommended, so users can plan windows without reading audio.

**Selection table columns.** In the split CSV the selection table is a TSV string. In the output item it is a `pd.DataFrame`, parsed in `_process` with `pd.read_csv(StringIO(row["selection_table"]), sep="\t")`.

| Column | Dtype | Required | Meaning |
|--------|-------|----------|---------|
| `Begin Time (s)` | `float` | yes | Event start, in seconds from the start of the returned audio |
| `End Time (s)` | `float` | yes | Event end, in seconds from the start of the returned audio |
| `species_gbif_2023` | `str` | when the source gives species | Accepted GBIF scientific name of the event's taxon (see [Taxonomy](#taxonomy)) |
| `species_original` | `str` | when the source gives species | Species label as given by the source |
| `Low Freq (Hz)`, `High Freq (Hz)` | `float` | no | Frequency bounds of the event |
| Other label columns | `str` | no | E.g. `call_type`, `individual_id`, following the [classification](#classification-datasets) names |

**Rules for events and labels:**

- List the selection table's label columns in `self.annotation_columns` and implement `get_available_labels` (see [Labels](#labels)).
- Every event has `End Time (s) > Begin Time (s)` and `Begin Time (s) >= 0`. Drop zero-length events when building the CSV.
- In `_process`, drop events that begin at or after the end of the returned audio.
- Keep recordings that have no events, with an empty selection table that still has the column headers.
- For events labeled `"Unknown"`, either provide a split with them and one without (e.g. `all` and `all_with_unknowns`), or document how they should be handled in scoring.
- **Say in the docstring whether annotation is exhaustive:** whether every audible event of the annotated classes is labeled, or only some. Detection metrics treat unlabeled time as negative, so this changes how the dataset can be used.

**Long recordings.** For recordings longer than a few minutes, support [windowed reads](#audio-output). When a row is windowed, re-base its selection table onto the window: drop events that don't overlap it, then shift and clip the remaining times so they line up with the returned audio. `FASD13._window_selection_table` is the implementation to follow.

**Output item:** every column of the row, plus `audio` (`np.ndarray`, `float32`, 1-D), `sample_rate` (`int`) and `selection_table` (`pd.DataFrame`).

### Detection references

- [`SapsuckerWoods`][sapsucker]: the reference. Its docstring has clear Labels and Splits sections (label mapping, `Unknown` handling, dropped events), and it provides `all` and `all_with_unknowns` splits. `HighSierraNevada`, `ColombiaCostaRica`, `SouthwesternAmazonBasin` and `SierraNevada` share its implementation.
    - Differs from the spec: its only label column is `Species`, holding GBIF names, not `species_gbif_2023`, and the source's eBird codes are not kept in a `species_original` column; `_process` doesn't drop events that begin after the audio ends; it has no `duration_s` column; and it has no pre-resampled audio.
- [`FASD13`][fasd13]: the model for windowed reads with re-based selection tables, pre-resampled audio (including documenting upsampled sources), per-row `license` and `citation`, and thorough tests.
    - Differs from the spec: its audio column is `audio_fp`, not `audio_path`. Its labels are nameless by design (few-shot detection), so there are no taxonomy columns.
- [`WABAD`][wabad]: the same row shape as `SapsuckerWoods`, plus pre-resampled audio and the guard against events past the end of the audio.
    - Differs from the spec: its audio column is `audio_fp`, not `audio_path`, and its label column is `Species`, not `species_gbif_2023` and `species_original`.

## Soundscape datasets

A soundscape dataset has one row per long, unlabeled recording. Recordings must be soundscapes: continuous recordings minutes to hours long, typically from passive acoustic monitoring, not short clips cut around vocalizations. Soundscape datasets are used for self-supervised training, unsupervised analysis and running detectors over real-world audio.

Soundscape datasets follow the [classification](#classification-datasets) template, with two differences:

- There are **no required metadata columns**. The only required column is the audio path.
- There are no label columns, no `annotation_columns` and no `get_available_labels`. If a dataset has event annotations, it is a [detection dataset](#detection-datasets).

**Required columns:**

| Column | Dtype | Meaning |
|--------|-------|---------|
| `audio_path` | `str` | Original audio file, relative to `data_root` |

**Optional columns:** the [shared optional columns](#metadata-columns).

**Long recordings.** Support [windowed reads](#audio-output), so users can read segments without loading hour-long files into memory. Shipping pre-resampled audio is strongly recommended, for the same reason.

**Output item:** every column of the row, plus `audio` (`np.ndarray`, `float32`, 1-D) and `sample_rate` (`int`).

### Soundscape references

There is no soundscape dataset in the library yet. Follow [`Watkins`][watkins] for the overall structure and [`FASD13`][fasd13] for windowed reads and pre-resampled audio.

## Checklist

Before opening a pull request for a new dataset:

- Module in `alp_data/datasets/`, class decorated with `@register_dataset`
- Class imported in `alp_data/datasets/__init__.py` and listed in `__all__`
- `DatasetInfo` complete: snake_case `name` matching the module, SPDX `license`, no placeholders
- Class docstring has Description, Available Metadata Fields, Available Splits, References and Examples (plus Labels for classification and detection)
- Required columns present with the documented dtypes and no missing values; column names in snake_case; no extra columns that duplicate a required or optional one
- When the source gives species: names converted to GBIF in `species_gbif_2023`, with the source's name kept in `species_original`
- Audio returned as 1-D `float32` at the requested `sample_rate`.
- Classification and detection: `annotation_columns` set and `get_available_labels` implemented
- Detection: events past the end of the audio dropped
- Detection and soundscape: windowed reads supported for long recordings
- Test module with at least one test per method, plus reference-hash, schema and audio tests
- Every raised exception documented in a `Raises` section; `ruff check --fix` passes

[watkins]: https://github.com/earthspecies/alp-data/blob/main/alp_data/datasets/watkins.py
[marmosets]: https://github.com/earthspecies/alp-data/blob/main/alp_data/datasets/infant_marmosets_vox.py
[sapsucker]: https://github.com/earthspecies/alp-data/blob/main/alp_data/datasets/sapsucker_woods.py
[fasd13]: https://github.com/earthspecies/alp-data/blob/main/alp_data/datasets/fasd13.py
[wabad]: https://github.com/earthspecies/alp-data/blob/main/alp_data/datasets/wabad.py
[test_sapsucker]: https://github.com/earthspecies/alp-data/blob/main/tests/unittests/datasets_tests/test_sapsucker_woods.py
[test_fasd13]: https://github.com/earthspecies/alp-data/blob/main/tests/unittests/datasets_tests/test_fasd13.py
