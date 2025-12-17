"""Audio-language dataset wrapper.

This module provides the AudioLanguageDataset class, which wraps any esp-data
dataset to output samples in audio-language format: (audio, prompt, text).

The wrapper applies a prompt template to each item from the source dataset,
generating instruction prompts and expected responses from the dataset's fields.

Examples
--------
Basic usage with a custom template:

>>> from esp_data.datasets import XenoCanto
>>> from esp_data.audio_language import AudioLanguageDataset
>>> from esp_data.prompts import BasePromptTemplate, register_prompt
>>>
>>> class SpeciesTemplate(BasePromptTemplate):
...     name = "my_species"
...     response_field = "species_common"
...
...     @property
...     def default_prompt(self):
...         return "What species is in this recording?"
>>>
>>> _ = register_prompt(SpeciesTemplate())
>>>
>>> base = XenoCanto(split="train", sample_rate=16000)  # doctest: +SKIP
>>> dataset = AudioLanguageDataset(base, prompt="my_species")  # doctest: +SKIP
>>>
>>> sample = dataset[0]  # doctest: +SKIP
>>> print(sample.keys())  # doctest: +SKIP
dict_keys(['audio', 'prompt', 'text', ...])

With a dataset that already has prompt/text fields:

>>> # If base_dataset already provides 'prompt' and 'text' fields,
>>> # use passthrough mode (automatic detection):
>>> dataset = AudioLanguageDataset(base_dataset, prompt="passthrough")  # doctest: +SKIP
"""

from typing import Any, Dict, Iterator, Sequence

from esp_data.dataset import Dataset, DatasetConfig, DatasetInfo
from esp_data.prompts import PromptTemplate, get_prompt


def is_audio_language_dataset(dataset: Dataset) -> bool:
    """Check if a dataset natively provides audio-language format.

    A dataset is considered audio-language native if its items contain
    'audio', 'prompt', and 'text' fields.

    Parameters
    ----------
    dataset : Dataset
        The dataset to check.

    Returns
    -------
    bool
        True if the dataset provides native audio-language format.

    Notes
    -----
    This function accesses dataset[0] to check fields, which may trigger
    audio loading for that sample.
    """
    try:
        sample = dataset[0]
        return all(k in sample for k in ("audio", "prompt", "text"))
    except (IndexError, KeyError, RuntimeError):
        return False


class AudioLanguageDataset(Dataset):
    """Wraps any esp-data dataset to output (audio, prompt, text) samples.

    This enables easy conversion of existing datasets to audio-language format
    with configurable prompt templates.

    The wrapper:
    - Loads items from the source dataset (including audio)
    - Applies a prompt template to generate 'prompt' and 'text' fields
    - Optionally includes all source fields in output

    Parameters
    ----------
    source : Dataset
        The source esp-data dataset to wrap.
    prompt : str | PromptTemplate
        Either a registered prompt name (str) or a PromptTemplate instance.
        Use "passthrough" for datasets that already have prompt/text fields.
    include_source_fields : bool, optional
        If True (default), include all source fields in output.
        If False, output only audio, prompt, text.

    Attributes
    ----------
    source : Dataset
        The wrapped source dataset.
    template : PromptTemplate
        The prompt template being applied.
    sample_rate : int or None
        Sample rate from source dataset if available.

    Examples
    --------
    Wrap XenoCanto for species identification:

    >>> from esp_data.datasets import XenoCanto  # doctest: +SKIP
    >>> from esp_data.audio_language import AudioLanguageDataset  # doctest: +SKIP
    >>>
    >>> base = XenoCanto(split="train", sample_rate=16000)  # doctest: +SKIP
    >>> dataset = AudioLanguageDataset(base, prompt="species_common")  # doctest: +SKIP
    >>> sample = dataset[0]  # doctest: +SKIP
    >>> print(sample["prompt"])  # "What species is vocalizing..."  # doctest: +SKIP
    >>> print(sample["text"])    # "American Robin"  # doctest: +SKIP

    Use with ConcatenatedDataset:

    >>> from esp_data.concat import ConcatenatedDataset  # doctest: +SKIP
    >>> xc = XenoCanto(split="train")  # doctest: +SKIP
    >>> ds1 = AudioLanguageDataset(xc, prompt="species_common")  # doctest: +SKIP
    >>> ds2 = AudioLanguageDataset(AudioSet(split="train"), prompt="captioning")  # doctest: +SKIP
    >>> combined = ConcatenatedDataset([ds1, ds2])  # doctest: +SKIP

    Auto-detect native audio-language datasets:

    >>> # If source already has prompt/text, passthrough is used automatically
    >>> native_al_dataset = SomeNativeALDataset()  # doctest: +SKIP
    >>> dataset = AudioLanguageDataset(native_al_dataset, prompt="passthrough")  # doctest: +SKIP
    """

    info = DatasetInfo(
        name="audio_language",
        owner="ESP Data Team",
        split_paths={"wrapped": "virtual://wrapped_dataset"},
        version="0.1.0",
        description="Audio-language wrapper for esp-data datasets",
        sources=["Wrapped dataset"],
        license="Same as source",
    )

    def __init__(
        self,
        source: Dataset,
        prompt: str | PromptTemplate,
        include_source_fields: bool = True,
    ) -> None:
        """Initialize the audio-language wrapper.

        Parameters
        ----------
        source : Dataset
            The source esp-data dataset to wrap.
        prompt : str | PromptTemplate
            Either a registered prompt name (str) or a PromptTemplate instance.
        include_source_fields : bool, optional
            If True, include all source fields in output (default).
            If False, output only audio, prompt, text.
        """
        super().__init__()
        self.source = source
        self.include_source_fields = include_source_fields
        # Point to source's _data for compatibility with ConcatenatedDataset
        self._data = source._data

        # Resolve prompt template
        if isinstance(prompt, str):
            self.template = get_prompt(prompt)
        else:
            self.template = prompt

        # Update info based on source
        self.info = self.info.model_copy(deep=True)
        self.info.name = f"audio_language({source.info.name})"
        self.info.sources = source.info.sources if source.info.sources else ["Unknown"]
        self.info.license = source.info.license

        # Copy sample_rate from source if available
        self.sample_rate = getattr(source, "sample_rate", None)

        # Copy split from source if available
        self.split = getattr(source, "split", "wrapped")

    @property
    def columns(self) -> Sequence[str]:
        """Get the columns of the dataset.

        Returns
        -------
        Sequence[str]
            List of column names. Always includes 'audio', 'prompt', 'text'.
            May include source columns if include_source_fields is True.
        """
        base_cols = ["audio", "prompt", "text"]
        if self.include_source_fields:
            source_cols = [c for c in self.source.columns if c not in base_cols]
            return base_cols + source_cols
        return base_cols

    @property
    def available_splits(self) -> Sequence[str]:
        """Get the available splits of the dataset.

        Returns
        -------
        Sequence[str]
            Available splits from the source dataset.
        """
        return self.source.available_splits

    def _load(self) -> None:
        """Load the dataset (no-op, source handles loading)."""
        pass  # Source dataset handles loading

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the source dataset.
        """
        return len(self.source)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to get.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing at minimum 'audio', 'prompt', 'text' keys.
            If include_source_fields is True, also includes all source fields.
        """
        # Get source item (includes audio loading via source._process)
        source_item = self.source[idx]

        # Apply prompt template (adds 'prompt' and 'text' keys)
        result = self.template(source_item)

        if not self.include_source_fields:
            # Return only core fields
            return {
                "audio": result.get("audio"),
                "prompt": result["prompt"],
                "text": result["text"],
            }

        return result

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        ------
        Dict[str, Any]
            Each sample in the dataset with audio-language format.
        """
        for idx in range(len(self)):
            yield self[idx]

    def __str__(self) -> str:
        """Return a string representation of the dataset.

        Returns
        -------
        str
            A string representation including source and template info.
        """
        return (
            f"{self.info.name}\n"
            f"Source: {self.source.info.name} (v{self.source.info.version})\n"
            f"Template: {self.template.name}\n"
            f"Split: {self.split}\n"
            f"Length: {len(self)}"
        )

    @classmethod
    def from_config(cls, config: DatasetConfig) -> tuple["AudioLanguageDataset", Dict[str, Any]]:
        """Create a dataset from configuration (not implemented).

        For AudioLanguageDataset, use direct instantiation with a source dataset.

        Raises
        ------
        NotImplementedError
            Always raised. Use direct instantiation instead.
        """
        raise NotImplementedError(
            "AudioLanguageDataset does not support from_config. "
            "Use direct instantiation: AudioLanguageDataset(source_dataset, prompt=...)"
        )
