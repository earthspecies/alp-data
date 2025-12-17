"""Base classes for prompt templates.

This module provides the protocol and base implementation for prompt templates
used in audio-language datasets. Templates define how to generate prompts
(instructions/questions) and responses (expected answers) from dataset items.

Examples
--------
>>> from esp_data.prompts import BasePromptTemplate, PromptVariant
>>>
>>> class SpeciesTemplate(BasePromptTemplate):
...     name = "species"
...
...     def __init__(self, seed=None):
...         super().__init__(
...             variants=[
...                 PromptVariant("What species is this?", "{species_common}"),
...                 PromptVariant("Identify the species.", "{species_common}"),
...             ],
...             seed=seed,
...         )
>>>
>>> template = SpeciesTemplate(seed=42)
>>> item = {"species_common": "American Robin"}
>>> result = template(item)
>>> result["prompt"]
'What species is this?'
>>> result["text"]
'American Robin'

Multiple response formats with coupled variants:

>>> class MultiFormatTemplate(BasePromptTemplate):
...     name = "multi_format"
...
...     def __init__(self, seed=None):
...         super().__init__(
...             variants=[
...                 PromptVariant("What species?", "{species_common}"),
...                 PromptVariant("Scientific name?", "{species_scientific}"),
...                 PromptVariant(
...                     "Describe the species",
...                     "{species_common} ({species_scientific})",
...                 ),
...             ],
...             seed=seed,
...         )
>>>
>>> template = MultiFormatTemplate(seed=1)
>>> item = {"species_common": "Robin", "species_scientific": "Turdus migratorius"}
>>> result = template(item)
>>> result["prompt"]
'What species?'
>>> result["text"]
'Robin'
"""

import random
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class PromptVariant:
    """A prompt-response pair with optional metadata.

    This class couples a prompt template with its expected response format,
    ensuring consistency when multiple prompt styles are used.

    Parameters
    ----------
    prompt : str
        The prompt/instruction template. Can include {field_name} placeholders.
    response : str
        The response template. Can include {field_name} placeholders.
    style : str, optional
        Style descriptor (e.g., "formal", "conversational"). Default: "default".
    metadata : dict, optional
        Additional metadata for the variant (e.g., difficulty, domain).

    Examples
    --------
    >>> variant = PromptVariant(
    ...     prompt="What species is this?",
    ...     response="{species_common}",
    ...     style="simple",
    ... )
    >>> variant.prompt
    'What species is this?'
    >>> variant.response
    '{species_common}'
    """

    prompt: str
    response: str
    style: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PromptTemplate(Protocol):
    """Protocol for prompt templates.

    All templates must have:
    - name: unique identifier for registry
    - format_prompt(): generates the instruction/question
    - format_response(): generates the expected answer/completion

    The __call__ method applies the template to an item dict,
    adding 'prompt' and 'text' keys.
    """

    name: str

    @abstractmethod
    def format_prompt(self, item: dict[str, Any]) -> str:
        """Generate the instruction/prompt from item data.

        Parameters
        ----------
        item : dict[str, Any]
            A dictionary containing dataset fields (e.g., species, caption).

        Returns
        -------
        str
            The formatted prompt/instruction string.
        """
        ...

    @abstractmethod
    def format_response(self, item: dict[str, Any]) -> str:
        """Generate the expected response/completion from item data.

        Parameters
        ----------
        item : dict[str, Any]
            A dictionary containing dataset fields.

        Returns
        -------
        str
            The formatted response/target string.
        """
        ...

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        """Apply template to item, adding 'prompt' and 'text' keys.

        Parameters
        ----------
        item : dict[str, Any]
            The source item dictionary.

        Returns
        -------
        dict[str, Any]
            A copy of the item with 'prompt' and 'text' keys added.
        """
        ...


class BasePromptTemplate:
    """Base implementation with variant sampling and response formatting.

    This class provides common functionality for prompt templates:
    - Random selection from multiple prompt-response variants
    - Each variant defines both its prompt and response template
    - Optional response prefix/suffix formatting

    Subclasses should override:
    - name: unique identifier
    - Pass variants to __init__

    Examples
    --------
    >>> class SpeciesTemplate(BasePromptTemplate):
    ...     name = "species_id"
    ...
    ...     def __init__(self, seed=None):
    ...         super().__init__(
    ...             variants=[
    ...                 PromptVariant("What species is this?", "{species_common}"),
    ...                 PromptVariant("Identify the species.", "{species_common}"),
    ...             ],
    ...             seed=seed,
    ...         )
    """

    name: str = "base"

    def __init__(
        self,
        variants: list[PromptVariant] | None = None,
        response_prefix: str = "",
        response_suffix: str = "",
        seed: int | None = None,
    ) -> None:
        """Initialize the template.

        Parameters
        ----------
        variants : list[PromptVariant], optional
            List of PromptVariant objects. Each defines both prompt and response
            templates with {field_name} placeholders for substitution.
            If None, uses a default "Describe this audio." prompt.
        response_prefix : str, optional
            Prefix to add before the response.
        response_suffix : str, optional
            Suffix to add after the response.
        seed : int, optional
            Random seed for deterministic variant selection.
        """
        if variants is None:
            variants = [PromptVariant("Describe this audio.", "{text}")]

        self._variants = variants
        self._selected_variant: PromptVariant | None = None
        self.response_prefix = response_prefix
        self.response_suffix = response_suffix
        self._rng = random.Random(seed)

    @property
    def variants(self) -> list[PromptVariant]:
        """Get the list of prompt-response variants.

        Returns
        -------
        list[PromptVariant]
            The configured variants for this template.
        """
        return self._variants

    def format_prompt(self, item: dict[str, Any]) -> str:
        """Select and format a prompt variant.

        Randomly selects from variants and formats with item fields.
        The selected variant is stored for use by format_response.

        Parameters
        ----------
        item : dict[str, Any]
            Item dictionary with fields for substitution.

        Returns
        -------
        str
            The formatted prompt string.

        Raises
        ------
        KeyError
            If the prompt variant contains placeholders not found in item.
        """
        self._selected_variant = self._rng.choice(self._variants)
        try:
            return self._selected_variant.prompt.format(**item)
        except KeyError as e:
            available_fields = list(item.keys())
            raise KeyError(
                f"Prompt template '{self._selected_variant.prompt}' requires field {e} "
                f"but item only has fields: {available_fields}"
            ) from e

    def format_response(self, item: dict[str, Any]) -> str:
        """Format the response using the selected variant.

        Parameters
        ----------
        item : dict[str, Any]
            Item dictionary containing the response field(s).

        Returns
        -------
        str
            The formatted response with optional prefix/suffix.

        Raises
        ------
        KeyError
            If response template contains placeholders not found in item.
        """
        if self._selected_variant is None:
            # format_prompt wasn't called first; select a variant now
            self._selected_variant = self._rng.choice(self._variants)

        try:
            response = self._selected_variant.response.format(**item)
        except KeyError as e:
            available_fields = list(item.keys())
            raise KeyError(
                f"Response template '{self._selected_variant.response}' "
                f"requires field {e} but item only has fields: {available_fields}"
            ) from e

        return f"{self.response_prefix}{response}{self.response_suffix}"

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        """Apply template to item, adding 'prompt' and 'text' keys.

        Parameters
        ----------
        item : dict[str, Any]
            The source item dictionary.

        Returns
        -------
        dict[str, Any]
            A copy of the item with 'prompt' and 'text' keys added.
        """
        result = item.copy()
        result["prompt"] = self.format_prompt(item)
        result["text"] = self.format_response(item)
        return result
