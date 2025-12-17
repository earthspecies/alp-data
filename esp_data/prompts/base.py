"""Base classes for prompt templates.

This module provides the protocol and base implementation for prompt templates
used in audio-language datasets. Templates define how to generate prompts
(instructions/questions) and responses (expected answers) from dataset items.

Examples
--------
Simple template with response field:

>>> from esp_data.prompts import BasePromptTemplate, register_prompt
>>>
>>> class MyTemplate(BasePromptTemplate):
...     name = "my_species_template"
...     response_field = "species_common"
...
...     @property
...     def default_prompt(self) -> str:
...         return "What species is in this recording?"
>>>
>>> template = MyTemplate()
>>> item = {"audio": ..., "species_common": "American Robin"}
>>> result = template(item)
>>> result["prompt"]
'What species is in this recording?'
>>> result["text"]
'American Robin'

Coupled prompt-response variants using PromptVariant:

>>> from esp_data.prompts import BasePromptTemplate, PromptVariant
>>>
>>> class CoupledTemplate(BasePromptTemplate):
...     name = "coupled_species"
...
...     def __init__(self, seed=None):
...         super().__init__(
...             variants=[
...                 PromptVariant(
...                     prompt="What species is this?",
...                     response="{species_common}",
...                 ),
...                 PromptVariant(
...                     prompt="Give the scientific name.",
...                     response="{species_scientific}",
...                 ),
...             ],
...             seed=seed,
...         )
>>>
>>> template = CoupledTemplate(seed=42)
>>> item = {"species_common": "American Robin", "species_scientific": "Turdus migratorius"}
>>> result = template(item)
>>> result["prompt"]
'What species is this?'
>>> result["text"]
'American Robin'
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
    - Support for coupled prompt-response pairs via PromptVariant
    - Optional response prefix/suffix formatting

    There are two ways to define variants:

    1. **Simple mode** (backwards compatible): Provide `prompt_variants` as strings
       and define `response_field` to specify which item field contains the response.

    2. **Coupled mode**: Provide `variants` as PromptVariant objects where each
       variant defines both its prompt and response template.

    Subclasses should override:
    - name: unique identifier
    - For simple mode: response_field, default_prompt
    - For coupled mode: pass variants to __init__

    Examples
    --------
    Simple mode with response_field:

    >>> class SpeciesTemplate(BasePromptTemplate):
    ...     name = "species_id"
    ...     response_field = "species_common"
    ...
    ...     @property
    ...     def default_prompt(self) -> str:
    ...         return "What species is this?"
    ...
    ...     def __init__(self, seed=None):
    ...         super().__init__(
    ...             prompt_variants=[
    ...                 "What species is this?",
    ...                 "Identify the species.",
    ...             ],
    ...             seed=seed,
    ...         )

    Coupled mode with PromptVariant:

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
    ...                     style="verbose",
    ...                 ),
    ...             ],
    ...             seed=seed,
    ...         )
    """

    name: str = "base"

    def __init__(
        self,
        prompt_variants: list[str] | None = None,
        variants: list[PromptVariant] | None = None,
        response_prefix: str = "",
        response_suffix: str = "",
        seed: int | None = None,
    ) -> None:
        """Initialize the template.

        Parameters
        ----------
        prompt_variants : list[str], optional
            List of prompt strings (simple mode). Can include {field_name}
            placeholders. If None and variants is None, uses self.default_prompt.
            Mutually exclusive with `variants`.
        variants : list[PromptVariant], optional
            List of PromptVariant objects (coupled mode). Each defines both
            prompt and response templates. Mutually exclusive with `prompt_variants`.
        response_prefix : str, optional
            Prefix to add before the response.
        response_suffix : str, optional
            Suffix to add after the response.
        seed : int, optional
            Random seed for deterministic variant selection.

        Raises
        ------
        ValueError
            If both prompt_variants and variants are provided.
        """
        if prompt_variants is not None and variants is not None:
            raise ValueError(
                "Cannot specify both 'prompt_variants' and 'variants'. "
                "Use 'prompt_variants' for simple mode (with response_field) "
                "or 'variants' for coupled mode (with PromptVariant objects)."
            )

        self._coupled_mode = variants is not None
        self._selected_variant: PromptVariant | None = None

        if variants is not None:
            # Coupled mode: use PromptVariant objects
            self._variants = variants
        elif prompt_variants is not None:
            # Simple mode with explicit prompt_variants: create variants from strings
            # Response will use response_field via _extract_response
            self._variants = [
                PromptVariant(prompt=p, response="{__response_field__}") for p in prompt_variants
            ]
        else:
            # Default: use default_prompt with response_field
            self._variants = [
                PromptVariant(prompt=self.default_prompt, response="{__response_field__}")
            ]

        self.response_prefix = response_prefix
        self.response_suffix = response_suffix
        self._rng = random.Random(seed)

    @property
    def default_prompt(self) -> str:
        """Default prompt template. Override in subclasses."""
        return "Describe this audio."

    @property
    def response_field(self) -> str:
        """Field name to extract response from (simple mode).

        Override in subclasses when using simple mode (prompt_variants as strings).

        Returns
        -------
        str
            The name of the item field containing the response/target.

        Raises
        ------
        NotImplementedError
            If not overridden in subclass and simple mode is used.
        """
        raise NotImplementedError(
            "Subclasses must define response_field when using simple mode "
            "(prompt_variants as strings). Use coupled mode (variants with "
            "PromptVariant objects) to avoid this requirement."
        )

    @property
    def variants(self) -> list[PromptVariant]:
        """Get the list of prompt-response variants.

        Returns
        -------
        list[PromptVariant]
            The configured variants for this template.
        """
        return self._variants

    @property
    def prompt_variants(self) -> list[str]:
        """Get list of prompt strings (backwards compatibility).

        Returns
        -------
        list[str]
            The prompt templates from all variants.
        """
        return [v.prompt for v in self._variants]

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
                f"Prompt variant '{self._selected_variant.prompt}' requires field {e} "
                f"but item only has fields: {available_fields}"
            ) from e

    def format_response(self, item: dict[str, Any]) -> str:
        """Format the response using the selected variant.

        In coupled mode, uses the response template from the selected variant.
        In simple mode, extracts from response_field.

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
            If coupled mode response template contains placeholders not in item.
        """
        if self._selected_variant is None:
            # format_prompt wasn't called first; select a variant now
            self._selected_variant = self._rng.choice(self._variants)

        if self._coupled_mode:
            # Coupled mode: format the response template
            try:
                response = self._selected_variant.response.format(**item)
            except KeyError as e:
                available_fields = list(item.keys())
                raise KeyError(
                    f"Response template '{self._selected_variant.response}' "
                    f"requires field {e} but item only has fields: {available_fields}"
                ) from e
        else:
            # Simple mode: use response_field
            response = self._extract_response(item)
            if response is None:
                return ""

        return f"{self.response_prefix}{response}{self.response_suffix}"

    def _extract_response(self, item: dict[str, Any]) -> str | None:
        """Extract raw response from item (simple mode).

        Override this method for complex extraction logic.

        Parameters
        ----------
        item : dict[str, Any]
            Item dictionary.

        Returns
        -------
        str or None
            The extracted response, or None if not found.
        """
        return item.get(self.response_field)

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


class PassthroughTemplate(BasePromptTemplate):
    """Template that passes through existing prompt/text fields.

    Use this template when wrapping datasets that already have
    'prompt' and 'text' fields defined. This provides a uniform
    interface while preserving the original values.

    Examples
    --------
    >>> template = PassthroughTemplate()
    >>> item = {"audio": ..., "prompt": "Describe this.", "text": "A bird singing."}
    >>> result = template(item)
    >>> result["prompt"]
    'Describe this.'
    >>> result["text"]
    'A bird singing.'
    """

    name: str = "passthrough"

    def __init__(
        self,
        prompt_field: str = "prompt",
        text_field: str = "text",
        seed: int | None = None,
    ) -> None:
        """Initialize the passthrough template.

        Parameters
        ----------
        prompt_field : str, optional
            Name of the field containing the prompt. Default: "prompt".
        text_field : str, optional
            Name of the field containing the text/response. Default: "text".
        seed : int, optional
            Random seed (unused, for interface compatibility).
        """
        # Don't call super().__init__ with variants to avoid the response_field check
        self._prompt_field = prompt_field
        self._text_field = text_field
        self._coupled_mode = True  # Treat as coupled to skip response_field
        self._selected_variant = None
        self._variants = []
        self._rng = random.Random(seed)
        self.response_prefix = ""
        self.response_suffix = ""

    @property
    def default_prompt(self) -> str:
        """Not used for passthrough."""
        return ""

    @property
    def response_field(self) -> str:
        """The text field being passed through."""
        return self._text_field

    def format_prompt(self, item: dict[str, Any]) -> str:
        """Return the existing prompt field value.

        Parameters
        ----------
        item : dict[str, Any]
            Item dictionary.

        Returns
        -------
        str
            The value of the prompt field, or empty string if not found.
        """
        return item.get(self._prompt_field, "")

    def format_response(self, item: dict[str, Any]) -> str:
        """Return the existing text field value.

        Parameters
        ----------
        item : dict[str, Any]
            Item dictionary.

        Returns
        -------
        str
            The value of the text field, or empty string if not found.
        """
        return item.get(self._text_field, "")
