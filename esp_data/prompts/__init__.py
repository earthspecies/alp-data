"""Prompt template system for audio-language datasets.

This module provides a flexible system for defining and applying prompt templates
to convert audio datasets into audio-language format (audio, prompt, text).

Key Components
--------------
- PromptTemplate : Protocol defining the template interface
- PromptVariant : Dataclass for coupled prompt-response pairs
- BasePromptTemplate : Base class with variant sampling and response formatting
- PassthroughTemplate : Template for datasets that already have prompt/text fields
- register_prompt : Register a template in the global registry
- get_prompt : Retrieve a registered template by name

Examples
--------
Define a custom template with simple mode:

>>> from esp_data.prompts import BasePromptTemplate, register_prompt
>>>
>>> class SpeciesTemplate(BasePromptTemplate):
...     name = "species_id"
...     response_field = "species_common"
...
...     @property
...     def default_prompt(self) -> str:
...         return "What species is in this recording?"
...
...     def __init__(self, seed=None):
...         super().__init__(
...             prompt_variants=[
...                 "What species is in this recording?",
...                 "Identify the species.",
...                 "Name the animal heard in this audio.",
...             ],
...             seed=seed,
...         )
>>>
>>> # Register and use
>>> register_prompt(SpeciesTemplate())
>>> template = get_prompt("species_id")

Define a template with coupled prompt-response variants:

>>> from esp_data.prompts import BasePromptTemplate, PromptVariant
>>>
>>> class CoupledTemplate(BasePromptTemplate):
...     name = "coupled_species"
...
...     def __init__(self, seed=None):
...         super().__init__(
...             variants=[
...                 PromptVariant("What species?", "{species_common}"),
...                 PromptVariant("Scientific name?", "{species_scientific}"),
...             ],
...             seed=seed,
...         )

Use with AudioLanguageDataset:

>>> from esp_data.datasets import XenoCanto
>>> from esp_data.audio_language import AudioLanguageDataset
>>>
>>> base = XenoCanto(split="train")
>>> dataset = AudioLanguageDataset(base, prompt="species_id")
>>> sample = dataset[0]
>>> print(sample["prompt"], sample["text"])
"""

# Registry must be imported first since base.py registers PassthroughTemplate
from .base import (
    BasePromptTemplate,
    PassthroughTemplate,
    PromptTemplate,
    PromptVariant,
)
from .registry import (
    clear_registry,
    get_prompt,
    is_registered,
    list_prompts,
    register_prompt,
    unregister_prompt,
)

# Register built-in templates
register_prompt(PassthroughTemplate())

__all__ = [
    # Protocol and base classes
    "PromptTemplate",
    "PromptVariant",
    "BasePromptTemplate",
    "PassthroughTemplate",
    # Registry functions
    "register_prompt",
    "get_prompt",
    "list_prompts",
    "is_registered",
    "unregister_prompt",
    "clear_registry",
]
