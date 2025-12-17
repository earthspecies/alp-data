"""Prompt template registry.

This module provides a global registry for prompt templates, allowing
templates to be registered by name and retrieved anywhere in the codebase.

Examples
--------
>>> from esp_data.prompts import BasePromptTemplate, register_prompt, get_prompt
>>>
>>> class MyTemplate(BasePromptTemplate):
...     name = "my_template"
...     response_field = "caption"
...
>>> # Register the template
>>> register_prompt(MyTemplate())
>>>
>>> # Later, retrieve it by name
>>> template = get_prompt("my_template")
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import PromptTemplate

__all__ = [
    "register_prompt",
    "get_prompt",
    "list_prompts",
    "is_registered",
]

# Global registry mapping template names to instances
_PROMPT_REGISTRY: dict[str, "PromptTemplate"] = {}


def register_prompt(template: "PromptTemplate") -> "PromptTemplate":
    """Register a prompt template instance.

    Parameters
    ----------
    template : PromptTemplate
        The template instance to register. Must have a unique `name` attribute.

    Returns
    -------
    PromptTemplate
        The same template instance (for decorator-style usage).

    Raises
    ------
    ValueError
        If a template with the same name is already registered.

    Examples
    --------
    >>> @register_prompt
    ... class MyTemplate(BasePromptTemplate):
    ...     name = "my_template"
    ...     ...

    Or as a function call:

    >>> template = MyTemplate()
    >>> register_prompt(template)
    """
    name = template.name
    if name in _PROMPT_REGISTRY:
        raise ValueError(
            f"Prompt template '{name}' is already registered. "
            f"Use a unique name or unregister the existing template first."
        )
    _PROMPT_REGISTRY[name] = template
    return template


def get_prompt(name: str) -> "PromptTemplate":
    """Get a registered prompt template by name.

    Parameters
    ----------
    name : str
        The name of the template to retrieve.

    Returns
    -------
    PromptTemplate
        The registered template instance.

    Raises
    ------
    KeyError
        If no template with the given name is registered.

    Examples
    --------
    >>> template = get_prompt("species_common")
    >>> result = template({"species_common": "Robin"})
    """
    if name not in _PROMPT_REGISTRY:
        available = list(_PROMPT_REGISTRY.keys())
        raise KeyError(
            f"Prompt template '{name}' is not registered. Available templates: {available}"
        )
    return _PROMPT_REGISTRY[name]


def list_prompts() -> list[str]:
    """List all registered prompt template names.

    Returns
    -------
    list[str]
        A list of registered template names.

    Examples
    --------
    >>> from esp_data.prompts import list_prompts
    >>> print(list_prompts())
    ['passthrough', 'species_common', ...]
    """
    return list(_PROMPT_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """Check if a prompt template is registered.

    Parameters
    ----------
    name : str
        The template name to check.

    Returns
    -------
    bool
        True if a template with this name is registered.
    """
    return name in _PROMPT_REGISTRY


def unregister_prompt(name: str) -> None:
    """Unregister a prompt template by name.

    Parameters
    ----------
    name : str
        The name of the template to unregister.

    Raises
    ------
    KeyError
        If no template with the given name is registered.
    """
    if name not in _PROMPT_REGISTRY:
        raise KeyError(f"Prompt template '{name}' is not registered.")
    del _PROMPT_REGISTRY[name]


def clear_registry() -> None:
    """Clear all registered prompt templates.

    Primarily useful for testing.
    """
    _PROMPT_REGISTRY.clear()
