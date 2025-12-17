"""Prompt template registry."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import PromptTemplate

_PROMPT_REGISTRY: dict[str, "PromptTemplate"] = {}


def register_prompt(template: "PromptTemplate") -> "PromptTemplate":
    """Register a prompt template instance.

    Returns
    -------
    PromptTemplate
        The registered template.

    Raises
    ------
    ValueError
        If template name is already registered.
    """
    name = template.name
    if name in _PROMPT_REGISTRY:
        raise ValueError(f"Prompt template '{name}' is already registered.")
    _PROMPT_REGISTRY[name] = template
    return template


def get_prompt(name: str) -> "PromptTemplate":
    """Get a registered prompt template by name.

    Returns
    -------
    PromptTemplate
        The registered template.

    Raises
    ------
    KeyError
        If template name is not registered.
    """
    if name not in _PROMPT_REGISTRY:
        available = list(_PROMPT_REGISTRY.keys())
        raise KeyError(f"Prompt template '{name}' not registered. Available: {available}")
    return _PROMPT_REGISTRY[name]


def list_prompts() -> list[str]:
    """List all registered prompt template names.

    Returns
    -------
    list[str]
        Registered template names.
    """
    return list(_PROMPT_REGISTRY.keys())
