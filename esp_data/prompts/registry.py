"""Prompt template registry."""

from __future__ import annotations

from typing import Any

from .template import PromptTemplate, PromptTemplateConfig

_REGISTRY: dict[str, PromptTemplate] = {}


def register_prompt(template: PromptTemplate) -> PromptTemplate:
    """Register a prompt template by name.

    Parameters
    ----------
    template : PromptTemplate
        Template instance with a unique name attribute.

    Returns
    -------
    PromptTemplate
        The registered template (for decorator use).

    Raises
    ------
    ValueError
        If a template with the same name is already registered.
    """
    if template.name in _REGISTRY:
        raise ValueError(f"Prompt '{template.name}' already registered.")
    _REGISTRY[template.name] = template
    return template


def register_prompt_from_config(
    config: dict[str, Any] | PromptTemplateConfig,
) -> PromptTemplate:
    """Register a prompt template from a configuration dict or PromptTemplateConfig.

    Creates a PromptTemplate from the config and registers it.

    Parameters
    ----------
    config : dict[str, Any] | PromptTemplateConfig
        Configuration for the prompt template. If dict, must have 'name' and 'variants' keys.

    Returns
    -------
    PromptTemplate
        The registered template.
    """
    if isinstance(config, dict):
        config = PromptTemplateConfig.model_validate(config)
    return register_prompt(PromptTemplate(name=config.name, variants=config.variants))


def get_prompt(name: str) -> PromptTemplate:
    """Retrieve a registered prompt template by name.

    Parameters
    ----------
    name : str
        Name of the registered template.

    Returns
    -------
    PromptTemplate
        The registered template.

    Raises
    ------
    KeyError
        If no template with the given name is registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Prompt '{name}' not registered. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_prompts() -> list[str]:
    """List all registered prompt template names.

    Returns
    -------
    list[str]
        Names of all registered templates.
    """
    return list(_REGISTRY.keys())
