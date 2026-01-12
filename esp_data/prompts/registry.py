"""Prompt template registry."""

from __future__ import annotations

from typing import Any

from .template import PromptResponseTemplate, PromptResponseTemplateConfig

_REGISTRY: dict[str, PromptResponseTemplate] = {}


def register_prompt(template: PromptResponseTemplate) -> PromptResponseTemplate:
    """Register a prompt template by name.

    Parameters
    ----------
    template : PromptResponseTemplate
        Template instance with a unique name attribute.

    Returns
    -------
    PromptResponseTemplate
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
    config: dict[str, Any] | PromptResponseTemplateConfig,
) -> PromptResponseTemplate:
    """Register a prompt template from a configuration dict or PromptResponseTemplateConfig.

    Creates a PromptResponseTemplate from the config and registers it.

    Parameters
    ----------
    config : dict[str, Any] | PromptResponseTemplateConfig
        Configuration for the prompt template. If dict, must have 'name' and 'variants' keys.

    Returns
    -------
    PromptResponseTemplate
        The registered template.
    """
    if isinstance(config, dict):
        config = PromptResponseTemplateConfig.model_validate(config)
    return register_prompt(PromptResponseTemplate(name=config.name, variants=config.variants))


def get_prompt(name: str) -> PromptResponseTemplate:
    """Retrieve a registered prompt template by name.

    Parameters
    ----------
    name : str
        Name of the registered template.

    Returns
    -------
    PromptResponseTemplate
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
