"""Conversation template registry."""

from __future__ import annotations

from typing import Any

from .template import ConversationTemplate, ConversationTemplateConfig

_REGISTRY: dict[str, ConversationTemplate] = {}


def register_prompt(template: ConversationTemplate) -> ConversationTemplate:
    """Register a conversation template by name.

    Parameters
    ----------
    template : ConversationTemplate
        Template instance with a unique name attribute.

    Returns
    -------
    ConversationTemplate
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
    config: dict[str, Any] | ConversationTemplateConfig,
) -> ConversationTemplate:
    """Register a conversation template from a configuration dict or ConversationTemplateConfig.

    Creates a ConversationTemplate from the config and registers it.

    Parameters
    ----------
    config : dict[str, Any] | ConversationTemplateConfig
        Configuration for the conversation template. If dict, must have 'name' and 'variants' keys.

    Returns
    -------
    ConversationTemplate
        The registered template.
    """
    if isinstance(config, dict):
        config = ConversationTemplateConfig.model_validate(config)
    return register_prompt(ConversationTemplate(name=config.name, variants=config.variants))


def get_prompt(name: str) -> ConversationTemplate:
    """Retrieve a registered conversation template by name.

    Parameters
    ----------
    name : str
        Name of the registered template.

    Returns
    -------
    ConversationTemplate
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
    """List all registered conversation template names.

    Returns
    -------
    list[str]
        Names of all registered templates.
    """
    return list(_REGISTRY.keys())
