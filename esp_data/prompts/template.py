"""Base classes for multi-turn conversational prompt templates using Jinja2."""

from __future__ import annotations

import json
import random
from typing import Any, Literal

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, PrivateAttr
from typing_extensions import Self

from esp_data.io import AnyPathT, read_yaml


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str
    _compiled_template: Template | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Pre-compile the Jinja2 template after model initialization."""
        self._compiled_template = Template(self.content, undefined=StrictUndefined)

    def render(self, **kwargs: Any) -> "Message":
        """Render message content with Jinja2 template substitution.

        Uses StrictUndefined to raise clear errors for missing variables.

        Parameters
        ----------
        **kwargs : Any
            Template variables to substitute into the content.

        Returns
        -------
        Message
            New Message with rendered content (same role, substituted content).

        Examples
        --------
        >>> msg = Message(role="assistant", content="The species is {{ species }}.")
        >>> rendered = msg.render(species="American Robin")
        >>> rendered.content
        'The species is American Robin.'
        >>> rendered.role
        'assistant'
        """
        rendered = self._compiled_template.render(**kwargs)
        return Message(role=self.role, content=rendered)


class Conversation(BaseModel):
    """A conversation consisting of a sequence of messages.

    During training, any assistant messages can be selected as targets.
    """

    messages: list[Message]
    task: str = "default"

    def render(self, **kwargs: Any) -> list[dict[str, str]]:
        """Render all messages with template substitution.

        Returns
        -------
        list[dict[str, str]]
            List of rendered message dicts with 'role' and 'content' keys.
        """
        return [m.render(**kwargs).model_dump() for m in self.messages]


class ConversationTemplateConfig(BaseModel):
    """Configuration for a conversation template.

    Used for YAML-based configuration where conversation variants
    are defined declaratively.
    """

    name: str
    variants: list[Conversation]


class ConversationTemplate:
    """Conversation template with optional random variant selection.

    With a single variant, behavior is deterministic.
    With multiple variants, randomly selects one per call.
    Use seed parameter for reproducible results.

    Returns a dict with 'messages' (JSON of message sequence) and 'task'.
    """

    def __init__(
        self,
        name: str,
        variants: list[Conversation],
        seed: int | None = None,
    ) -> None:
        self.name = name
        self.variants = variants
        self._rng = random.Random(seed)

    @classmethod
    def from_config(
        cls,
        config: ConversationTemplateConfig | AnyPathT | dict[str, Any] | str,
        seed: int | None = None,
    ) -> Self:
        """Create a ConversationTemplate from a config, YAML file, or dict.

        Parameters
        ----------
        config : ConversationTemplateConfig | AnyPathT | dict | str
            Configuration source. Can be:
            - A ConversationTemplateConfig object
            - A path to a YAML file (str or AnyPathT)
            - A dict with 'name' and 'variants' keys
        seed : int | None, optional
            Random seed for variant selection, by default None.

        Returns
        -------
        Self
            A new ConversationTemplate instance.

        Raises
        ------
        ValueError
            If the config format is invalid.
        """
        if isinstance(config, ConversationTemplateConfig):
            return cls(name=config.name, variants=config.variants, seed=seed)

        if isinstance(config, (str, AnyPathT)):
            config = read_yaml(config)

        if isinstance(config, dict):
            parsed = ConversationTemplateConfig.model_validate(config)
            return cls(name=parsed.name, variants=parsed.variants, seed=seed)

        raise ValueError(
            f"Invalid config type: {type(config)}. "
            "Expected ConversationTemplateConfig, path, or dict."
        )

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        variant = self._rng.choice(self.variants)
        rendered_messages = variant.render(**item)

        if not rendered_messages:
            raise ValueError(f"Conversation '{self.name}' has no messages.")

        return {**item, "messages": json.dumps(rendered_messages), "task": variant.task}
