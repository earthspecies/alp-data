"""Base classes for multi-turn conversational prompt templates using Jinja2."""

from __future__ import annotations

import json
import random
from typing import Any, Literal

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str

    def render(self, **kwargs: Any) -> Message:
        """Render message content with Jinja2 template substitution.

        Uses StrictUndefined to raise clear errors for missing variables.

        Returns
        -------
        Message
            New Message with rendered content.
        """
        rendered = Template(self.content, undefined=StrictUndefined).render(**kwargs)
        return Message(role=self.role, content=rendered)


class PromptResponsePair(BaseModel):
    """A single prompt-response conversation as a sequence of messages.

    The last message with role="assistant" is treated as the response;
    all preceding messages form the prompt.
    """

    messages: list[Message]
    task: str = "default"

    def render(self, **kwargs: Any) -> list[dict[str, str]]:
        """Render all messages in this pair.

        Returns
        -------
        list[dict[str, str]]
            List of rendered message dicts with 'role' and 'content' keys.
        """
        return [m.render(**kwargs).model_dump() for m in self.messages]


class PromptTemplateConfig(BaseModel):
    """Configuration for a prompt template.

    Used for YAML-based configuration where prompt variants
    are defined declaratively.
    """

    name: str
    variants: list[PromptResponsePair]


class PromptTemplate:
    """Prompt template with optional random variant selection.

    With a single variant, behavior is deterministic.
    With multiple variants, randomly selects one per call.
    Use seed parameter for reproducible results.

    Splits conversation into 'prompt' (JSON of messages) and 'response'
    (final assistant message content).
    """

    def __init__(
        self,
        name: str,
        variants: list[PromptResponsePair] | PromptResponsePair,
        seed: int | None = None,
    ) -> None:
        self.name = name
        self.variants = [variants] if isinstance(variants, PromptResponsePair) else variants
        self._rng = random.Random(seed)

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        variant = self._rng.choice(self.variants)
        rendered = variant.render(**item)

        if not rendered:
            raise ValueError(f"Prompt '{self.name}' has no messages.")

        last = rendered[-1]
        if last["role"] == "assistant":
            prompt_msgs = rendered[:-1]
            response = last["content"]
        else:
            prompt_msgs = rendered
            response = ""

        return {**item, "prompt": json.dumps(prompt_msgs), "response": response}
