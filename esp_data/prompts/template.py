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
    """A prompt-response pair with explicit separation.

    The messages list contains the prompt (input to the model).
    The response field is a template string for the target output.
    """

    messages: list[Message]
    response: str
    task: str = "default"

    def render(self, **kwargs: Any) -> tuple[list[dict[str, str]], str]:
        """Render all messages and the response template.

        Returns
        -------
        tuple[list[dict[str, str]], str]
            Tuple of (rendered message dicts, rendered response string).
        """
        rendered_msgs = [m.render(**kwargs).model_dump() for m in self.messages]
        rendered_response = Template(self.response, undefined=StrictUndefined).render(**kwargs)
        return rendered_msgs, rendered_response


class PromptResponseTemplateConfig(BaseModel):
    """Configuration for a prompt-response template.

    Used for YAML-based configuration where prompt variants
    are defined declaratively.
    """

    name: str
    variants: list[PromptResponsePair]


class PromptResponseTemplate:
    """Prompt-response template with optional random variant selection.

    With a single variant, behavior is deterministic.
    With multiple variants, randomly selects one per call.
    Use seed parameter for reproducible results.

    Returns a dict with 'prompt' (JSON of messages) and 'response' (string).
    """

    def __init__(
        self,
        name: str,
        variants: list[PromptResponsePair],
        seed: int | None = None,
    ) -> None:
        self.name = name
        self.variants = variants
        self._rng = random.Random(seed)

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        variant = self._rng.choice(self.variants)
        prompt_msgs, response = variant.render(**item)

        if not prompt_msgs:
            raise ValueError(f"Prompt '{self.name}' has no messages.")

        return {**item, "prompt": json.dumps(prompt_msgs), "response": response}
