"""Audio-language dataset transformation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from esp_data.backends.protocol import DataBackend
from esp_data.prompts import (
    PromptResponseTemplate,
    PromptResponseTemplateConfig,
    get_prompt,
)

from . import register_transform


class AudioLanguageConfig(BaseModel):
    """Configuration for the AudioLanguage transform.

    Parameters
    ----------
    type : Literal["audio_language"]
        Must be "audio_language" for transform registry.
    prompt : str | PromptResponseTemplateConfig | None
        Prompt template name, inline config, or None for passthrough.
    seed : int | None
        Random seed for stochastic variant selection.
    """

    type: Literal["audio_language"]
    prompt: str | PromptResponseTemplateConfig | None = None
    seed: int | None = None


class AudioLanguage:
    """Transform to add 'prompt' and 'response' columns using a prompt template.

    Parameters
    ----------
    prompt : str | PromptResponseTemplateConfig | None
        - str: Name of a registered prompt template
        - PromptResponseTemplateConfig: Inline prompt configuration with variants
        - None: Use passthrough (copies existing prompt/response fields)
    seed : int | None
        Random seed for reproducible stochastic variant selection.
    """

    def __init__(
        self,
        prompt: str | PromptResponseTemplateConfig | None = None,
        seed: int | None = None,
    ) -> None:
        if prompt is None:
            self.template = get_prompt("passthrough")
        elif isinstance(prompt, str):
            self.template = get_prompt(prompt)
        else:
            self.template = PromptResponseTemplate(
                name=prompt.name, variants=prompt.variants, seed=seed
            )

    @classmethod
    def from_config(cls, cfg: AudioLanguageConfig) -> AudioLanguage:
        return cls(prompt=cfg.prompt, seed=cfg.seed)

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        prompts, responses = [], []

        for row in backend:
            result = self.template(row)
            prompts.append(result["prompt"])
            responses.append(result["response"])

        backend = backend.add_column("prompt", prompts).add_column("response", responses)
        metadata = {"prompt_template": self.template.name, "num_rows": len(prompts)}

        return backend, metadata


register_transform(AudioLanguageConfig, AudioLanguage)
