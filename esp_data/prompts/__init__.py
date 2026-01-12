from .registry import get_prompt, list_prompts, register_prompt, register_prompt_from_config
from .template import (
    Message,
    PromptResponsePair,
    PromptResponseTemplate,
    PromptResponseTemplateConfig,
)

# Default passthrough template - copies existing prompt/response fields
_passthrough = PromptResponseTemplate(
    name="passthrough",
    variants=[
        PromptResponsePair(
            messages=[Message(role="user", content="{{ prompt }}")],
            response="{{ response }}",
        )
    ],
)
register_prompt(_passthrough)

__all__ = [
    # Template class
    "PromptResponseTemplate",
    # Data classes
    "Message",
    "PromptResponsePair",
    "PromptResponseTemplateConfig",
    # Registry functions
    "register_prompt",
    "register_prompt_from_config",
    "get_prompt",
    "list_prompts",
]
