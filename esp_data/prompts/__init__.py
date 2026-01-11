from .registry import get_prompt, list_prompts, register_prompt, register_prompt_from_config
from .template import (
    Message,
    PromptResponsePair,
    PromptTemplate,
    PromptTemplateConfig,
)

# Default passthrough template - copies existing prompt/response fields
_passthrough = PromptTemplate(
    name="passthrough",
    variants=PromptResponsePair(
        messages=[
            Message(role="user", content="{{ prompt }}"),
            Message(role="assistant", content="{{ response }}"),
        ]
    ),
)
register_prompt(_passthrough)

__all__ = [
    # Template class
    "PromptTemplate",
    # Data classes
    "Message",
    "PromptResponsePair",
    "PromptTemplateConfig",
    # Registry functions
    "register_prompt",
    "register_prompt_from_config",
    "get_prompt",
    "list_prompts",
]
