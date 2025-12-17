from .base import BasePromptTemplate, PromptTemplate, PromptVariant
from .registry import get_prompt, list_prompts, register_prompt

# Register built-in passthrough template
_passthrough = BasePromptTemplate(variants=[PromptVariant("{prompt}", "{text}")])
_passthrough.name = "passthrough"
register_prompt(_passthrough)

__all__ = [
    "PromptTemplate",
    "PromptVariant",
    "BasePromptTemplate",
    "register_prompt",
    "get_prompt",
    "list_prompts",
]
