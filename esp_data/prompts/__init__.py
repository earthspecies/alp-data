from .registry import get_prompt, list_prompts, register_prompt, register_prompt_from_config
from .template import (
    Conversation,
    ConversationTemplate,
    ConversationTemplateConfig,
    Message,
)

# Default passthrough template - copies existing messages field
_passthrough = ConversationTemplate(
    name="passthrough",
    variants=[
        Conversation(
            messages=[Message(role="user", content="{{ prompt }}")],
        )
    ],
)
register_prompt(_passthrough)

__all__ = [
    # Template class
    "ConversationTemplate",
    # Data classes
    "Conversation",
    "ConversationTemplateConfig",
    "Message",
    # Registry functions
    "get_prompt",
    "list_prompts",
    "register_prompt",
    "register_prompt_from_config",
]
