"""Tests for esp_data.prompts module."""

import json

import pytest
from jinja2 import UndefinedError

from esp_data.prompts import (
    Conversation,
    ConversationTemplate,
    ConversationTemplateConfig,
    Message,
)


# --- Test Message ---


class TestMessage:
    """Tests for Message dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic Message."""
        msg = Message(role="user", content="What species is this?")
        assert msg.role == "user"
        assert msg.content == "What species is this?"

    def test_render_with_variables(self) -> None:
        """Test rendering message content with Jinja2 variables."""
        msg = Message(role="assistant", content="{{ species_common }}")
        rendered = msg.render(species_common="American Robin")
        assert rendered.role == "assistant"
        assert rendered.content == "American Robin"

    def test_render_complex_template(self) -> None:
        """Test rendering with complex Jinja2 template."""
        msg = Message(
            role="assistant",
            content="The species is {{ species }} from {{ location }}.",
        )
        rendered = msg.render(species="robin", location="California")
        assert rendered.content == "The species is robin from California."

    def test_render_missing_variable_raises(self) -> None:
        """Test that missing variables raise UndefinedError with StrictUndefined."""
        msg = Message(role="assistant", content="{{ missing_var }}")
        with pytest.raises(UndefinedError):
            msg.render(other_var="value")

    def test_roles(self) -> None:
        """Test all valid roles."""
        for role in ["system", "user", "assistant"]:
            msg = Message(role=role, content="test")
            assert msg.role == role


# --- Test Conversation ---


class TestConversation:
    """Tests for Conversation model."""

    def test_basic_creation(self) -> None:
        """Test creating a basic Conversation."""
        conv = Conversation(
            messages=[Message(role="user", content="What species?")],
        )
        assert len(conv.messages) == 1
        assert conv.task == "default"

    def test_with_task(self) -> None:
        """Test creating a Conversation with custom task."""
        conv = Conversation(
            messages=[Message(role="user", content="Identify the species")],
            task="species_id",
        )
        assert conv.task == "species_id"

    def test_render(self) -> None:
        """Test rendering a conversation with data."""
        conv = Conversation(
            messages=[
                Message(role="user", content="What is {{ field }}?"),
                Message(role="assistant", content="{{ answer }}"),
            ],
        )
        rendered_msgs = conv.render(field="the species", answer="Robin")
        assert len(rendered_msgs) == 2
        assert rendered_msgs[0] == {"role": "user", "content": "What is the species?"}
        assert rendered_msgs[1] == {"role": "assistant", "content": "Robin"}

    def test_multi_turn_conversation(self) -> None:
        """Test a multi-turn conversation."""
        conv = Conversation(
            messages=[
                Message(role="system", content="You are a bioacoustics expert."),
                Message(role="user", content="What species is in this audio?"),
                Message(role="assistant", content="{{ species_common }}"),
            ],
        )
        rendered_msgs = conv.render(species_common="Blue Jay")
        assert len(rendered_msgs) == 3
        assert rendered_msgs[0]["role"] == "system"
        assert rendered_msgs[1]["role"] == "user"
        assert rendered_msgs[2]["role"] == "assistant"
        assert rendered_msgs[2]["content"] == "Blue Jay"


# --- Test ConversationTemplateConfig ---


class TestConversationTemplateConfig:
    """Tests for ConversationTemplateConfig model."""

    def test_basic_creation(self) -> None:
        """Test creating a ConversationTemplateConfig."""
        config = ConversationTemplateConfig(
            name="test_prompt",
            variants=[
                Conversation(
                    messages=[
                        Message(role="user", content="Question?"),
                        Message(role="assistant", content="{{ answer }}"),
                    ],
                )
            ],
        )
        assert config.name == "test_prompt"
        assert len(config.variants) == 1

    def test_multiple_variants(self) -> None:
        """Test config with multiple variants."""
        config = ConversationTemplateConfig(
            name="multi_variant",
            variants=[
                Conversation(
                    messages=[
                        Message(role="user", content="What species?"),
                        Message(role="assistant", content="{{ species }}"),
                    ],
                ),
                Conversation(
                    messages=[
                        Message(role="user", content="Identify the animal."),
                        Message(role="assistant", content="{{ species }}"),
                    ],
                ),
            ],
        )
        assert len(config.variants) == 2


# --- Test ConversationTemplate ---


class TestConversationTemplate:
    """Tests for ConversationTemplate."""

    def test_basic_call(self) -> None:
        """Test basic template call adds messages and task."""
        conv = Conversation(
            messages=[
                Message(role="user", content="What species?"),
                Message(role="assistant", content="{{ species }}"),
            ],
        )
        template = ConversationTemplate(name="test_basic", variants=[conv])
        item = {"species": "Robin", "extra": "data"}
        result = template(item)

        assert "messages" in result
        assert "task" in result
        assert result["task"] == "default"
        assert result["extra"] == "data"  # Original data preserved

        # Messages should be JSON of messages
        messages = json.loads(result["messages"])
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What species?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Robin"

    def test_original_item_not_modified(self) -> None:
        """Test that original item is not modified."""
        conv = Conversation(
            messages=[
                Message(role="user", content="Q?"),
                Message(role="assistant", content="{{ a }}"),
            ],
        )
        template = ConversationTemplate(name="test_no_modify", variants=[conv])
        item = {"a": "answer"}
        result = template(item)

        assert "messages" not in item
        assert "task" not in item
        assert "messages" in result

    def test_multi_turn_conversation(self) -> None:
        """Test multi-turn conversation produces correct messages."""
        conv = Conversation(
            messages=[
                Message(role="system", content="You are an expert."),
                Message(role="user", content="What is {{ thing }}?"),
                Message(role="assistant", content="{{ answer }}"),
            ],
        )
        template = ConversationTemplate(name="test_multi_turn", variants=[conv])
        result = template({"thing": "this", "answer": "A bird"})

        messages = json.loads(result["messages"])
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "A bird"

    def test_single_variant_deterministic(self) -> None:
        """Test that single variant is deterministic."""
        conv = Conversation(
            messages=[
                Message(role="user", content="The only prompt"),
                Message(role="assistant", content="{{ x }}"),
            ],
        )
        template = ConversationTemplate(name="test_deterministic", variants=[conv])
        item = {"x": "value"}

        # Call multiple times - should always get same result
        for _ in range(10):
            result = template(item)
            messages = json.loads(result["messages"])
            assert messages[0]["content"] == "The only prompt"

    def test_multiple_variants_with_seed(self) -> None:
        """Test that seed produces deterministic variant selection."""
        variants = [
            Conversation(
                messages=[
                    Message(role="user", content="A"),
                    Message(role="assistant", content="{{ x }}"),
                ],
            ),
            Conversation(
                messages=[
                    Message(role="user", content="B"),
                    Message(role="assistant", content="{{ x }}"),
                ],
            ),
            Conversation(
                messages=[
                    Message(role="user", content="C"),
                    Message(role="assistant", content="{{ x }}"),
                ],
            ),
        ]

        template1 = ConversationTemplate(name="test_seed", variants=variants, seed=12345)
        template2 = ConversationTemplate(name="test_seed", variants=variants, seed=12345)

        item = {"x": "value"}
        results1 = [json.loads(template1(item)["messages"])[0]["content"] for _ in range(20)]
        results2 = [json.loads(template2(item)["messages"])[0]["content"] for _ in range(20)]

        assert results1 == results2

    def test_multiple_variants_randomness(self) -> None:
        """Test that multiple variants are randomly selected without seed."""
        variants = [
            Conversation(
                messages=[
                    Message(role="user", content="A"),
                    Message(role="assistant", content="{{ x }}"),
                ],
            ),
            Conversation(
                messages=[
                    Message(role="user", content="B"),
                    Message(role="assistant", content="{{ x }}"),
                ],
            ),
            Conversation(
                messages=[
                    Message(role="user", content="C"),
                    Message(role="assistant", content="{{ x }}"),
                ],
            ),
        ]

        template = ConversationTemplate(name="test_random", variants=variants)
        item = {"x": "value"}

        # Collect first user messages from multiple calls
        user_contents = set()
        for _ in range(100):
            result = template(item)
            messages = json.loads(result["messages"])
            user_contents.add(messages[0]["content"])

        # Should see variety (though not guaranteed, very likely with 100 samples)
        assert len(user_contents) > 1

    def test_jinja2_rendering(self) -> None:
        """Test Jinja2 template rendering in messages."""
        conv = Conversation(
            messages=[
                Message(
                    role="user",
                    content="Describe the {{ species }} from {{ location }}.",
                ),
                Message(
                    role="assistant",
                    content="The {{ species }} is a bird found in {{ location }}.",
                ),
            ],
        )
        template = ConversationTemplate(name="test_jinja", variants=[conv])
        result = template({"species": "robin", "location": "California"})

        messages = json.loads(result["messages"])
        assert messages[0]["content"] == "Describe the robin from California."
        assert messages[1]["content"] == "The robin is a bird found in California."

    def test_jinja2_conditionals(self) -> None:
        """Test Jinja2 conditional logic in templates."""
        conv = Conversation(
            messages=[
                Message(role="user", content="Describe this audio."),
                Message(
                    role="assistant",
                    content="{% if behavior == 'song' %}A singing {{ species }}{% else %}A {{ species }} {{ behavior }}{% endif %}",
                ),
            ],
        )
        template = ConversationTemplate(name="test_conditionals", variants=[conv])

        # Test "song" branch
        result = template({"behavior": "song", "species": "robin"})
        messages = json.loads(result["messages"])
        assert messages[1]["content"] == "A singing robin"

        # Test else branch
        result = template({"behavior": "call", "species": "jay"})
        messages = json.loads(result["messages"])
        assert messages[1]["content"] == "A jay call"

    def test_jinja2_filters(self) -> None:
        """Test Jinja2 built-in filters."""
        conv = Conversation(
            messages=[
                Message(role="user", content="What species?"),
                Message(role="assistant", content="{{ species | upper }}"),
            ],
        )
        template = ConversationTemplate(name="test_filters", variants=[conv])
        result = template({"species": "robin"})
        messages = json.loads(result["messages"])
        assert messages[1]["content"] == "ROBIN"

    def test_jinja2_loops(self) -> None:
        """Test Jinja2 loop constructs."""
        conv = Conversation(
            messages=[
                Message(role="user", content="List the species."),
                Message(
                    role="assistant",
                    content="{% for s in species %}{{ s }}{% if not loop.last %}, {% endif %}{% endfor %}",
                ),
            ],
        )
        template = ConversationTemplate(name="test_loops", variants=[conv])
        result = template({"species": ["robin", "jay", "cardinal"]})
        messages = json.loads(result["messages"])
        assert messages[1]["content"] == "robin, jay, cardinal"

    def test_empty_messages_raises(self) -> None:
        """Test that empty messages raises ValueError."""
        conv = Conversation(messages=[])
        template = ConversationTemplate(name="test_empty", variants=[conv])

        with pytest.raises(ValueError, match="no messages"):
            template({})

    def test_missing_variable_raises(self) -> None:
        """Test that missing template variables raise UndefinedError."""
        conv = Conversation(
            messages=[
                Message(role="user", content="What is {{ missing }}?"),
                Message(role="assistant", content="Answer"),
            ],
        )
        template = ConversationTemplate(name="test_missing", variants=[conv])

        with pytest.raises(UndefinedError):
            template({"other": "value"})

    def test_name_property(self) -> None:
        """Test that name property is set correctly."""
        conv = Conversation(
            messages=[
                Message(role="user", content="Q"),
                Message(role="assistant", content="A"),
            ],
        )
        template = ConversationTemplate(name="my_template", variants=[conv])
        assert template.name == "my_template"

    def test_accepts_list_of_variants(self) -> None:
        """Test that variants must be a list."""
        variants = [
            Conversation(
                messages=[
                    Message(role="user", content="Question"),
                    Message(role="assistant", content="{{ answer }}"),
                ],
            )
        ]
        template = ConversationTemplate(name="test_list", variants=variants)
        result = template({"answer": "Response"})

        messages = json.loads(result["messages"])
        assert messages[1]["content"] == "Response"

    def test_task_in_output(self) -> None:
        """Test that task is included in output."""
        conv = Conversation(
            messages=[
                Message(role="user", content="Q"),
                Message(role="assistant", content="A"),
            ],
            task="species_id",
        )
        template = ConversationTemplate(name="test_task", variants=[conv])
        result = template({})

        assert result["task"] == "species_id"
