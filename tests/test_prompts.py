"""Tests for esp_data.prompts module."""

import json

import pytest
from jinja2 import UndefinedError

from esp_data.prompts import (
    Message,
    PromptResponsePair,
    PromptResponseTemplate,
    PromptResponseTemplateConfig,
    get_prompt,
    list_prompts,
    register_prompt,
)
from esp_data.prompts.registry import _REGISTRY


# --- Fixtures ---


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean up registry before and after each test."""
    original = set(list_prompts())
    yield
    # Remove any test prompts added during test
    for name in list(list_prompts()):
        if name not in original:
            del _REGISTRY[name]


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


# --- Test PromptResponsePair ---


class TestPromptResponsePair:
    """Tests for PromptResponsePair model."""

    def test_basic_creation(self) -> None:
        """Test creating a basic PromptResponsePair."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="What species?")],
            response="{{ species }}",
        )
        assert len(pair.messages) == 1
        assert pair.response == "{{ species }}"
        assert pair.task == "default"

    def test_with_task(self) -> None:
        """Test creating a PromptResponsePair with custom task."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="Identify the species")],
            response="{{ species_common }}",
            task="species_id",
        )
        assert pair.task == "species_id"

    def test_render(self) -> None:
        """Test rendering a pair with data."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="What is {{ field }}?")],
            response="{{ answer }}",
        )
        prompt_msgs, response = pair.render(field="the species", answer="Robin")
        assert len(prompt_msgs) == 1
        assert prompt_msgs[0] == {"role": "user", "content": "What is the species?"}
        assert response == "Robin"

    def test_multi_turn_conversation(self) -> None:
        """Test a multi-turn conversation pair."""
        pair = PromptResponsePair(
            messages=[
                Message(role="system", content="You are a bioacoustics expert."),
                Message(role="user", content="What species is in this audio?"),
            ],
            response="{{ species_common }}",
        )
        prompt_msgs, response = pair.render(species_common="Blue Jay")
        assert len(prompt_msgs) == 2
        assert prompt_msgs[0]["role"] == "system"
        assert prompt_msgs[1]["role"] == "user"
        assert response == "Blue Jay"


# --- Test PromptResponseTemplateConfig ---


class TestPromptResponseTemplateConfig:
    """Tests for PromptResponseTemplateConfig model."""

    def test_basic_creation(self) -> None:
        """Test creating a PromptResponseTemplateConfig."""
        config = PromptResponseTemplateConfig(
            name="test_prompt",
            variants=[
                PromptResponsePair(
                    messages=[Message(role="user", content="Question?")],
                    response="{{ answer }}",
                )
            ],
        )
        assert config.name == "test_prompt"
        assert len(config.variants) == 1

    def test_multiple_variants(self) -> None:
        """Test config with multiple variants."""
        config = PromptResponseTemplateConfig(
            name="multi_variant",
            variants=[
                PromptResponsePair(
                    messages=[Message(role="user", content="What species?")],
                    response="{{ species }}",
                ),
                PromptResponsePair(
                    messages=[Message(role="user", content="Identify the animal.")],
                    response="{{ species }}",
                ),
            ],
        )
        assert len(config.variants) == 2


# --- Test PromptResponseTemplate ---


class TestPromptResponseTemplate:
    """Tests for PromptResponseTemplate."""

    def test_basic_call(self) -> None:
        """Test basic template call adds prompt and response."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="What species?")],
            response="{{ species }}",
        )
        template = PromptResponseTemplate(name="test_basic", variants=[pair])
        item = {"species": "Robin", "extra": "data"}
        result = template(item)

        assert "prompt" in result
        assert "response" in result
        assert result["response"] == "Robin"
        assert result["extra"] == "data"  # Original data preserved

        # Prompt should be JSON of messages
        prompt_msgs = json.loads(result["prompt"])
        assert len(prompt_msgs) == 1
        assert prompt_msgs[0]["role"] == "user"
        assert prompt_msgs[0]["content"] == "What species?"

    def test_original_item_not_modified(self) -> None:
        """Test that original item is not modified."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="Q?")],
            response="{{ a }}",
        )
        template = PromptResponseTemplate(name="test_no_modify", variants=[pair])
        item = {"a": "answer"}
        result = template(item)

        assert "prompt" not in item
        assert "response" not in item
        assert "prompt" in result

    def test_multi_turn_prompt(self) -> None:
        """Test multi-turn conversation produces correct prompt."""
        pair = PromptResponsePair(
            messages=[
                Message(role="system", content="You are an expert."),
                Message(role="user", content="What is {{ thing }}?"),
            ],
            response="{{ answer }}",
        )
        template = PromptResponseTemplate(name="test_multi_turn", variants=[pair])
        result = template({"thing": "this", "answer": "A bird"})

        prompt_msgs = json.loads(result["prompt"])
        assert len(prompt_msgs) == 2
        assert prompt_msgs[0]["role"] == "system"
        assert prompt_msgs[1]["role"] == "user"
        assert result["response"] == "A bird"

    def test_single_variant_deterministic(self) -> None:
        """Test that single variant is deterministic."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="The only prompt")],
            response="{{ x }}",
        )
        template = PromptResponseTemplate(name="test_deterministic", variants=[pair])
        item = {"x": "value"}

        # Call multiple times - should always get same result
        for _ in range(10):
            result = template(item)
            prompt_msgs = json.loads(result["prompt"])
            assert prompt_msgs[0]["content"] == "The only prompt"

    def test_multiple_variants_with_seed(self) -> None:
        """Test that seed produces deterministic variant selection."""
        variants = [
            PromptResponsePair(
                messages=[Message(role="user", content="A")],
                response="{{ x }}",
            ),
            PromptResponsePair(
                messages=[Message(role="user", content="B")],
                response="{{ x }}",
            ),
            PromptResponsePair(
                messages=[Message(role="user", content="C")],
                response="{{ x }}",
            ),
        ]

        template1 = PromptResponseTemplate(name="test_seed", variants=variants, seed=12345)
        template2 = PromptResponseTemplate(name="test_seed", variants=variants, seed=12345)

        item = {"x": "value"}
        results1 = [json.loads(template1(item)["prompt"])[0]["content"] for _ in range(20)]
        results2 = [json.loads(template2(item)["prompt"])[0]["content"] for _ in range(20)]

        assert results1 == results2

    def test_multiple_variants_randomness(self) -> None:
        """Test that multiple variants are randomly selected without seed."""
        variants = [
            PromptResponsePair(
                messages=[Message(role="user", content="A")],
                response="{{ x }}",
            ),
            PromptResponsePair(
                messages=[Message(role="user", content="B")],
                response="{{ x }}",
            ),
            PromptResponsePair(
                messages=[Message(role="user", content="C")],
                response="{{ x }}",
            ),
        ]

        template = PromptResponseTemplate(name="test_random", variants=variants)
        item = {"x": "value"}

        # Collect prompts from multiple calls
        prompts = set()
        for _ in range(100):
            result = template(item)
            prompt_msgs = json.loads(result["prompt"])
            prompts.add(prompt_msgs[0]["content"])

        # Should see variety (though not guaranteed, very likely with 100 samples)
        assert len(prompts) > 1

    def test_jinja2_rendering(self) -> None:
        """Test Jinja2 template rendering in messages."""
        pair = PromptResponsePair(
            messages=[
                Message(
                    role="user",
                    content="Describe the {{ species }} from {{ location }}.",
                ),
            ],
            response="The {{ species }} is a bird found in {{ location }}.",
        )
        template = PromptResponseTemplate(name="test_jinja", variants=[pair])
        result = template({"species": "robin", "location": "California"})

        prompt_msgs = json.loads(result["prompt"])
        assert prompt_msgs[0]["content"] == "Describe the robin from California."
        assert result["response"] == "The robin is a bird found in California."

    def test_jinja2_conditionals(self) -> None:
        """Test Jinja2 conditional logic in templates."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="Describe this audio.")],
            response="{% if behavior == 'song' %}A singing {{ species }}{% else %}A {{ species }} {{ behavior }}{% endif %}",
        )
        template = PromptResponseTemplate(name="test_conditionals", variants=[pair])

        # Test "song" branch
        result = template({"behavior": "song", "species": "robin"})
        assert result["response"] == "A singing robin"

        # Test else branch
        result = template({"behavior": "call", "species": "jay"})
        assert result["response"] == "A jay call"

    def test_jinja2_filters(self) -> None:
        """Test Jinja2 built-in filters."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="What species?")],
            response="{{ species | upper }}",
        )
        template = PromptResponseTemplate(name="test_filters", variants=[pair])
        result = template({"species": "robin"})
        assert result["response"] == "ROBIN"

    def test_jinja2_loops(self) -> None:
        """Test Jinja2 loop constructs."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="List the species.")],
            response="{% for s in species %}{{ s }}{% if not loop.last %}, {% endif %}{% endfor %}",
        )
        template = PromptResponseTemplate(name="test_loops", variants=[pair])
        result = template({"species": ["robin", "jay", "cardinal"]})
        assert result["response"] == "robin, jay, cardinal"

    def test_empty_messages_raises(self) -> None:
        """Test that empty messages raises ValueError."""
        pair = PromptResponsePair(messages=[], response="answer")
        template = PromptResponseTemplate(name="test_empty", variants=[pair])

        with pytest.raises(ValueError, match="no messages"):
            template({})

    def test_missing_variable_raises(self) -> None:
        """Test that missing template variables raise UndefinedError."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="What is {{ missing }}?")],
            response="Answer",
        )
        template = PromptResponseTemplate(name="test_missing", variants=[pair])

        with pytest.raises(UndefinedError):
            template({"other": "value"})

    def test_missing_response_variable_raises(self) -> None:
        """Test that missing response template variables raise UndefinedError."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="Question")],
            response="{{ missing_var }}",
        )
        template = PromptResponseTemplate(name="test_missing_response", variants=[pair])

        with pytest.raises(UndefinedError):
            template({"other": "value"})

    def test_name_property(self) -> None:
        """Test that name property is set correctly."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="Q")],
            response="A",
        )
        template = PromptResponseTemplate(name="my_template", variants=[pair])
        assert template.name == "my_template"

    def test_accepts_list_of_variants(self) -> None:
        """Test that variants must be a list."""
        variants = [
            PromptResponsePair(
                messages=[Message(role="user", content="Question")],
                response="{{ answer }}",
            )
        ]
        template = PromptResponseTemplate(name="test_list", variants=variants)
        result = template({"answer": "Response"})

        assert result["response"] == "Response"


# --- Test Passthrough ---


class TestPassthrough:
    """Tests for passthrough template functionality."""

    def test_passthrough_registered(self) -> None:
        """Test that passthrough template is registered."""
        assert "passthrough" in list_prompts()

    def test_passthrough_passes_fields(self) -> None:
        """Test passthrough preserves prompt/response fields from source."""
        template = get_prompt("passthrough")
        item = {
            "prompt": "Original prompt",
            "response": "Original response",
            "other": "data",
        }
        result = template(item)

        # Passthrough renders {{ prompt }} and {{ response }} from input
        prompt_msgs = json.loads(result["prompt"])
        assert prompt_msgs[0]["content"] == "Original prompt"
        assert result["response"] == "Original response"
        assert result["other"] == "data"

    def test_passthrough_missing_fields_raises(self) -> None:
        """Test passthrough with missing fields raises UndefinedError.

        With StrictUndefined, missing variables raise clear errors.
        """
        template = get_prompt("passthrough")
        item = {"other": "data"}

        with pytest.raises(UndefinedError):
            template(item)


# --- Test Registry ---


class TestPromptRegistry:
    """Tests for prompt template registry."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving a template."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="Q")],
            response="{{ a }}",
        )
        template = PromptResponseTemplate(name="test_register_get", variants=[pair])
        register_prompt(template)

        retrieved = get_prompt("test_register_get")
        assert retrieved is template

    def test_register_duplicate_raises(self) -> None:
        """Test that registering duplicate name raises ValueError."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="Q")],
            response="{{ a }}",
        )
        register_prompt(PromptResponseTemplate(name="test_duplicate", variants=[pair]))

        with pytest.raises(ValueError, match="already registered"):
            register_prompt(PromptResponseTemplate(name="test_duplicate", variants=[pair]))

    def test_get_nonexistent_raises(self) -> None:
        """Test that getting nonexistent template raises KeyError."""
        with pytest.raises(KeyError, match="not registered"):
            get_prompt("nonexistent_template_xyz")

    def test_list_prompts(self) -> None:
        """Test listing registered prompts."""
        pair = PromptResponsePair(
            messages=[Message(role="user", content="Q")],
            response="{{ a }}",
        )
        register_prompt(PromptResponseTemplate(name="test_list_item", variants=[pair]))

        prompts = list_prompts()
        assert "test_list_item" in prompts
        assert "passthrough" in prompts

    def test_register_from_config(self) -> None:
        """Test registering from PromptResponseTemplateConfig."""
        from esp_data.prompts import register_prompt_from_config

        config = PromptResponseTemplateConfig(
            name="from_config_test",
            variants=[
                PromptResponsePair(
                    messages=[Message(role="user", content="Q")],
                    response="{{ a }}",
                )
            ],
        )
        template = register_prompt_from_config(config)

        assert template.name == "from_config_test"
        assert get_prompt("from_config_test") is template
