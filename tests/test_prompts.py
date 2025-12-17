"""Tests for esp_data.prompts module."""

import pytest

from esp_data.prompts import (
    BasePromptTemplate,
    PromptTemplate,
    PromptVariant,
    get_prompt,
    list_prompts,
    register_prompt,
)
from esp_data.prompts.registry import _PROMPT_REGISTRY


# --- Fixtures ---


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean up registry before and after each test."""
    original = set(list_prompts())
    yield
    # Remove any test prompts added during test
    for name in list(list_prompts()):
        if name not in original:
            del _PROMPT_REGISTRY[name]


# --- Test PromptVariant ---


class TestPromptVariant:
    """Tests for PromptVariant dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic PromptVariant."""
        variant = PromptVariant(prompt="What species?", response="{species}")
        assert variant.prompt == "What species?"
        assert variant.response == "{species}"
        assert variant.style == "default"
        assert variant.metadata == {}

    def test_with_style(self) -> None:
        """Test creating a PromptVariant with style."""
        variant = PromptVariant(
            prompt="Identify the species",
            response="{species_common}",
            style="formal",
        )
        assert variant.style == "formal"

    def test_with_metadata(self) -> None:
        """Test creating a PromptVariant with metadata."""
        variant = PromptVariant(
            prompt="What bird?",
            response="{species}",
            metadata={"difficulty": "easy", "domain": "birds"},
        )
        assert variant.metadata["difficulty"] == "easy"
        assert variant.metadata["domain"] == "birds"


# --- Test BasePromptTemplate ---


class TestBasePromptTemplate:
    """Tests for BasePromptTemplate."""

    def test_default_variant(self) -> None:
        """Test default variant when none provided."""
        template = BasePromptTemplate()
        assert len(template.variants) == 1
        assert template.variants[0].prompt == "Describe this audio."

    def test_custom_variants(self) -> None:
        """Test custom variants."""
        template = BasePromptTemplate(
            variants=[
                PromptVariant("What species?", "{species}"),
                PromptVariant("Identify.", "{species}"),
            ]
        )
        assert len(template.variants) == 2

    def test_variant_selection(self) -> None:
        """Test random variant selection."""
        template = BasePromptTemplate(
            variants=[
                PromptVariant("A", "{x}"),
                PromptVariant("B", "{x}"),
                PromptVariant("C", "{x}"),
            ],
            seed=42,
        )

        item = {"x": "value"}
        prompts = [template.format_prompt(item) for _ in range(10)]
        assert all(p in ["A", "B", "C"] for p in prompts)

    def test_prompt_format_substitution(self) -> None:
        """Test placeholder substitution in prompts."""
        template = BasePromptTemplate(
            variants=[
                PromptVariant("This is a {species} from {location}.", "{species}")
            ]
        )
        item = {"species": "robin", "location": "California"}
        prompt = template.format_prompt(item)
        assert prompt == "This is a robin from California."

    def test_prompt_format_missing_field_raises(self) -> None:
        """Test that missing format fields raise KeyError."""
        template = BasePromptTemplate(
            variants=[PromptVariant("What is {missing_field}?", "{x}")]
        )
        item = {"x": "value"}

        with pytest.raises(KeyError) as exc_info:
            template.format_prompt(item)

        assert "missing_field" in str(exc_info.value)

    def test_response_formatting(self) -> None:
        """Test response formatting."""
        template = BasePromptTemplate(
            variants=[PromptVariant("Question?", "{answer}")]
        )
        item = {"answer": "42"}
        template.format_prompt(item)  # Select variant
        response = template.format_response(item)
        assert response == "42"

    def test_response_with_prefix_suffix(self) -> None:
        """Test response with prefix and suffix."""
        template = BasePromptTemplate(
            variants=[PromptVariant("Question?", "{answer}")],
            response_prefix="Answer: ",
            response_suffix="!",
        )
        item = {"answer": "Robin"}
        template.format_prompt(item)
        response = template.format_response(item)
        assert response == "Answer: Robin!"

    def test_response_missing_field_raises(self) -> None:
        """Test response raises KeyError when field is missing."""
        template = BasePromptTemplate(
            variants=[PromptVariant("Question?", "{missing}")]
        )
        item = {"other": "value"}
        template.format_prompt(item)

        with pytest.raises(KeyError) as exc_info:
            template.format_response(item)

        assert "missing" in str(exc_info.value)

    def test_call_method(self) -> None:
        """Test __call__ method adds prompt and text keys."""
        template = BasePromptTemplate(
            variants=[PromptVariant("Question?", "{answer}")]
        )
        item = {"answer": "42", "extra": "data"}
        result = template(item)

        assert "prompt" in result
        assert "text" in result
        assert result["prompt"] == "Question?"
        assert result["text"] == "42"
        assert result["extra"] == "data"
        assert item.get("prompt") is None  # Original not modified

    def test_seed_determinism(self) -> None:
        """Test that seed produces deterministic results."""
        variants = [
            PromptVariant("A", "{x}"),
            PromptVariant("B", "{x}"),
            PromptVariant("C", "{x}"),
        ]

        template1 = BasePromptTemplate(variants=variants, seed=12345)
        template2 = BasePromptTemplate(variants=variants, seed=12345)

        item = {"x": "value"}
        results1 = [template1.format_prompt(item) for _ in range(20)]
        results2 = [template2.format_prompt(item) for _ in range(20)]

        assert results1 == results2

    def test_coupled_prompt_response(self) -> None:
        """Test that prompt and response are coupled correctly."""
        template = BasePromptTemplate(
            variants=[
                PromptVariant("Common name?", "{common}"),
                PromptVariant("Scientific name?", "{scientific}"),
                PromptVariant("Both?", "{common} ({scientific})"),
            ],
            seed=123,
        )
        item = {"common": "Robin", "scientific": "Turdus"}

        for _ in range(10):
            result = template(item)
            prompt = result["prompt"]
            text = result["text"]

            if prompt == "Common name?":
                assert text == "Robin"
            elif prompt == "Scientific name?":
                assert text == "Turdus"
            else:
                assert prompt == "Both?"
                assert text == "Robin (Turdus)"

    def test_variants_property(self) -> None:
        """Test the variants property returns configured variants."""
        template = BasePromptTemplate(
            variants=[
                PromptVariant("Q1", "{a}"),
                PromptVariant("Q2", "{b}"),
            ]
        )
        assert len(template.variants) == 2
        assert template.variants[0].prompt == "Q1"
        assert template.variants[1].response == "{b}"


# --- Test Passthrough ---


class TestPassthrough:
    """Tests for passthrough template functionality."""

    def test_passthrough_registered(self) -> None:
        """Test that passthrough template is registered."""
        assert "passthrough" in list_prompts()

    def test_passthrough_passes_fields(self) -> None:
        """Test passthrough passes prompt/text fields."""
        template = get_prompt("passthrough")
        item = {"prompt": "Original prompt", "text": "Original text", "other": "data"}
        result = template(item)
        assert result["prompt"] == "Original prompt"
        assert result["text"] == "Original text"
        assert result["other"] == "data"

    def test_passthrough_missing_fields_raises(self) -> None:
        """Test passthrough with missing fields raises KeyError."""
        template = get_prompt("passthrough")
        item = {"other": "data"}

        with pytest.raises(KeyError):
            template(item)

    def test_custom_passthrough(self) -> None:
        """Test creating passthrough with custom field names."""
        template = BasePromptTemplate(
            variants=[PromptVariant("{instruction}", "{response}")]
        )
        item = {"instruction": "Do this", "response": "Done"}
        result = template(item)
        assert result["prompt"] == "Do this"
        assert result["text"] == "Done"


# --- Test Registry ---


class TestPromptRegistry:
    """Tests for prompt template registry."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving a template."""

        class TestTemplate(BasePromptTemplate):
            name = "test_register_get"

        template = TestTemplate(variants=[PromptVariant("Q", "{a}")])
        register_prompt(template)

        retrieved = get_prompt("test_register_get")
        assert retrieved is template

    def test_register_duplicate_raises(self) -> None:
        """Test that registering duplicate name raises ValueError."""

        class TestTemplate(BasePromptTemplate):
            name = "test_duplicate"

        register_prompt(TestTemplate(variants=[PromptVariant("Q", "{a}")]))

        with pytest.raises(ValueError) as exc_info:
            register_prompt(TestTemplate(variants=[PromptVariant("Q", "{a}")]))

        assert "already registered" in str(exc_info.value)

    def test_get_nonexistent_raises(self) -> None:
        """Test that getting nonexistent template raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            get_prompt("nonexistent_template_xyz")

        assert "not registered" in str(exc_info.value)

    def test_list_prompts(self) -> None:
        """Test listing registered prompts."""

        class TestTemplate(BasePromptTemplate):
            name = "test_list_item"

        register_prompt(TestTemplate(variants=[PromptVariant("Q", "{a}")]))

        prompts = list_prompts()
        assert "test_list_item" in prompts
        assert "passthrough" in prompts


# --- Test Protocol ---


class TestPromptTemplateProtocol:
    """Tests for PromptTemplate protocol compliance."""

    def test_base_implements_protocol(self) -> None:
        """Test that BasePromptTemplate implements PromptTemplate protocol."""

        class TestTemplate(BasePromptTemplate):
            name = "test_protocol"

        template = TestTemplate(variants=[PromptVariant("Q", "{a}")])
        assert isinstance(template, PromptTemplate)
