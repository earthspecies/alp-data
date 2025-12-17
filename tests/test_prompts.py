"""Tests for esp_data.prompts module."""

import pytest

from esp_data.prompts import (
    BasePromptTemplate,
    PassthroughTemplate,
    PromptTemplate,
    PromptVariant,
    clear_registry,
    get_prompt,
    is_registered,
    list_prompts,
    register_prompt,
    unregister_prompt,
)


# --- Fixtures ---


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean up registry before and after each test."""
    # Store original prompts
    original_prompts = list_prompts().copy()

    yield

    # Clean up any test prompts (keep passthrough)
    current = list_prompts()
    for name in current:
        if name not in original_prompts and name != "passthrough":
            try:
                unregister_prompt(name)
            except KeyError:
                pass


# --- Test BasePromptTemplate ---


class TestBasePromptTemplate:
    """Tests for BasePromptTemplate."""

    def test_default_prompt(self) -> None:
        """Test default prompt property."""

        class TestTemplate(BasePromptTemplate):
            name = "test_default"
            response_field = "text"

        template = TestTemplate()
        assert template.default_prompt == "Describe this audio."

    def test_custom_default_prompt(self) -> None:
        """Test custom default prompt."""

        class TestTemplate(BasePromptTemplate):
            name = "test_custom_default"
            response_field = "species"

            @property
            def default_prompt(self) -> str:
                return "What species is this?"

        template = TestTemplate()
        assert template.default_prompt == "What species is this?"

    def test_prompt_variants(self) -> None:
        """Test prompt variant selection."""

        class TestTemplate(BasePromptTemplate):
            name = "test_variants"
            response_field = "text"

        variants = ["Prompt A", "Prompt B", "Prompt C"]
        template = TestTemplate(prompt_variants=variants, seed=42)

        # With seed, should be deterministic
        item = {"text": "response"}
        prompts = [template.format_prompt(item) for _ in range(10)]
        assert all(p in variants for p in prompts)

    def test_prompt_format_substitution(self) -> None:
        """Test placeholder substitution in prompts."""

        class TestTemplate(BasePromptTemplate):
            name = "test_format"
            response_field = "text"

        template = TestTemplate(
            prompt_variants=["This is a {species} from {location}."],
        )
        item = {"species": "robin", "location": "California", "text": "response"}
        prompt = template.format_prompt(item)
        assert prompt == "This is a robin from California."

    def test_prompt_format_missing_field_raises(self) -> None:
        """Test that missing format fields raise KeyError."""

        class TestTemplate(BasePromptTemplate):
            name = "test_format_error"
            response_field = "text"

        template = TestTemplate(
            prompt_variants=["What is {missing_field}?"],
        )
        item = {"text": "response"}

        with pytest.raises(KeyError) as exc_info:
            template.format_prompt(item)

        assert "missing_field" in str(exc_info.value)

    def test_response_extraction(self) -> None:
        """Test response field extraction."""

        class TestTemplate(BasePromptTemplate):
            name = "test_response"
            response_field = "species_common"

        template = TestTemplate()
        item = {"species_common": "American Robin", "other_field": "ignored"}
        response = template.format_response(item)
        assert response == "American Robin"

    def test_response_with_prefix_suffix(self) -> None:
        """Test response with prefix and suffix."""

        class TestTemplate(BasePromptTemplate):
            name = "test_prefix_suffix"
            response_field = "species"

        template = TestTemplate(
            response_prefix="The species is: ",
            response_suffix=".",
        )
        item = {"species": "Robin"}
        response = template.format_response(item)
        assert response == "The species is: Robin."

    def test_response_missing_field(self) -> None:
        """Test response when field is missing."""

        class TestTemplate(BasePromptTemplate):
            name = "test_missing_response"
            response_field = "nonexistent"

        template = TestTemplate()
        item = {"other": "value"}
        response = template.format_response(item)
        assert response == ""

    def test_call_method(self) -> None:
        """Test __call__ method adds prompt and text keys."""

        class TestTemplate(BasePromptTemplate):
            name = "test_call"
            response_field = "answer"

            @property
            def default_prompt(self) -> str:
                return "Question?"

        template = TestTemplate()
        item = {"answer": "42", "extra": "data"}
        result = template(item)

        assert "prompt" in result
        assert "text" in result
        assert result["prompt"] == "Question?"
        assert result["text"] == "42"
        assert result["extra"] == "data"  # Original data preserved
        assert item.get("prompt") is None  # Original not modified

    def test_response_field_not_implemented(self) -> None:
        """Test that base class raises NotImplementedError for response_field."""
        template = BasePromptTemplate()

        with pytest.raises(NotImplementedError):
            _ = template.response_field

    def test_seed_determinism(self) -> None:
        """Test that seed produces deterministic results."""

        class TestTemplate(BasePromptTemplate):
            name = "test_seed"
            response_field = "text"

        variants = ["A", "B", "C", "D", "E"]

        template1 = TestTemplate(prompt_variants=variants, seed=12345)
        template2 = TestTemplate(prompt_variants=variants, seed=12345)

        item = {"text": "x"}
        results1 = [template1.format_prompt(item) for _ in range(20)]
        results2 = [template2.format_prompt(item) for _ in range(20)]

        assert results1 == results2


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


# --- Test Coupled Mode ---


class TestCoupledMode:
    """Tests for coupled prompt-response variants."""

    def test_coupled_variants_basic(self) -> None:
        """Test basic coupled variant functionality."""

        class CoupledTemplate(BasePromptTemplate):
            name = "test_coupled"

            def __init__(self, seed=None):
                super().__init__(
                    variants=[
                        PromptVariant("What species?", "{species_common}"),
                        PromptVariant("Scientific name?", "{species_scientific}"),
                    ],
                    seed=seed,
                )

        template = CoupledTemplate(seed=42)
        item = {
            "species_common": "American Robin",
            "species_scientific": "Turdus migratorius",
        }

        # With seed=42, first choice should be deterministic
        result = template(item)
        assert result["prompt"] in ["What species?", "Scientific name?"]
        # Response should match the selected prompt
        if result["prompt"] == "What species?":
            assert result["text"] == "American Robin"
        else:
            assert result["text"] == "Turdus migratorius"

    def test_coupled_prompt_response_consistency(self) -> None:
        """Test that prompt and response are coupled correctly."""

        class CoupledTemplate(BasePromptTemplate):
            name = "test_coupled_consistency"

            def __init__(self, seed=None):
                super().__init__(
                    variants=[
                        PromptVariant("Common name?", "{common}"),
                        PromptVariant("Scientific name?", "{scientific}"),
                        PromptVariant("Both names?", "{common} ({scientific})"),
                    ],
                    seed=seed,
                )

        template = CoupledTemplate(seed=123)
        item = {"common": "Robin", "scientific": "Turdus"}

        # Run multiple times to test consistency
        for _ in range(10):
            result = template(item)
            prompt = result["prompt"]
            text = result["text"]

            if prompt == "Common name?":
                assert text == "Robin"
            elif prompt == "Scientific name?":
                assert text == "Turdus"
            else:
                assert prompt == "Both names?"
                assert text == "Robin (Turdus)"

    def test_coupled_with_prefix_suffix(self) -> None:
        """Test coupled mode with response prefix/suffix."""

        class CoupledTemplate(BasePromptTemplate):
            name = "test_coupled_affix"

            def __init__(self, seed=None):
                super().__init__(
                    variants=[
                        PromptVariant("What is it?", "{answer}"),
                    ],
                    response_prefix="The answer is: ",
                    response_suffix="!",
                    seed=seed,
                )

        template = CoupledTemplate()
        item = {"answer": "Robin"}
        result = template(item)
        assert result["text"] == "The answer is: Robin!"

    def test_coupled_response_missing_field_raises(self) -> None:
        """Test that missing response field raises KeyError in coupled mode."""

        class CoupledTemplate(BasePromptTemplate):
            name = "test_coupled_error"

            def __init__(self, seed=None):
                super().__init__(
                    variants=[
                        PromptVariant("Question?", "{missing_field}"),
                    ],
                    seed=seed,
                )

        template = CoupledTemplate()
        item = {"other": "value"}

        with pytest.raises(KeyError) as exc_info:
            template(item)

        assert "missing_field" in str(exc_info.value)

    def test_cannot_mix_modes(self) -> None:
        """Test that providing both prompt_variants and variants raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            BasePromptTemplate(
                prompt_variants=["Question?"],
                variants=[PromptVariant("Question?", "{answer}")],
            )

        assert "Cannot specify both" in str(exc_info.value)

    def test_variants_property(self) -> None:
        """Test the variants property returns configured variants."""

        class CoupledTemplate(BasePromptTemplate):
            name = "test_variants_prop"

            def __init__(self):
                super().__init__(
                    variants=[
                        PromptVariant("Q1", "{a}"),
                        PromptVariant("Q2", "{b}"),
                    ],
                )

        template = CoupledTemplate()
        assert len(template.variants) == 2
        assert template.variants[0].prompt == "Q1"
        assert template.variants[1].response == "{b}"

    def test_prompt_variants_property_backwards_compat(self) -> None:
        """Test prompt_variants property returns prompt strings."""

        class CoupledTemplate(BasePromptTemplate):
            name = "test_prompt_variants_compat"

            def __init__(self):
                super().__init__(
                    variants=[
                        PromptVariant("Question A", "{a}"),
                        PromptVariant("Question B", "{b}"),
                    ],
                )

        template = CoupledTemplate()
        assert template.prompt_variants == ["Question A", "Question B"]

    def test_coupled_seed_determinism(self) -> None:
        """Test that seed produces deterministic coupled results."""

        class CoupledTemplate(BasePromptTemplate):
            name = "test_coupled_seed"

            def __init__(self, seed=None):
                super().__init__(
                    variants=[
                        PromptVariant("A?", "{x}"),
                        PromptVariant("B?", "{y}"),
                        PromptVariant("C?", "{z}"),
                    ],
                    seed=seed,
                )

        template1 = CoupledTemplate(seed=99999)
        template2 = CoupledTemplate(seed=99999)

        item = {"x": "X", "y": "Y", "z": "Z"}
        results1 = [template1(item) for _ in range(20)]
        results2 = [template2(item) for _ in range(20)]

        for r1, r2 in zip(results1, results2):
            assert r1["prompt"] == r2["prompt"]
            assert r1["text"] == r2["text"]


# --- Test PassthroughTemplate ---


class TestPassthroughTemplate:
    """Tests for PassthroughTemplate."""

    def test_passthrough_prompt(self) -> None:
        """Test passthrough passes prompt field."""
        template = PassthroughTemplate()
        item = {"prompt": "Original prompt", "text": "Original text"}
        result = template(item)
        assert result["prompt"] == "Original prompt"
        assert result["text"] == "Original text"

    def test_passthrough_custom_fields(self) -> None:
        """Test passthrough with custom field names."""
        template = PassthroughTemplate(prompt_field="instruction", text_field="response")
        item = {"instruction": "Do this", "response": "Done"}
        result = template(item)
        assert result["prompt"] == "Do this"
        assert result["text"] == "Done"

    def test_passthrough_missing_fields(self) -> None:
        """Test passthrough with missing fields returns empty strings."""
        template = PassthroughTemplate()
        item = {"other": "data"}
        result = template(item)
        assert result["prompt"] == ""
        assert result["text"] == ""

    def test_passthrough_name(self) -> None:
        """Test passthrough template name."""
        template = PassthroughTemplate()
        assert template.name == "passthrough"


# --- Test Registry ---


class TestPromptRegistry:
    """Tests for prompt template registry."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving a template."""

        class TestTemplate(BasePromptTemplate):
            name = "test_register_get"
            response_field = "x"

        template = TestTemplate()
        register_prompt(template)

        retrieved = get_prompt("test_register_get")
        assert retrieved is template

    def test_register_duplicate_raises(self) -> None:
        """Test that registering duplicate name raises ValueError."""

        class TestTemplate(BasePromptTemplate):
            name = "test_duplicate"
            response_field = "x"

        register_prompt(TestTemplate())

        with pytest.raises(ValueError) as exc_info:
            register_prompt(TestTemplate())

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
            response_field = "x"

        register_prompt(TestTemplate())

        prompts = list_prompts()
        assert "test_list_item" in prompts
        assert "passthrough" in prompts  # Built-in

    def test_is_registered(self) -> None:
        """Test is_registered function."""
        assert is_registered("passthrough")
        assert not is_registered("definitely_not_registered_xyz")

    def test_unregister(self) -> None:
        """Test unregistering a template."""

        class TestTemplate(BasePromptTemplate):
            name = "test_unregister"
            response_field = "x"

        register_prompt(TestTemplate())
        assert is_registered("test_unregister")

        unregister_prompt("test_unregister")
        assert not is_registered("test_unregister")

    def test_unregister_nonexistent_raises(self) -> None:
        """Test that unregistering nonexistent template raises KeyError."""
        with pytest.raises(KeyError):
            unregister_prompt("nonexistent_xyz")

    def test_passthrough_registered_by_default(self) -> None:
        """Test that passthrough template is registered by default."""
        assert is_registered("passthrough")
        template = get_prompt("passthrough")
        assert isinstance(template, PassthroughTemplate)


# --- Test Protocol ---


class TestPromptTemplateProtocol:
    """Tests for PromptTemplate protocol compliance."""

    def test_base_implements_protocol(self) -> None:
        """Test that BasePromptTemplate implements PromptTemplate protocol."""

        class TestTemplate(BasePromptTemplate):
            name = "test_protocol"
            response_field = "x"

        template = TestTemplate()
        assert isinstance(template, PromptTemplate)

    def test_passthrough_implements_protocol(self) -> None:
        """Test that PassthroughTemplate implements PromptTemplate protocol."""
        template = PassthroughTemplate()
        assert isinstance(template, PromptTemplate)
