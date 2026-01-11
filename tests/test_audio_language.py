"""Tests for esp_data.transforms.audio_language module."""

import json

import polars as pl
import pytest

from esp_data.backends import PolarsBackend
from esp_data.prompts import (
    Message,
    PromptResponsePair,
    PromptTemplate,
    PromptTemplateConfig,
    register_prompt,
)
from esp_data.prompts.registry import _REGISTRY
from esp_data.transforms import AudioLanguage, AudioLanguageConfig


# --- Fixtures ---


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean up registry before and after each test."""
    original = set(_REGISTRY.keys())
    yield
    # Remove any test prompts added during test
    for name in list(_REGISTRY.keys()):
        if name not in original:
            del _REGISTRY[name]


@pytest.fixture
def species_prompt_config() -> PromptTemplateConfig:
    """Create a species identification prompt config."""
    return PromptTemplateConfig(
        name="test_species",
        variants=[
            PromptResponsePair(
                messages=[
                    Message(role="user", content="What species is this?"),
                    Message(role="assistant", content="{{ species_common }}"),
                ],
                task="species_id",
            ),
            PromptResponsePair(
                messages=[
                    Message(role="user", content="Identify the species."),
                    Message(role="assistant", content="{{ species_common }}"),
                ],
                task="species_id",
            ),
        ],
    )


@pytest.fixture
def registered_species_prompt(species_prompt_config) -> PromptTemplate:
    """Register and return a species prompt template."""
    template = PromptTemplate(
        name=species_prompt_config.name,
        variants=species_prompt_config.variants,
        seed=42,
    )
    try:
        register_prompt(template)
    except ValueError:
        pass  # Already registered
    return template


@pytest.fixture
def mock_backend() -> PolarsBackend:
    """Create a mock backend with sample data."""
    df = pl.DataFrame(
        {
            "audio_path": [
                "/path/to/audio1.wav",
                "/path/to/audio2.wav",
                "/path/to/audio3.wav",
            ],
            "species_common": ["American Robin", "Blue Jay", "Cardinal"],
            "behavior": ["song", "call", "alarm"],
            "duration": [5.0, 3.2, 4.1],
        }
    )
    return PolarsBackend(df)


@pytest.fixture
def native_al_backend() -> PolarsBackend:
    """Create a backend that already has prompt/response fields (native AL dataset)."""
    df = pl.DataFrame(
        {
            "audio_path": ["/path/to/audio1.wav", "/path/to/audio2.wav"],
            "prompt": ["Describe this sound.", "What is this?"],
            "response": ["A bird singing.", "An alarm call."],
        }
    )
    return PolarsBackend(df)


# --- Test AudioLanguageConfig ---


class TestAudioLanguageConfig:
    """Tests for AudioLanguageConfig model."""

    def test_with_prompt_name(self) -> None:
        """Test config with prompt name string."""
        config = AudioLanguageConfig(type="audio_language", prompt="species_common")
        assert config.type == "audio_language"
        assert config.prompt == "species_common"

    def test_with_prompt_config(self) -> None:
        """Test config with inline PromptTemplateConfig."""
        prompt_config = PromptTemplateConfig(
            name="inline_prompt",
            variants=[
                PromptResponsePair(
                    messages=[
                        Message(role="user", content="Q?"),
                        Message(role="assistant", content="{{ a }}"),
                    ]
                )
            ],
        )
        config = AudioLanguageConfig(type="audio_language", prompt=prompt_config)
        assert config.prompt == prompt_config

    def test_with_prompt_none(self) -> None:
        """Test config with prompt=None defaults to passthrough."""
        config = AudioLanguageConfig(type="audio_language", prompt=None)
        assert config.prompt is None

    def test_with_seed(self) -> None:
        """Test config with seed for stochastic variant selection."""
        config = AudioLanguageConfig(type="audio_language", prompt="test", seed=42)
        assert config.seed == 42


# --- Test AudioLanguage Transform ---


class TestAudioLanguage:
    """Tests for AudioLanguage transform."""

    def test_init_with_prompt_name(self, registered_species_prompt) -> None:
        """Test initialization with registered prompt name."""
        transform = AudioLanguage(prompt="test_species")
        assert transform.template.name == "test_species"

    def test_init_with_prompt_config(self) -> None:
        """Test initialization with PromptTemplateConfig."""
        config = PromptTemplateConfig(
            name="direct_config",
            variants=[
                PromptResponsePair(
                    messages=[
                        Message(role="user", content="What?"),
                        Message(role="assistant", content="{{ answer }}"),
                    ]
                )
            ],
        )
        transform = AudioLanguage(prompt=config)
        assert transform.template.name == "direct_config"

    def test_init_with_none_uses_passthrough(self) -> None:
        """Test that prompt=None uses passthrough template."""
        transform = AudioLanguage(prompt=None)
        assert transform.template.name == "passthrough"

    def test_init_default_is_passthrough(self) -> None:
        """Test that default (no args) uses passthrough template."""
        transform = AudioLanguage()
        assert transform.template.name == "passthrough"

    def test_from_config(self, registered_species_prompt) -> None:
        """Test creating transform from AudioLanguageConfig."""
        config = AudioLanguageConfig(type="audio_language", prompt="test_species")
        transform = AudioLanguage.from_config(config)
        assert transform.template.name == "test_species"

    def test_from_config_with_none(self) -> None:
        """Test from_config with prompt=None uses passthrough."""
        config = AudioLanguageConfig(type="audio_language", prompt=None)
        transform = AudioLanguage.from_config(config)
        assert transform.template.name == "passthrough"

    def test_unknown_prompt_raises(self) -> None:
        """Test that unknown prompt name raises KeyError."""
        with pytest.raises(KeyError):
            AudioLanguage(prompt="nonexistent_prompt_xyz")

    def test_call_adds_columns(
        self, mock_backend, registered_species_prompt
    ) -> None:
        """Test that transform adds prompt and response columns."""
        transform = AudioLanguage(prompt="test_species")
        new_backend, metadata = transform(mock_backend)

        assert "prompt" in new_backend.columns
        assert "response" in new_backend.columns
        # Original columns preserved
        assert "audio_path" in new_backend.columns
        assert "species_common" in new_backend.columns

    def test_call_returns_metadata(
        self, mock_backend, registered_species_prompt
    ) -> None:
        """Test that transform returns correct metadata."""
        transform = AudioLanguage(prompt="test_species")
        _, metadata = transform(mock_backend)

        assert metadata["prompt_template"] == "test_species"
        assert metadata["num_rows"] == 3

    def test_prompt_content(self, mock_backend, registered_species_prompt) -> None:
        """Test that prompt column contains correct JSON structure."""
        transform = AudioLanguage(prompt="test_species")
        new_backend, _ = transform(mock_backend)

        row = new_backend[0]
        prompt_msgs = json.loads(row["prompt"])

        assert len(prompt_msgs) == 1
        assert prompt_msgs[0]["role"] == "user"
        assert prompt_msgs[0]["content"] in [
            "What species is this?",
            "Identify the species.",
        ]

    def test_response_content(self, mock_backend, registered_species_prompt) -> None:
        """Test that response column contains rendered response."""
        transform = AudioLanguage(prompt="test_species")
        new_backend, _ = transform(mock_backend)

        # Check each row has correct species in response
        expected_species = ["American Robin", "Blue Jay", "Cardinal"]
        for i, species in enumerate(expected_species):
            row = new_backend[i]
            assert row["response"] == species

    def test_preserves_row_count(
        self, mock_backend, registered_species_prompt
    ) -> None:
        """Test that transform preserves row count."""
        transform = AudioLanguage(prompt="test_species")
        new_backend, _ = transform(mock_backend)

        assert len(new_backend) == len(mock_backend)

    def test_passthrough_template(self, native_al_backend) -> None:
        """Test passthrough template with native AL data."""
        transform = AudioLanguage(prompt="passthrough")
        new_backend, metadata = transform(native_al_backend)

        row = new_backend[0]
        prompt_msgs = json.loads(row["prompt"])

        # Passthrough renders the original prompt field into user message
        assert prompt_msgs[0]["content"] == "Describe this sound."
        assert row["response"] == "A bird singing."

    def test_passthrough_default(self, native_al_backend) -> None:
        """Test that prompt=None uses passthrough."""
        transform = AudioLanguage()  # No prompt specified
        new_backend, _ = transform(native_al_backend)

        row = new_backend[0]
        prompt_msgs = json.loads(row["prompt"])
        assert prompt_msgs[0]["content"] == "Describe this sound."
        assert row["response"] == "A bird singing."

    def test_multi_turn_prompt(self, mock_backend) -> None:
        """Test transform with multi-turn conversation prompt."""
        pair = PromptResponsePair(
            messages=[
                Message(role="system", content="You are a bioacoustics expert."),
                Message(role="user", content="What species is in this audio?"),
                Message(role="assistant", content="{{ species_common }}"),
            ]
        )
        template = PromptTemplate(name="multi_turn_test", variants=pair)
        register_prompt(template)

        transform = AudioLanguage(prompt="multi_turn_test")
        new_backend, _ = transform(mock_backend)

        row = new_backend[0]
        prompt_msgs = json.loads(row["prompt"])

        assert len(prompt_msgs) == 2
        assert prompt_msgs[0]["role"] == "system"
        assert prompt_msgs[1]["role"] == "user"
        assert row["response"] == "American Robin"

    def test_iteration_over_backend(
        self, mock_backend, registered_species_prompt
    ) -> None:
        """Test that transform correctly iterates over all backend rows."""
        transform = AudioLanguage(prompt="test_species")
        new_backend, _ = transform(mock_backend)

        rows = list(new_backend)
        assert len(rows) == 3

        for row in rows:
            assert "prompt" in row
            assert "response" in row
            assert row["response"] in ["American Robin", "Blue Jay", "Cardinal"]


# --- Test Integration with Backend Operations ---


class TestAudioLanguageIntegration:
    """Integration tests for AudioLanguage with backend operations."""

    def test_chain_with_filter(self, mock_backend, registered_species_prompt) -> None:
        """Test chaining AudioLanguage after a filter operation."""
        from esp_data.transforms import Filter

        # First filter to only Robin
        filter_transform = Filter(
            property="species_common", values=["American Robin"], mode="include"
        )
        filtered_backend, _ = filter_transform(mock_backend)

        # Then apply audio language transform
        al_transform = AudioLanguage(prompt="test_species")
        new_backend, metadata = al_transform(filtered_backend)

        assert len(new_backend) == 1
        assert new_backend[0]["response"] == "American Robin"

    def test_empty_backend(self, registered_species_prompt) -> None:
        """Test transform handles empty backend."""
        empty_df = pl.DataFrame(
            {
                "audio_path": [],
                "species_common": [],
            }
        )
        empty_backend = PolarsBackend(empty_df)

        transform = AudioLanguage(prompt="test_species")
        new_backend, metadata = transform(empty_backend)

        assert len(new_backend) == 0
        assert metadata["num_rows"] == 0
        assert "prompt" in new_backend.columns
        assert "response" in new_backend.columns

    def test_with_pandas_backend(self, registered_species_prompt) -> None:
        """Test transform works with pandas backend."""
        import pandas as pd

        from esp_data.backends import PandasBackend

        df = pd.DataFrame(
            {
                "audio_path": ["/path/to/audio.wav"],
                "species_common": ["Test Species"],
            }
        )
        backend = PandasBackend(df)

        transform = AudioLanguage(prompt="test_species")
        new_backend, _ = transform(backend)

        assert "prompt" in new_backend.columns
        assert new_backend[0]["response"] == "Test Species"


# --- Test Complex Prompts ---


class TestComplexPrompts:
    """Tests for complex prompt scenarios."""

    def test_prompt_with_multiple_fields(self, mock_backend) -> None:
        """Test prompt that uses multiple fields from data."""
        pair = PromptResponsePair(
            messages=[
                Message(
                    role="user",
                    content="What {{ behavior }} is the {{ species_common }} making?",
                ),
                Message(
                    role="assistant",
                    content="The {{ species_common }} is making a {{ behavior }}.",
                ),
            ]
        )
        template = PromptTemplate(name="multi_field_test", variants=pair)
        register_prompt(template)

        transform = AudioLanguage(prompt="multi_field_test")
        new_backend, _ = transform(mock_backend)

        row = new_backend[0]
        prompt_msgs = json.loads(row["prompt"])

        assert "song" in prompt_msgs[0]["content"]
        assert "American Robin" in prompt_msgs[0]["content"]
        assert row["response"] == "The American Robin is making a song."

    def test_prompt_with_conditional_jinja(self, mock_backend) -> None:
        """Test prompt with Jinja2 conditional logic."""
        pair = PromptResponsePair(
            messages=[
                Message(role="user", content="Describe this audio."),
                Message(
                    role="assistant",
                    content="{% if behavior == 'song' %}A singing {{ species_common }}{% else %}A {{ species_common }} {{ behavior }}{% endif %}",
                ),
            ]
        )
        template = PromptTemplate(name="conditional_test", variants=pair)
        register_prompt(template)

        transform = AudioLanguage(prompt="conditional_test")
        new_backend, _ = transform(mock_backend)

        # First row has behavior="song"
        assert new_backend[0]["response"] == "A singing American Robin"
        # Second row has behavior="call"
        assert new_backend[1]["response"] == "A Blue Jay call"

    def test_stochastic_variant_selection(self, mock_backend) -> None:
        """Test that PromptTemplate with seed produces deterministic results."""
        config = PromptTemplateConfig(
            name="stochastic_test",
            variants=[
                PromptResponsePair(
                    messages=[
                        Message(role="user", content="Variant A"),
                        Message(role="assistant", content="{{ species_common }}"),
                    ]
                ),
                PromptResponsePair(
                    messages=[
                        Message(role="user", content="Variant B"),
                        Message(role="assistant", content="{{ species_common }}"),
                    ]
                ),
            ],
        )

        # Run 1: Create fresh template with seed
        template1 = PromptTemplate(name=config.name, variants=config.variants, seed=42)
        register_prompt(template1)
        transform1 = AudioLanguage(prompt="stochastic_test")
        backend1, _ = transform1(mock_backend)

        # Clean up registry
        del _REGISTRY["stochastic_test"]

        # Run 2: Create fresh template with same seed
        template2 = PromptTemplate(name=config.name, variants=config.variants, seed=42)
        register_prompt(template2)
        transform2 = AudioLanguage(prompt="stochastic_test")
        backend2, _ = transform2(mock_backend)

        # Results should be identical with same seed
        for i in range(len(mock_backend)):
            assert json.loads(backend1[i]["prompt"]) == json.loads(
                backend2[i]["prompt"]
            )

    def test_inline_config_uses_template(self, mock_backend) -> None:
        """Test that inline PromptTemplateConfig creates PromptTemplate."""
        config = PromptTemplateConfig(
            name="inline_test",
            variants=[
                PromptResponsePair(
                    messages=[
                        Message(role="user", content="Variant A"),
                        Message(role="assistant", content="{{ species_common }}"),
                    ]
                ),
                PromptResponsePair(
                    messages=[
                        Message(role="user", content="Variant B"),
                        Message(role="assistant", content="{{ species_common }}"),
                    ]
                ),
            ],
        )

        # With seed, should be deterministic
        transform1 = AudioLanguage(prompt=config, seed=123)
        backend1, _ = transform1(mock_backend)

        transform2 = AudioLanguage(prompt=config, seed=123)
        backend2, _ = transform2(mock_backend)

        for i in range(len(mock_backend)):
            assert json.loads(backend1[i]["prompt"]) == json.loads(
                backend2[i]["prompt"]
            )
