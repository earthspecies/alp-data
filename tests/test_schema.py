"""Tests for schema validation."""

import pytest
from pydantic import ValidationError

from esp_data.backends.pandas_backend import PandasBackend
from esp_data.backends.polars_backend import PolarsBackend
from esp_data.schema import ColumnSchema, DatasetSchema, SchemaValidationError


class TestColumnSchema:
    """Tests for ColumnSchema."""

    def test_column_schema_creation(self) -> None:
        """Test creating a ColumnSchema."""
        col = ColumnSchema(name="test_col", dtype="str", required=True)
        assert col.name == "test_col"
        assert col.dtype == "str"
        assert col.required is True

    def test_column_schema_defaults(self) -> None:
        """Test ColumnSchema default values."""
        col = ColumnSchema(name="test_col", dtype="int")
        assert col.required is True  # Default is True

    def test_column_schema_optional(self) -> None:
        """Test creating an optional ColumnSchema."""
        col = ColumnSchema(name="test_col", dtype="float", required=False)
        assert col.required is False

    def test_column_schema_frozen(self) -> None:
        """Test that ColumnSchema is immutable (frozen)."""
        col = ColumnSchema(name="test_col", dtype="str")
        with pytest.raises(ValidationError):
            col.name = "new_name"


class TestDatasetSchema:
    """Tests for DatasetSchema."""

    def test_schema_creation(self) -> None:
        """Test creating a DatasetSchema."""
        schema = DatasetSchema(
            columns=[
                ColumnSchema(name="col1", dtype="str"),
                ColumnSchema(name="col2", dtype="int"),
            ]
        )
        assert len(schema.columns) == 2

    def test_required_columns(self) -> None:
        """Test getting required columns."""
        schema = DatasetSchema(
            columns=[
                ColumnSchema(name="col1", dtype="str", required=True),
                ColumnSchema(name="col2", dtype="int", required=False),
                ColumnSchema(name="col3", dtype="float", required=True),
            ]
        )
        assert schema.required_columns == ["col1", "col3"]

    def test_column_names(self) -> None:
        """Test getting all column names."""
        schema = DatasetSchema(
            columns=[
                ColumnSchema(name="col1", dtype="str"),
                ColumnSchema(name="col2", dtype="int"),
            ]
        )
        assert schema.column_names == ["col1", "col2"]


class TestSchemaValidationPolars:
    """Tests for schema validation with Polars backend."""

    def test_valid_schema(self) -> None:
        """Test validation passes for matching schema."""
        import polars as pl

        df = pl.DataFrame({"col1": ["a", "b"], "col2": [1, 2]})
        backend = PolarsBackend(df)

        schema = DatasetSchema(
            columns=[
                ColumnSchema(name="col1", dtype="str"),
                ColumnSchema(name="col2", dtype="int"),
            ]
        )
        # Should not raise
        schema.validate_backend(backend)

    def test_missing_required_column(self) -> None:
        """Test validation fails for missing required column."""
        import polars as pl

        df = pl.DataFrame({"col1": ["a", "b"]})
        backend = PolarsBackend(df)

        schema = DatasetSchema(
            columns=[
                ColumnSchema(name="col1", dtype="str"),
                ColumnSchema(name="col2", dtype="int", required=True),
            ]
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            schema.validate_backend(backend)
        assert "Missing required column: 'col2'" in str(exc_info.value)

    def test_missing_optional_column(self) -> None:
        """Test validation passes for missing optional column."""
        import polars as pl

        df = pl.DataFrame({"col1": ["a", "b"]})
        backend = PolarsBackend(df)

        schema = DatasetSchema(
            columns=[
                ColumnSchema(name="col1", dtype="str"),
                ColumnSchema(name="col2", dtype="int", required=False),
            ]
        )
        # Should not raise
        schema.validate_backend(backend)

    def test_wrong_dtype(self) -> None:
        """Test validation fails for wrong dtype."""
        import polars as pl

        df = pl.DataFrame({"col1": ["a", "b"], "col2": ["x", "y"]})
        backend = PolarsBackend(df)

        schema = DatasetSchema(
            columns=[
                ColumnSchema(name="col1", dtype="str"),
                ColumnSchema(name="col2", dtype="int"),  # Actually strings
            ]
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            schema.validate_backend(backend)
        assert "col2" in str(exc_info.value)
        assert "dtype" in str(exc_info.value)

    def test_multiple_errors(self) -> None:
        """Test validation collects multiple errors."""
        import polars as pl

        df = pl.DataFrame({"col1": [1, 2]})  # Wrong dtype and missing col2
        backend = PolarsBackend(df)

        schema = DatasetSchema(
            columns=[
                ColumnSchema(name="col1", dtype="str"),  # Actually int
                ColumnSchema(name="col2", dtype="int"),  # Missing
            ]
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            schema.validate_backend(backend)
        assert "2 error(s)" in str(exc_info.value)


class TestSchemaValidationPandas:
    """Tests for schema validation with Pandas backend."""

    def test_valid_schema(self) -> None:
        """Test validation passes for matching schema."""
        import pandas as pd

        df = pd.DataFrame({"col1": ["a", "b"], "col2": [1, 2]})
        backend = PandasBackend(df)

        schema = DatasetSchema(
            columns=[
                ColumnSchema(name="col1", dtype="str"),
                ColumnSchema(name="col2", dtype="int"),
            ]
        )
        # Should not raise
        schema.validate_backend(backend)

    def test_missing_required_column(self) -> None:
        """Test validation fails for missing required column."""
        import pandas as pd

        df = pd.DataFrame({"col1": ["a", "b"]})
        backend = PandasBackend(df)

        schema = DatasetSchema(
            columns=[
                ColumnSchema(name="col1", dtype="str"),
                ColumnSchema(name="col2", dtype="int", required=True),
            ]
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            schema.validate_backend(backend)
        assert "Missing required column: 'col2'" in str(exc_info.value)

    def test_wrong_dtype(self) -> None:
        """Test validation fails for wrong dtype."""
        import pandas as pd

        df = pd.DataFrame({"col1": ["a", "b"], "col2": [1.5, 2.5]})
        backend = PandasBackend(df)

        schema = DatasetSchema(
            columns=[
                ColumnSchema(name="col1", dtype="str"),
                ColumnSchema(name="col2", dtype="int"),  # Actually float
            ]
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            schema.validate_backend(backend)
        assert "col2" in str(exc_info.value)


class TestGetDtypePolars:
    """Tests for get_dtype with Polars backend."""

    def test_string_dtype(self) -> None:
        """Test string dtype detection."""
        import polars as pl

        df = pl.DataFrame({"col": ["a", "b"]})
        backend = PolarsBackend(df)
        assert backend.get_dtype("col") == "str"

    def test_int_dtype(self) -> None:
        """Test int dtype detection."""
        import polars as pl

        df = pl.DataFrame({"col": [1, 2]})
        backend = PolarsBackend(df)
        assert backend.get_dtype("col") == "int"

    def test_float_dtype(self) -> None:
        """Test float dtype detection."""
        import polars as pl

        df = pl.DataFrame({"col": [1.5, 2.5]})
        backend = PolarsBackend(df)
        assert backend.get_dtype("col") == "float"

    def test_bool_dtype(self) -> None:
        """Test bool dtype detection."""
        import polars as pl

        df = pl.DataFrame({"col": [True, False]})
        backend = PolarsBackend(df)
        assert backend.get_dtype("col") == "bool"

    def test_list_str_dtype(self) -> None:
        """Test list[str] dtype detection."""
        import polars as pl

        df = pl.DataFrame({"col": [["a", "b"], ["c"]]})
        backend = PolarsBackend(df)
        assert backend.get_dtype("col") == "list[str]"

    def test_list_int_dtype(self) -> None:
        """Test list[int] dtype detection."""
        import polars as pl

        df = pl.DataFrame({"col": [[1, 2], [3]]})
        backend = PolarsBackend(df)
        assert backend.get_dtype("col") == "list[int]"


class TestGetDtypePandas:
    """Tests for get_dtype with Pandas backend."""

    def test_string_dtype(self) -> None:
        """Test string dtype detection."""
        import pandas as pd

        df = pd.DataFrame({"col": ["a", "b"]})
        backend = PandasBackend(df)
        assert backend.get_dtype("col") == "str"

    def test_int_dtype(self) -> None:
        """Test int dtype detection."""
        import pandas as pd

        df = pd.DataFrame({"col": [1, 2]})
        backend = PandasBackend(df)
        assert backend.get_dtype("col") == "int"

    def test_float_dtype(self) -> None:
        """Test float dtype detection."""
        import pandas as pd

        df = pd.DataFrame({"col": [1.5, 2.5]})
        backend = PandasBackend(df)
        assert backend.get_dtype("col") == "float"

    def test_bool_dtype(self) -> None:
        """Test bool dtype detection."""
        import pandas as pd

        df = pd.DataFrame({"col": [True, False]})
        backend = PandasBackend(df)
        assert backend.get_dtype("col") == "bool"

    def test_list_str_dtype(self) -> None:
        """Test list[str] dtype detection."""
        import pandas as pd

        df = pd.DataFrame({"col": [["a", "b"], ["c"]]})
        backend = PandasBackend(df)
        assert backend.get_dtype("col") == "list[str]"
