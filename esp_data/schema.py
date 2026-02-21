"""Schema definitions for dataset validation.

This module provides a schema abstraction for validating dataset columns
with expected names, dtypes, and required status.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from esp_data.backends.protocol import DataBackend


# Abstract dtype enum - maps to polars/pandas types
DType = Literal["str", "int", "float", "bool", "datetime", "list[str]", "list[int]", "list[float]"]


class SchemaValidationError(Exception):
    """Raised when data doesn't match expected schema."""

    ...


class ColumnSchema(BaseModel):
    """Schema for a single column.

    Attributes
    ----------
    name : str
        Column name
    dtype : DType
        Expected data type
    required : bool
        Whether column must be present (default True)
    description : str
        Human-readable description of what the column contains
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Column name")
    dtype: DType = Field(description="Expected data type")
    required: bool = Field(default=True, description="Whether column must be present")
    description: str = Field(default="", description="Human-readable description of the column")


class DatasetSchema(BaseModel):
    """Schema defining expected columns for a dataset.

    Attributes
    ----------
    columns : list[ColumnSchema]
        List of column schemas defining the expected structure
    """

    model_config = ConfigDict(frozen=True)

    columns: list[ColumnSchema] = Field(description="List of column schemas")

    @property
    def required_columns(self) -> list[str]:
        """Return names of required columns.

        Returns
        -------
        list[str]
            Names of columns marked as required
        """
        return [col.name for col in self.columns if col.required]

    @property
    def column_names(self) -> list[str]:
        """Return all column names.

        Returns
        -------
        list[str]
            Names of all columns in the schema
        """
        return [col.name for col in self.columns]

    def validate_backend(self, backend: "DataBackend") -> None:
        """Validate a loaded backend against this schema.

        Parameters
        ----------
        backend : DataBackend
            The backend to validate

        Raises
        ------
        SchemaValidationError
            If the backend data doesn't match the expected schema
        """
        backend_columns = set(backend.columns)
        errors = []

        for col_schema in self.columns:
            # Check if required column is present
            if col_schema.name not in backend_columns:
                if col_schema.required:
                    errors.append(f"Missing required column: '{col_schema.name}'")
                continue

            # Check dtype
            actual_dtype = backend.get_dtype(col_schema.name)
            if actual_dtype != col_schema.dtype:
                errors.append(
                    f"Column '{col_schema.name}' has dtype '{actual_dtype}', "
                    f"expected '{col_schema.dtype}'"
                )

        if errors:
            raise SchemaValidationError(
                f"Schema validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
