"""Utility functions for working with Pydantic models and BigQuery schemas."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel


def pydantic_to_bigquery_schema(model: type[BaseModel]) -> list[dict]:
    """Convert Pydantic model to BigQuery schema.

    Args:
        model (type[BaseModel]): Pydantic model class

    Returns:
        list[dict]: BigQuery schema as a list of dictionaries
    """
    type_mapping = {
        str: "STRING",
        int: "INTEGER",
        float: "FLOAT64",
        bool: "BOOLEAN",
        datetime: "TIMESTAMP",
        dict: "JSON",  # BigQuery supports JSON type
        list: "ARRAY",  # BigQuery supports ARRAY type
    }

    schema = []

    for field_name, field in model.model_fields.items():
        field_type = field.annotation

        # Handle Optional types
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Optional:
            field_type = field_type.__args__[0]
            mode = "NULLABLE"
        else:
            mode = "REQUIRED"

        # Get the base type for Union types (e.g., str | Path -> str)
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            field_type = field_type.__args__[0]

        bq_type = type_mapping.get(field_type, "STRING")

        field_schema = {
            "name": field_name,
            "type": bq_type,
            "mode": mode,
        }

        # Add description if available
        if field.description:
            field_schema["description"] = field.description

        schema.append(field_schema)

    return schema


def get_bigquery_schema_json(model: type[BaseModel]) -> str:
    """Convert Pydantic model to BigQuery schema JSON"""
    schema = pydantic_to_bigquery_schema(model)
    return json.dumps(schema, indent=2)


# Create schema file
def save_bigquery_schema(model: type[BaseModel], output_path: Union[str, Path]) -> None:
    """Save BigQuery schema to a JSON file"""
    schema_json = get_bigquery_schema_json(model)
    with open(output_path, "w") as f:
        f.write(schema_json)
