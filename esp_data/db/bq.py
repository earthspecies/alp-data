"""CRUD operations for BigQuery."""

import pandas as pd
import logging
from google.cloud import bigquery


logger = logging.getLogger(__name__)


def create_table_from_schema(
    project_id: str,
    dataset_id: str,
    table_id: str,
    schema: list[bigquery.SchemaField],
    client: bigquery.Client | None = None,
):
    """Create a table in BigQuery from a schema."""
    if client is None:
        client = bigquery.Client(project=project_id)
    dataset_ref = bigquery.DatasetReference(project=project_id, dataset_id=dataset_id)
    table_ref = dataset_ref.table(table_id)
    table = bigquery.Table(table_ref, schema=schema)
    try:
        client.create_table(table)
    except Exception as e:
        raise RuntimeError(f"Error creating table: {e}")
    logger.info(f"Created table {table_id} in dataset {dataset_id} in project {project_id}")


def insert_row(
    project_id: str,
    dataset_id: str,
    table_id: str,
    row: dict,
    client: bigquery.Client | None = None,
):
    """Insert a row into a BigQuery table."""
    if client is None:
        client = bigquery.Client(project=project_id)
    dataset_ref = bigquery.DatasetReference(project=project_id, dataset_id=dataset_id)
    table_ref = dataset_ref.table(table_id)
    table = client.get_table(table_ref)
    errors = client.insert_rows(table, [row])
    if errors:
        raise ValueError(f"Errors inserting row: {errors}")
    logger.info(f"Inserted row into table {table_id} in dataset {dataset_id} in project {project_id}")


def query(
    project_id: str,
    query: str,
    return_df: bool = True,
    client: bigquery.Client | None = None,
) -> pd.DataFrame | list[dict]:
    """Query a BigQuery table."""
    if client is None:
        client = bigquery.Client(project=project_id)
    query_job = client.query(query)

    try:
        query_job.result()
    except Exception as e:
        raise RuntimeError(f"Error running query: {e}")

    if return_df:
        return query_job.to_dataframe()
    else:
        return list(query_job)


def update_row(
    project_id: str,
    dataset_id: str,
    table_id: str,
    row: dict,
    primary_key: str,
    client: bigquery.Client | None = None,
):
    """Update a row in a BigQuery table."""
    if client is None:
        client = bigquery.Client(project=project_id)
    # Get primary key field
    # primary_key_field = [field for field in table.schema if field.name == primary_key][0]
    primary_key_value = row[primary_key]
    query = f"""
        UPDATE `{project_id}.{dataset_id}.{table_id}`
        SET {", ".join([f"{k} = {v}" for k, v in row.items()])}
        WHERE {primary_key} = {primary_key_value}
    """
    query_job = client.query(query)
    try:
        query_job.result()
    except Exception as e:
        raise RuntimeError(f"Error running query: {e}")
    logger.info(f"Updated row in table {table_id} in dataset {dataset_id} in project {project_id}")


class BigQueryTable:
    def __init__(self, project_id: str, dataset_id: str, table_id: str):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.client = bigquery.Client(project=project_id)

    def create(self, schema: list[bigquery.SchemaField]) -> None:
        create_table_from_schema(self.project_id, self.dataset_id, self.table_id, schema, self.client)

    def insert(self, row: dict) -> None:
        insert_row(self.project_id, self.dataset_id, self.table_id, row, self.client)

    def query(self, query: str, return_df: bool = True) -> pd.DataFrame | list[dict]:
        return query(self.project_id, query, return_df, self.client)

    def update(self, row: dict, primary_key: str) -> None:
        update_row(self.project_id, self.dataset_id, self.table_id, row, primary_key, self.client)
