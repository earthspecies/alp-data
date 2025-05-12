"""Make NatureLM annotations from the shards"""

import json
import logging

import pandas as pd

import esp_data.file_io.functional as F
from esp_data.paths import AnyPath

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("naturelm")


def main():
    output_path = AnyPath("gs://esp-ml-datasets/naturelm/processed/v0.1.0/parquet/train")

    # append to a annotations.parquet file in the output path
    annotations_path = output_path / "annotations.jsonl"

    logger.info("Loading metadata...")
    metadata_path = output_path / "metadata.jsonl"
    metadata = pd.read_json(metadata_path, lines=True, orient="records")
    logger.info(f"Metadata loaded, shape: {metadata.shape}")

    # find all parquet shards in output path
    shard_paths = list(F.list_files(output_path, "shard*", use_fs=True))
    logger.info(f"Found {len(shard_paths)} shards in {output_path}")

    write_batch_size = 100_000_000
    annotations = pd.DataFrame()

    for i, shard_path in enumerate(shard_paths):
        logger.info(f"Loading data from {shard_path}...")
        data = pd.read_parquet(shard_path)
        ids = data["id"].tolist()

        # find subset of metadata with ids
        metadata_subset = metadata[metadata["id"].isin(ids)]
        annotations_df = [json.loads(row["metadata"]) for _, row in data.iterrows()]
        annotations_df = pd.DataFrame(annotations_df)
        data = data[["id", "source_dataset", "task", "file_name"]]

        annotations_df = pd.concat([data, annotations_df], axis=1)
        # logger.info(
        #     "Created annotations dataframe, shape: %s, columns: %s",
        #     annotations_df.shape,
        #     annotations_df.columns,
        # )

        # merge metadata_subset and annotations on 'id', keep all columns from both
        data = pd.merge(
            annotations_df,
            metadata_subset,
            on="id",
            how="outer",
        )
        # logger.info("Combined metadata and annotations, new data shape: %s", data.shape)

        if data.empty:
            logger.warning("Merged data is empty, skipping this shard")
            continue

        if annotations.empty:
            annotations = data
        else:
            annotations = pd.concat([annotations, data], ignore_index=True, axis=0)

        logger.info("Combined all annotations, new shape: %s", annotations.shape)

        if annotations.shape[0] >= write_batch_size:
            logger.info(f"Writing annotations to {annotations_path}...")

            if annotations_path.exists():
                # append to existing file
                annotations.to_json(
                    str(annotations_path),
                    orient="records",
                    lines=True,
                    mode="a",
                    storage_options={"project": "okapi-274503"},
                )
            else:
                # write new file
                annotations.to_json(
                    str(annotations_path),
                    orient="records",
                    lines=True,
                    storage_options={"project": "okapi-274503"},
                )
            annotations = pd.DataFrame()

    # write any remaining annotations
    if not annotations.empty:
        logger.info(f"Writing remaining annotations to {annotations_path}...")
        if annotations_path.exists():
            # append to existing file
            annotations.to_json(
                str(annotations_path),
                orient="records",
                lines=True,
                mode="a",
                storage_options={"project": "okapi-274503"},
            )
        else:
            # write new file
            annotations.to_json(
                str(annotations_path),
                orient="records",
                lines=True,
                storage_options={"project": "okapi-274503"},
            )


if __name__ == "__main__":
    main()
