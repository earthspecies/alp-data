import argparse
import io
import json
import logging

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
from pydantic import Field, field_validator

import esp_data.file_io.functional as F
from esp_data.config.db_config import DataSample, DatasetConfig
from esp_data.dataset.shard_creator import create_sharded_dataset
from esp_data.paths import AnyPath

MIN_AUDIO_SAMPLES = 1
# Hawaii samples are wrongly formatted, so we exclude them for now
DATASETS = [
    "powdermill",
    "OZF",
    "hawaii",
    "Anuraset",
    "OZF_synthetic",
    "BV",
    "humpback",
    "katydids",
    "MT",
]
# DATASETS = ["powdermill"]
# DATASETS = ["Anuraset", "BV", "humpback", "katydids", "MT", "powdermill", "OZF", "OZF_synthetic"]
VERSION = "1.0.0"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voxaboxen")


def encode_bytes(audio_data: np.ndarray, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="flac")
    buffer.seek(0)
    # Get the bytes content
    return buffer.read()


class VoxaboxenSample(DataSample):
    """A sample for the Voxaboxen dataset."""

    audio: np.ndarray = Field(min_length=1, description="The audio data.")
    file_name: str = Field(min_length=1, description="The name of the audio file.")
    selection_table: str = Field(min_length=1, description="The selection table in Raven format")

    @field_validator("selection_table", mode="after")
    def validate_selection_table(cls, value):
        """Validate the selection table."""
        try:
            df = pd.read_csv(io.StringIO(value), sep="\t")
            # empty selection tables are allowed!
            # assert not df.empty, "Selection table is empty."
            assert "Begin Time (s)" in df.columns, "Column 'Begin Time (s)' not found."
            assert "End Time (s)" in df.columns, "Column 'End Time (s)' not found."
            assert "Annotation" in df.columns, "Column 'Annotation' not found."
            # Optional ? Benj says this is allowed
            # assert "Low Freq (Hz)" in df.columns, "Column 'Low Freq (Hz)' not found."
            # assert "High Freq (Hz)" in df.columns, "Column 'High Freq (Hz)' not found."
            assert (df["Begin Time (s)"] > df["End Time (s)"]).sum() == 0, "Begin time is greater than end time."
        except Exception as e:
            raise ValueError(f"Invalid selection table: {e}")

        return value

    @field_validator("audio", mode="after")
    def validate_audio(cls, value):
        """Validate the audio data."""
        assert len(value) > MIN_AUDIO_SAMPLES, "Audio is empty"
        assert np.std(value) > 0.0, "Audio is empty"
        assert np.sum(np.isnan(value)) == 0, "Audio contains nan"

        return value


voxaboxen_cfg = DatasetConfig(
    name="Bioacoustics Sound Event Detection Datasets (BSED)",
    description="""
    Datasets used for training and evaluation in 'Robust detection of overlapping bioacoustic sound events' by Mahon et al. 2025;
    https://arxiv.org/abs/2503.02389
    """,
    sources=[
        "Anuraset",
        "BV",
        "hawaii",
        "humpback",
        "katydids",
        "MT",
        "powdermill",
        "OZF",
        "OZF_synthetic",
    ],
    version="0.1.0",
    creator="L. Mahon, Louis.Mahon at ed dot ac dot uk ; B. Hoffmann, benjamin at earthspecies dot org; L. James, logansmithjames at gmail.com; S. Woolley, sarah.woolley at McGill dot ca",
)


def prepare_voxaboxen_sample(sample: dict) -> dict:
    """Prepare a sample for the Voxaboxen dataset."""
    file_name = sample["fn"]

    # Read audio file
    try:
        # with AnyPath(sample["audio_path"]).open("rb") as f:
        with F.open_file(sample["audio_path"], mode="rb", use_fs=True) as f:
            audio_data, sr = torchaudio.load(f)
            audio_data = audio_data.numpy().squeeze().astype(np.float32)
            if audio_data.ndim > 1 and audio_data.shape[0] == 2:
                # switch stereo channel
                audio_data = audio_data.transpose()

    except Exception:
        raise ValueError(f"Error loading audio file: {sample['audio_path']}")

    # compute duration
    duration = len(audio_data) / sr
    sample["metadata"] = {}
    sample["metadata"]["duration"] = duration
    sample["metadata"]["sample_rate"] = sr

    st_path = sample["selection_table_path"]
    with F.open_file(st_path, mode="r", use_fs=True) as f:
        selection_table_str = f.read()

    sample["selection_table"] = selection_table_str
    sample["audio"] = audio_data
    sample["file_name"] = file_name

    del sample["audio_path"]
    del sample["selection_table_path"]

    try:
        voxaboxen_sample = VoxaboxenSample(**sample)
    except Exception:
        raise ValueError(f"Validation error in sample id: {sample['id']}")

    voxaboxen_sample = voxaboxen_sample.to_dict()
    voxaboxen_sample.pop("derived_from")
    voxaboxen_sample.pop("version")
    voxaboxen_sample.pop("created_at")
    voxaboxen_sample.pop("license")

    # metadata to json string
    voxaboxen_sample["metadata"] = json.dumps(voxaboxen_sample["metadata"])
    # have to write audio as bytes array
    try:
        audio_bytes = encode_bytes(voxaboxen_sample["audio"], sr)
        voxaboxen_sample["audio"] = {
            "bytes": audio_bytes,
            "path": str(voxaboxen_sample["file_name"]),
            "sample_rate": sr,
        }
    except Exception:
        breakpoint()

    return voxaboxen_sample


def make_dataset_for_split(
    path_to_split: AnyPath,
    split: str,
    dataset_name: str,
    output_path: AnyPath,
    args: argparse.Namespace,
):
    metadata_path = path_to_split / f"{split}_info.csv"
    df = pd.read_csv(str(metadata_path), storage_options={"project": "okapi-274503"})

    df["audio_path"] = df["audio_fp"].apply(lambda x: str(path_to_split / x))
    df["selection_table_path"] = df["selection_table_fp"].apply(lambda x: str(path_to_split / x))
    df["source_dataset"] = dataset_name

    logger.info(f"Number of samples in split {split}: {df.shape[0]}")

    # now create the sharded dataset for this component and split
    component_split_path = output_path / split

    cfg = voxaboxen_cfg.copy()
    cfg.name = dataset_name
    cfg.description = f"Split: {split} of component: {dataset_name}"
    cfg.update_changelog(f"Dataset contains {df.shape[0]} samples from split: {split} of component: {dataset_name}")

    create_sharded_dataset(
        df,
        output_path=component_split_path,
        sample_prep_function=prepare_voxaboxen_sample,
        dataset_config=cfg,
        num_samples_per_shard=args.num_samples_per_shard,
        num_workers=args.num_workers,
        shard_type=args.shard_type,
        save_metadata_as="metadata.jsonl",
        merge_data_and_metadata=False,
        log_every=args.log_every,
        writer_batch_size=10,
        error_handling="raise",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Create a sharded Voxaboxen dataset from raw audio files")
    parser.add_argument(
        "--data_path",
        type=AnyPath,
        default="gs://voxaboxen/voxaboxen_data",
        help="The base path.",
    )
    parser.add_argument(
        "--output_path",
        type=AnyPath,
        default="gs://esp-ml-datasets/voxaboxen",
        help="The output path.",
    )
    parser.add_argument(
        "--num_samples_per_shard",
        type=int,
        default=1000,
        help="The number of samples per shard.",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers.")
    parser.add_argument(
        "--shard_type",
        type=str,
        default="parquet",
        help="The type of sharded dataset to create. Either 'arrow', 'hf', or 'parquet'.",
    )
    parser.add_argument("--log_every", type=int, default=100, help="Log every N samples.")
    parser.add_argument("--version", type=str, default=VERSION, help="The version of the dataset.")
    parser.add_argument("--changelog", type=str, default="", help="The changelog.")

    return parser.parse_args()


def prepare_dataset_by_component(args):
    """Prepare the Voxaboxen dataset."""

    base_path = AnyPath(args.data_path)
    output_path = AnyPath(args.output_path)

    voxaboxen_cfg.version = args.version
    voxaboxen_cfg.update_changelog(args.changelog)

    # different approach create datasets as components
    for dataset in DATASETS:
        logger.info(f"Loading dataset: {dataset}")

        dataset_path = base_path / dataset

        if dataset == "OZF_synthetic":
            # this has a number of components
            components = list(dataset_path.glob("overlap*"))
            # filter out components with 'slowed' in the name
            components = [c.parts[-1] for c in components if "slowed" not in c.name]

            for split in ["train", "val", "test"]:
                for component in components:
                    component_dataset_path = dataset_path / component

                    # where to save the sharded dataset
                    # "split" is added the path in the make_dataset_for_split function
                    component_output_path = output_path / dataset / component
                    dataset_name = "OZF_synthetic_" + component
                    make_dataset_for_split(
                        component_dataset_path,
                        split,
                        dataset_name,
                        component_output_path,
                        args,
                    )

        else:
            dataset_path = dataset_path / "formatted"
            for split in ["train", "val", "test"]:
                make_dataset_for_split(dataset_path, split, dataset, output_path / dataset, args)


def prepare_dataset(args):
    """Prepare the Voxaboxen dataset in one go with 3 splits, all components together."""

    base_path = AnyPath(args.data_path)
    output_path = AnyPath(args.output_path)

    voxaboxen_cfg.version = args.version
    voxaboxen_cfg.update_changelog(args.changelog)

    for split in ["train", "val", "test"]:
        print(f"\n===== Creating sharded dataset for split: {split} =====\n")
        split_df = pd.DataFrame()

        # concatenate all the datasets for the split
        for dataset in DATASETS:
            print(f"Loading dataset: {dataset}")

            if dataset == "OZF_synthetic":
                # this has a number of components
                components = list((base_path / dataset).glob("overlap*"))
                # filter out components with 'slowed' in the name
                components = [c.parts[-1] for c in components if "slowed" not in c.name]

                for component in components:
                    component_dataset_path = base_path / dataset / component

                    # load the metadata
                    metadata_path = component_dataset_path / f"{split}_info.csv"
                    df = pd.read_csv(str(metadata_path), storage_options={"project": "okapi-274503"})

                    df["audio_path"] = df["audio_fp"].apply(lambda x: str(component_dataset_path / x))
                    df["selection_table_path"] = df["selection_table_fp"].apply(
                        lambda x: str(component_dataset_path / x)
                    )
                    df["source_dataset"] = f"OZF_synthetic_{component}"

                    split_df = pd.concat([split_df, df], ignore_index=True)

            else:
                # load the metadata
                dataset_path = base_path / dataset / "formatted"
                metadata_path = dataset_path / f"{split}_info.csv"
                df = pd.read_csv(str(metadata_path), storage_options={"project": "okapi-274503"})

                df["audio_path"] = df["audio_fp"].apply(lambda x: str(dataset_path / x))
                df["selection_table_path"] = df["selection_table_fp"].apply(lambda x: str(dataset_path / x))
                df["source_dataset"] = dataset

                split_df = pd.concat([split_df, df], ignore_index=True)

        print(f"Number of samples in split: {split_df.shape[0]}")

        # now create the sharded dataset
        outpath = output_path / split
        create_sharded_dataset(
            split_df,
            output_path=outpath,
            sample_prep_function=prepare_voxaboxen_sample,
            dataset_config=voxaboxen_cfg,
            num_samples_per_shard=args.num_samples_per_shard,
            num_workers=args.num_workers,
            shard_type=args.shard_type,
            save_metadata_as="metadata.jsonl",
            merge_data_and_metadata=False,
            log_every=args.log_every,
            error_handling="raise",
        )


if __name__ == "__main__":
    # prepare_dataset_by_component(parse_args())
    prepare_dataset(parse_args())
