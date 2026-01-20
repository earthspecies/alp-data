# Copyright (2024)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# (License text continues)

import argparse
import csv
import json
import os

import soundfile as sf
from datasets import Audio, load_dataset

"""
Download EbiRd taxonomy and set csv path.
"""


def load_ebird_taxonomy(
    csv_path: str = "../BirdSet/resources/ebird_codes/eBird_taxonomy_v2024.csv",
) -> dict:
    """Loads the eBird taxonomy CSV and returns a dict mapping.

    eBird code (e.g., 'amepip') -> {
        'common_name': ...,
        'sci_name': ...,
        'family': ...,
        'phylum': ...,
        'class': ...,
        'order': ...,
        'species': ...
    }

    Returns
    -------
    dict
        Mapping of eBird codes to taxonomy information
    """
    code_to_taxonomy = {}
    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            species_code = row["SPECIES_CODE"]
            if species_code:
                family_name = row["FAMILY"]
                if family_name:
                    # remove any trailing text after the first word
                    family_name = family_name.split()[0]
                code_to_taxonomy[species_code] = {
                    "common_name": row["PRIMARY_COM_NAME"],
                    "sci_name": row["SCI_NAME"],
                    "family": family_name,
                    "order": row["ORDER"],
                }
    return code_to_taxonomy


def save_formatted_jsonl(data: list, filepath: str) -> None:
    """Utility to save a list of JSON objects to a JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def download_and_save_birdset_with_labels(
    dataset_name: str = "HSN",
    is_train: bool = False,
    save_audio: bool = True,
    download_path: str = "audio",
    output_path: str = "../foundation-model-data/data",
    taxonomy_path: str = "../BirdSet/resources/ebird_codes/eBird_taxonomy_v2024.csv",
) -> None:
    split = "train" if is_train else "test"

    # 1) Load the dataset from Hugging Face
    dataset = load_dataset("DBD-research-group/BirdSet", dataset_name)

    # 2) Access the appropriate split
    if not is_train:
        if "test_5s" not in dataset:
            raise ValueError(f"Dataset does not contain 'test_5s' split for {dataset_name}")
        test_5s_dataset = dataset["test_5s"]
        if save_audio:
            test_5s_dataset = test_5s_dataset.cast_column("audio", Audio(sampling_rate=48_000))
    else:
        test_5s_dataset = dataset["train"]
        if save_audio:
            test_5s_dataset = test_5s_dataset.cast_column("audio", Audio(sampling_rate=48_000))

    # 3) Prepare output directories using provided paths
    audio_dir = os.path.join(download_path, f"birdset-{split}", dataset_name)
    os.makedirs(audio_dir, exist_ok=True)
    metadata_path = os.path.join(audio_dir, f"{dataset_name}_metadata.jsonl")

    formatted_dir = os.path.join(output_path, f"birdset-{split}", dataset_name)
    os.makedirs(formatted_dir, exist_ok=True)

    # Define paths for common, sci, and taxonomic JSONL files
    common_path = os.path.join(formatted_dir, f"{dataset_name}_common.jsonl")
    sci_path = os.path.join(formatted_dir, f"{dataset_name}_sci.jsonl")
    taxonomic_path = os.path.join(formatted_dir, f"{dataset_name}_taxonomic.jsonl")
    common_skip_path = os.path.join(formatted_dir, f"{dataset_name}_common_skip.jsonl")
    sci_skip_path = os.path.join(formatted_dir, f"{dataset_name}_sci_skip.jsonl")
    taxonomic_skip_path = os.path.join(formatted_dir, f"{dataset_name}_taxonomic_skip.jsonl")

    # 4) Retrieve the ClassLabel object for ebird_code_multilabel
    multi_label_feature = test_5s_dataset.features["ebird_code_multilabel"].feature

    # 5) Load local eBird taxonomy CSV to map codes -> taxonomy info
    # (common_name, sci_name, family, phylum, class, order, species)
    code_to_taxonomy = load_ebird_taxonomy(csv_path=taxonomy_path)

    # Track unique labels
    unique_common_names = set()
    unique_sci_names = set()
    unique_taxonomic_names = set()

    # Prepare formatted JSONL rows
    formatted_common = []
    formatted_sci = []
    formatted_taxonomic = []
    formatted_common_skip = []
    formatted_sci_skip = []
    formatted_taxonomic_skip = []

    # 6) Write out audio + metadata
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        for _idx, sample in enumerate(test_5s_dataset):
            extension = sample["filepath"].split(".")[-1]
            filename = sample["filepath"].split("/")[-1].replace(f".{extension}", ".flac")
            audio_filename = os.path.join(audio_dir, filename)

            # Write the FLAC file if requested
            if save_audio:
                audio_array = sample["audio"]["array"]
                sample_rate = sample["audio"]["sampling_rate"]
                sf.write(audio_filename, audio_array, sample_rate, format="FLAC")

            # Build metadata dictionary (excluding raw audio array)
            metadata = {k: v for k, v in sample.items() if k != "audio"}
            metadata["audio_filepath"] = audio_filename

            # Handle multi-label codes
            if isinstance(sample["ebird_code_multilabel"], list):
                ebird_codes_multilabel = [
                    multi_label_feature.int2str(label_id)
                    for label_id in sample["ebird_code_multilabel"]
                ]

                common_names = []
                sci_names = []
                taxonomic_names = []

                for code in ebird_codes_multilabel:
                    if code in code_to_taxonomy:
                        common_names.append(code_to_taxonomy[code]["common_name"])
                        sci_names.append(code_to_taxonomy[code]["sci_name"])
                        # Construct a "taxonomic name" string
                        classification = code_to_taxonomy[code]
                        taxonomic_name = (
                            f"{classification['order']} "
                            f"{classification['family']} "
                            f"{classification['sci_name']}"
                        )
                        taxonomic_names.append(taxonomic_name)

                # Update metadata and unique labels
                metadata["ebird_code_multilabel_common_names"] = common_names
                metadata["ebird_code_multilabel_sci_names"] = sci_names
                metadata["ebird_code_multilabel_taxonomic_names"] = taxonomic_names
                unique_common_names.update(common_names)
                unique_sci_names.update(sci_names)
                unique_taxonomic_names.update(taxonomic_names)

                # -----------------------------------------------------------------
                # DETECTION prompts (non-skip): replace "" with "None"
                # -----------------------------------------------------------------
                # Common
                common_label_str = ", ".join(common_names) if common_names else "None"
                formatted_common.append(
                    {
                        "local_path": audio_filename,
                        "label": common_label_str,
                        "text": common_label_str,
                        "prompt": (
                            "<Audio><AudioHere></Audio> "
                            "What are the common names for the focal species in the audio, if any?"
                        ),
                    }
                )

                # Scientific
                sci_label_str = ", ".join(sci_names) if sci_names else "None"
                formatted_sci.append(
                    {
                        "local_path": audio_filename,
                        "label": sci_label_str,
                        "text": sci_label_str,
                        "prompt": (
                            "<Audio><AudioHere></Audio> "
                            "What are the scientific names for the focal species in "
                            "the audio, if any?"
                        ),
                    }
                )

                # Taxonomic
                taxonomic_label_str = ", ".join(taxonomic_names) if taxonomic_names else "None"
                formatted_taxonomic.append(
                    {
                        "local_path": audio_filename,
                        "label": taxonomic_label_str,
                        "text": taxonomic_label_str,
                        "prompt": (
                            "<Audio><AudioHere></Audio> "
                            "What are the taxonomic names for the focal species in "
                            "the audio, if any?"
                        ),
                    }
                )

                # -----------------------------------------------------------------
                # CLASSIFICATION prompts (skip): unchanged (no "" -> "None")
                # -----------------------------------------------------------------
                if common_names:
                    formatted_common_skip.append(
                        {
                            "local_path": audio_filename,
                            "label": ", ".join(common_names),
                            "text": ", ".join(common_names),
                            "prompt": (
                                "<Audio><AudioHere></Audio> "
                                "What is the common name for the focal species in the audio?"
                            ),
                        }
                    )

                if sci_names:
                    formatted_sci_skip.append(
                        {
                            "local_path": audio_filename,
                            "label": ", ".join(sci_names),
                            "text": ", ".join(sci_names),
                            "prompt": (
                                "<Audio><AudioHere></Audio> "
                                "What is the scientific name for the focal species in the audio?"
                            ),
                        }
                    )

                if taxonomic_names:
                    formatted_taxonomic_skip.append(
                        {
                            "path": audio_filename,
                            "label": ", ".join(taxonomic_names),
                            "text": ", ".join(taxonomic_names),
                            "prompt": (
                                "<Audio><AudioHere></Audio> "
                                "What is the taxonomic name for the focal species in the audio?"
                            ),
                        }
                    )

            # Save this metadata row
            meta_file.write(json.dumps(metadata) + "\n")

    # 7) Save the formatted JSONL files
    save_formatted_jsonl(formatted_common, common_path)
    save_formatted_jsonl(formatted_sci, sci_path)
    save_formatted_jsonl(formatted_taxonomic, taxonomic_path)
    save_formatted_jsonl(formatted_common_skip, common_skip_path)
    save_formatted_jsonl(formatted_sci_skip, sci_skip_path)
    save_formatted_jsonl(formatted_taxonomic_skip, taxonomic_skip_path)

    # 8) Print unique labels for verification
    print("Unique common names:", list(unique_common_names))
    print("Unique scientific names:", list(unique_sci_names))
    print("Unique taxonomic names:", list(unique_taxonomic_names))

    print(f"Saved audio clips to: {audio_dir}")
    print(f"Saved metadata to:   {metadata_path}")
    print(f"Saved common names JSONL: {common_path}")
    print(f"Saved scientific names JSONL: {sci_path}")
    print(f"Saved taxonomic names JSONL: {taxonomic_path}")
    print(f"Saved common names (skipped empty) JSONL: {common_skip_path}")
    print(f"Saved scientific names (skipped empty) JSONL: {sci_skip_path}")
    print(f"Saved taxonomic names (skipped empty) JSONL: {taxonomic_skip_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess BirdSet dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HSN",
        help=(
            "Name of the dataset to download (e.g., UHH, SNE, SSW, VOX, NES, "
            "HSN, POW, NBP, PER). Use 'all' to process all datasets."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "all"],
        default="test",
        help="Which split to download: train, test, or all (both train and test)",
    )
    parser.add_argument(
        "--download-path",
        type=str,
        default="/mnt/home/marius_miron_earthspecies_org/data/BirdSet/audio",
        help="Directory to save audio clips",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/mnt/home/marius_miron_earthspecies_org/data/BirdSet/data",
        help="Directory to save formatted data",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=str,
        default="/mnt/home/marius_miron_earthspecies_org/data/BirdSet/eBird_taxonomy_v2024.csv",
        help="Path to the eBird taxonomy CSV file",
    )

    args = parser.parse_args()

    # Define all available datasets
    all_datasets = ["UHH", "SNE", "SSW", "VOX", "NES", "HSN", "POW", "NBP", "PER"]

    # Determine which datasets to process
    if args.dataset.lower() == "all":
        datasets_to_process = all_datasets
    else:
        datasets_to_process = [args.dataset]

    # Determine which splits to process
    splits_to_process = []
    if args.split == "all":
        splits_to_process = [True, False]  # True for train, False for test
    elif args.split == "train":
        splits_to_process = [True]
    else:  # test
        splits_to_process = [False]

    # Process each dataset and split combination
    for dataset_name in datasets_to_process:
        for is_train in splits_to_process:
            print(f"Processing dataset: {dataset_name}, split: {'train' if is_train else 'test'}")
            download_and_save_birdset_with_labels(
                dataset_name=dataset_name,
                is_train=is_train,
                save_audio=True,
                download_path=args.download_path,
                output_path=args.output_path,
                taxonomy_path=args.taxonomy_path,
            )


if __name__ == "__main__":
    main()
