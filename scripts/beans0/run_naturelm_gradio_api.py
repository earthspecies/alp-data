import argparse
import json
import tempfile

import pandas as pd
import soundfile as sf
from gradio_client import Client, file
from tqdm import tqdm

from esp_data.dataset.esp_dataset import load_esp_dataset
from esp_data.io import anypath
from esp_data.io.parsers import read_audio_bytes


def data_processor(data: dict):
    audio, _ = read_audio_bytes(data["audio.wav"], "wav")
    # audio is a numpy array
    metadata = json.loads(data["metadata.json"])
    return {"audio": audio, **metadata}


def empty_df():
    return pd.DataFrame(columns=["prediction", "task", "true_label", "instruction", "file_name", "id"])


def main():
    parser = argparse.ArgumentParser(description="Run the NatureLM Gradio API.")
    parser.add_argument("--api_url", type=str, default="http://localhost:7860", help="The URL of the API.")
    parser.add_argument("--path_to_dataset", type=str, help="The path to the dataset.")
    parser.add_argument("--output_path", type=str, help="The path to the output file.")
    parser.add_argument(
        "--write_every", type=int, default=10, help="The number of samples to write to the output file."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size.")
    parser.add_argument("--resume", action="store_true", help="Resume from the output file.")
    args = parser.parse_args()

    path_to_dataset = anypath(args.path_to_dataset)

    print(f"Loading dataset from {path_to_dataset}...")
    dataset = load_esp_dataset(
        "webdataset",
        path=path_to_dataset,
        shuffle_size=None,
        shard_shuffle=False,
        load_metadata=True,
        data_processor=data_processor,
    )
    print(f"Loaded dataset with {len(dataset)} samples.")

    gr_client = Client(args.api_url)

    def predict(sample):
        query = sample["instruction_text"]

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            sf.write(audio_file.name, sample["audio"], 16000)
            audio_file.seek(0)

            aud_file = file(audio_file.name)
            response = gr_client.predict(files=[aud_file], task=query, api_name="/run_batch_inference")

        # respones is like f"Batch summary:\n{batch_summary}\n\n"
        prediction = response.replace("Batch summary:\n", "").strip()
        prediction = prediction.split(":")[0].strip()

        return prediction, sample["task"], sample["output"], query, sample["file_name"], sample["id"]

    output_df = empty_df()

    # try to resume from the output file
    try:
        output_df = pd.read_json(args.output_path, orient="records", lines=True)
        print(f"Resumed from {args.output_path} with {len(output_df)} samples.")
        start = len(output_df)
        output_df = empty_df()

    except Exception as e:
        print(e)
        start = 0
        print("Starting from scratch.")

    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        if i < start:
            continue

        prediction, task, label, instruction, fname, id = predict(sample)

        # append to output_df
        output_dict = {
            "prediction": prediction,
            "task": task,
            "true_label": label,
            "instruction": instruction,
            "file_name": fname,
            "id": id,
        }
        output_df = pd.concat([output_df, pd.DataFrame([output_dict])], axis=0)

        if len(output_df) % args.write_every == 0:
            output_df.to_json(
                args.output_path,
                orient="records",
                lines=True,
                storage_options=anypath(args.output_path).storage_options,
                mode="a",
            )
            output_df = empty_df()

    # write the remaining samples
    if len(output_df) > 0:
        output_df.to_json(
            args.output_path,
            orient="records",
            lines=True,
            storage_options=anypath(args.output_path).storage_options,
            mode="a",
        )


if __name__ == "__main__":
    main()
