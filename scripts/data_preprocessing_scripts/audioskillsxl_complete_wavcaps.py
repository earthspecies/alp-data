"""
Complete the AudioSkillsXL ``wavcaps`` split.

Every WavCaps MCQ item in nvidia/AudioSkills references an AudioSet clip
(``sound`` = ``Y<ytid>.flac``). The original ``wavcaps.csv`` kept only 42,058 of
the 92,234 items, but ~88% of the rest have audio already present in our
AudioSet copy (``gs://esp-ml-datasets/audioset/v0.2.0/raw/``) — they were simply
dropped by the original build. This script resolves every item against that
audio and writes the completed split CSV to
``gs://esp-data-ingestion/AudioSkillsXL/v0.1.0/raw/wavcaps.csv``.

Only metadata is re-hosted: audio is still referenced under the AudioSet copy
(read-only) at load time, exactly like the existing rows. Items whose AudioSet
audio is genuinely absent from our copy are reported (and written to
``wavcaps_unresolved_ids.txt``) for optional later sourcing from cvssp/WavCaps.

Usage:
    uv run python scripts/data_preprocessing_scripts/audioskillsxl_complete_wavcaps.py \
        --wavcaps-json /tmp/wc/WavCaps.json
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from google.cloud import storage

AUDIOSET_ROOT = "gs://esp-ml-datasets/audioset/v0.2.0/raw/"
AUDIOSET_BUCKET = "esp-ml-datasets"
AUDIOSET_AUDIO_PREFIX = "audioset/v0.2.0/raw/audio"
SEGMENTS = ["eval_segments", "unbalanced_train_segments"]  # unbalanced last → wins on dup
AUDIO_TOKEN = "<Audio><AudioHere></Audio>"
OUT_GCS = "gs://esp-data-ingestion/AudioSkillsXL/v0.1.0/raw/wavcaps.csv"
COLUMNS = [
    "id",
    "source",
    "sound",
    "audio_path",
    "duration",
    "messages",
    "relative_path",
    "gcs_path",
    "16khz_path",
    "32khz_path",
]


def list_stems(client: storage.Client, seg: str) -> dict[str, str]:
    """Map every AudioSet ``<stem>`` under ``audio/<seg>/`` to its segment.

    Returns
    -------
    dict[str, str]
        ``{stem: seg}`` for each ``.wav`` file in the segment folder.
    """
    prefix = f"{AUDIOSET_AUDIO_PREFIX}/{seg}/"
    out: dict[str, str] = {}
    for blob in client.list_blobs(AUDIOSET_BUCKET, prefix=prefix):
        if blob.name.endswith(".wav"):
            out[blob.name[len(prefix) : -4]] = seg
    return out


def transform_messages(conversations: list[dict]) -> str:
    """Convert WavCaps ``conversations`` to esp-data ``messages`` JSON.

    Returns
    -------
    str
        JSON list of ``{"role", "content"}`` turns; the ``<sound>`` token in the
        user turn is replaced by the esp-data audio token at the front.
    """
    msgs = []
    for c in conversations:
        role = "user" if c["from"] == "human" else "assistant"
        val = c["value"]
        if role == "user":
            val = f"{AUDIO_TOKEN} {val.replace('<sound>', '').strip()}"
        msgs.append({"role": role, "content": val})
    return json.dumps(msgs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wavcaps-json", default="/tmp/wc/WavCaps.json")
    p.add_argument("--out", default=OUT_GCS)
    p.add_argument("--local-out", default="/tmp/wc/wavcaps_complete.csv")
    args = p.parse_args()

    data = json.load(open(args.wavcaps_json))
    print(f"WavCaps MCQ items: {len(data)}")

    client = storage.Client()
    stem2seg: dict[str, str] = {}
    for seg in SEGMENTS:
        s = list_stems(client, seg)
        print(f"  listed {len(s)} stems in {seg}")
        stem2seg.update(s)
    print(f"AudioSet audio stems available: {len(stem2seg)}")

    rows = []
    unresolved = []
    for e in data:
        yid = str(e["id"])
        stem = yid[1:] if yid.startswith("Y") else yid
        seg = stem2seg.get(stem)
        if seg is None:
            unresolved.append(yid)
            continue
        ap = f"audio/{seg}/{stem}.wav"
        rows.append(
            {
                "id": yid,
                "source": "wavcaps",
                "sound": e.get("sound", f"{yid}.flac"),
                "audio_path": ap,
                "duration": e.get("duration", ""),
                "messages": transform_messages(e["conversations"]),
                "relative_path": ap,
                "gcs_path": AUDIOSET_ROOT + ap,
                "16khz_path": f"audio_16khz/{seg}/{stem}.wav",
                "32khz_path": f"audio_32khz/{seg}/{stem}.wav",
            }
        )

    df = pd.DataFrame(rows, columns=COLUMNS)
    print(f"\nresolved: {len(df)}  unresolved (audio absent from our AudioSet): {len(unresolved)}")
    print(df["audio_path"].str.split("/").str[1].value_counts().to_string())

    # sample-verify 16k/32k parity for recovered clips
    bucket = client.bucket(AUDIOSET_BUCKET)
    sample = df.sample(n=min(40, len(df)), random_state=0)
    ok16 = ok32 = 0
    for _, r in sample.iterrows():
        if bucket.blob(f"{AUDIOSET_AUDIO_PREFIX.rsplit('/', 1)[0]}/{r['16khz_path']}").exists():
            ok16 += 1
        if bucket.blob(f"{AUDIOSET_AUDIO_PREFIX.rsplit('/', 1)[0]}/{r['32khz_path']}").exists():
            ok32 += 1
    print(f"16k/32k parity on {len(sample)} sampled: 16k={ok16} 32k={ok32}")

    df.to_csv(args.local_out, index=False)
    os.system(f"gsutil -q cp {args.local_out} {args.out}")
    print(f"wrote {len(df)} rows -> {args.out}")
    if unresolved:
        with open("wavcaps_unresolved_ids.txt", "w") as f:
            f.write("\n".join(unresolved))
        print(f"unresolved ids -> wavcaps_unresolved_ids.txt ({len(unresolved)})")


if __name__ == "__main__":
    main()
