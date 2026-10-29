"""A small registered dataset used by the export tests.

It lives in its own module, not in a conftest, so that spawned worker
processes can import it by name and register it.
"""

import json
from io import StringIO
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import soundfile as sf

from alp_data.dataset import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    register_config,
    register_dataset,
)
from alp_data.io import read_audio


@register_config
class PackTestConfig(DatasetConfig):
    dataset_name: str = "pack_test"
    csv_path: str = ""


@register_dataset
class PackTestDataset(Dataset):
    info = DatasetInfo(
        name="pack_test",
        owner="tests",
        split_paths={"train": "virtual://pack_test"},
        version="0.1.0",
        description="Fixture dataset for export tests",
        sources="tests",
    )

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        sample_rate: int | None = None,
        output_take_and_give: dict[str, str] | None = None,
        backend: str = "pandas",
    ) -> None:
        super().__init__(output_take_and_give=output_take_and_give, backend=backend)
        self.csv_path = csv_path
        self.split = split
        self.sample_rate = sample_rate
        self._load()

    def _load(self) -> None:
        self._data = self._backend_class.from_csv(self.csv_path)

    @property
    def columns(self) -> list[str]:
        return self._data.columns

    @property
    def available_splits(self) -> list[str]:
        return ["train"]

    @classmethod
    def from_config(cls, cfg: PackTestConfig) -> tuple["PackTestDataset", dict[str, Any]]:
        ds = cls(
            csv_path=cfg.csv_path,
            split=cfg.split,
            sample_rate=cfg.sample_rate,
            output_take_and_give=cfg.output_take_and_give,
            backend=cfg.backend,
        )
        meta = ds.apply_transformations(cfg.transformations) if cfg.transformations else {}
        return ds, meta

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._process(self._data[idx])

    def __str__(self) -> str:
        return f"PackTestDataset({len(self)} rows)"

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        audio, sr = read_audio(row["local_path"])
        audio = audio.astype(np.float32)
        row["audio"] = audio
        row["sample_rate"] = sr
        row["labels"] = json.loads(row["labels_json"])
        row["selection_table"] = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
        row["targets"] = np.array([len(audio), 1], dtype=np.int64)
        if row.pop("with_extra", False):
            row["extra"] = "only on some rows"
        if row.pop("unpackable", False):
            row["obj"] = object()
        if self.output_take_and_give:
            return {give: row[take] for take, give in self.output_take_and_give.items()}
        return row


def make_source(
    root: Path,
    n: int = 5,
    duplicate_of: dict[int, int] | None = None,
    corrupt: set[int] | None = None,
    optional_every: int | None = None,
    unpackable: bool = False,
    multiline_notes: bool = False,
    loud: set[int] | None = None,
) -> PackTestConfig:
    """Write `n` tiny wav files and a CSV, and return a config pointing at them.

    Parameters
    ----------
    root : Path
        Directory to write into.
    n : int
        Number of rows.
    duplicate_of : dict[int, int] | None
        Rows whose audio should be byte-identical to another row's.
    corrupt : set[int] | None
        Rows whose audio file is not a valid audio file.
    optional_every : int | None
        Every n-th row's `_process` output carries an extra key `extra`.
    unpackable : bool
        Every row's `_process` output carries a value that cannot be packed.
    multiline_notes : bool
        Add a free-text `notes` column containing tabs and newlines.
    loud : set[int] | None
        Rows whose audio peaks at 1.5, outside what 16-bit PCM can hold. Written
        as float WAV so the values survive the read.

    Returns
    -------
    PackTestConfig
        A config that rebuilds the dataset from the written CSV.
    """
    duplicate_of = duplicate_of or {}
    corrupt = corrupt or set()
    loud = loud or set()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    rows = []
    for i in range(n):
        path = root / f"clip_{i}.wav"
        if i in corrupt:
            path.write_bytes(b"this is not audio")
        else:
            seed = duplicate_of.get(i, i)
            local_rng = np.random.default_rng(seed)
            audio = (local_rng.integers(-32768, 32767, size=800 + 100 * seed) / 32768.0).astype(
                np.float32
            )
            if i in loud:
                audio = audio * 1.5
                sf.write(path, audio, 16000, subtype="FLOAT")
            else:
                sf.write(path, audio, 16000, subtype="PCM_16")
        st = pd.DataFrame({"Begin Time (s)": [0.0, 0.01 * i], "Annotation": ["a", "b"]})
        rows.append(
            {
                "local_path": str(path),
                "label": f"species_{i % 2}",
                "labels_json": json.dumps([f"species_{i % 2}", "extra"]),
                "selection_table": st.to_csv(sep="\t", index=False),
                "score": float(rng.uniform()),
                "with_extra": bool(optional_every and i % optional_every == 0),
                "unpackable": unpackable,
                **({"notes": f"line one\tcol\nline two {i}"} if multiline_notes else {}),
            }
        )
    csv_path = root / "table.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return PackTestConfig(csv_path=str(csv_path), backend="pandas")
