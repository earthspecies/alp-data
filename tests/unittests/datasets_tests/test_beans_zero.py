
import numpy as np
import pandas as pd
import pytest

from esp_data import DatasetConfig
from esp_data.datasets import BeansZero
from esp_data.utils import create_hash

SPLITS = list(BeansZero.info.split_paths.keys())
EXPECTED_LENGTHS = None
EXPECTED_COLS = [
    'source_dataset',
    'dataset_name',
    'output',
    'instruction_text',
    'instruction',
    'task',
    'file_name',
    'license',
    'id',
    'metadata',
    'audio_path_original_sample_rate',
    'audio_path_16KHz',
    'audio_path_32KHz'
]

# Hash values computed on 2025-12-20
_hashes = {
    'test': (91965, 'cab6616f00b2b0ada58bbf6a6cdd29c516bb4161142808ebe61a1ef2720a1441', 'c0242633324e0936f94e19e4af1f986fa696279a68ebb5b2a7b341f457291daf'),
    'cbi': (3620, '0619df489b4fb36ab1b023721755d4c74f071e7ca7850456e8fca127a1285141', 'c1ab5cfd4c76caf5c7a01985c68e0ff23e50fde223174c9b8230779d9c97e049'),
    'watkins': (339, 'e6f93b28b420efe275c4ce3211fe204bf5bfa8a5a9d284018fdbe82201c55cff', '7f030bca4992bbd42ce6f9fecec592fbed831824f79fc41ae1da5e1ff8f10cf9'),
    'hiceas': (1485, '2e4e8328a2199cb1e4b3d4f1726a2b7b13303d364df599f0c97582ba632a9f7e', '3397b98e3b6978b7c2d72b88fc251db144a22b4d71c19b51d67c2c4ff2779a03'),
    'dcase': (13688, 'cab6616f00b2b0ada58bbf6a6cdd29c516bb4161142808ebe61a1ef2720a1441', 'c1d013da672c001382d724ba4c766e2fb9a691bc293f36e3878335b2725841a2'),
    'enabirds': (4543, '2ddb2548774f5635ba89cb0e2f85b2ffb62703782a58601f408a727ffd477dc4', '0bda3d7ab56c122b208229168b6e0ec042bd5129dfbfd51554ab246862b9a008'),
    'esc50': (400, 'af40bb2a2968cc4e2fbc98b2f4f5e005359c33bf4418237cb6985314d0c1248a', '0b64c2adf0582dcf4b646ce34ec3321c944bb296ad286de6180926ccd596a600'),
    'humbugdb': (1859, '1f0577c94c9d0699b4837e6bae45d67c7874742636f089ec97a411139d4dd778', '5bbb19ec000e3b35319c9b5eaeb9a31b96e07937a7deda5682aaa602364b61ca'),
    'rfcx': (10406, '29fced3e3270514175be84a2f59508d2c2057acb5e8075f3a903f0a98661b98b', '4b158e6490be7b086f821f430a4a6eb73594b06138dd9116b863c884149221e3'),
    'gibbons': (18560, '40eda1df464f136ec47371c8327654ca405206714ae463218ede625e0756dfa8', '288538ebfafff40f644e87db99d9caf7485ecf3dbe524490d6005a1c8b80223f'),
    'lifestage': (466, '258baf1f14359b821a3cf8325258b88f3932e40e7424a0b75934ff44e22b456e', '1140d08c84eaf3906a8effe51e8c2ad876a90b195d71f270a9aa9659d20cfadc'),
    'call-type': (1000, '23b830015b46db965e2e069cd1c87bea34e521453765389f0b1cbe36c3c65372', '681ef6cff3e25be0d8e65fb2833861509dab43d6ad8b64320820dac900ad6095'),
    'captioning': (26468, 'f2f772e40dbab07938c7ea05e628c957191514880d1142805d808a9e484918f4', '3ac887660712b3dc8e53f6b615fbe79ae3718281d9eca8339807fe5e34c9bf0b'),
    'zf-indiv': (1160, '312ddbcdf21f8f41663d4586f3073e004691c2d5618fa3150a5c6b532e1c872a', '7785758e578e263f95fd4ad9f9e2435b07fd67ce2686993e251c684dbed2faa2'),
    'unseen-family-cmn': (451, 'df5ada315c9a3f3a1d1d46359061e87dad0d5e05462589d2c65fc9d7fcadfc40', 'ef7eeff1577cfe67335b30e3dd25203a6ac46ce8abae6bd7cdff8950890d3531'),
    'unseen-family-sci': (451, '14519105c1967860709398f113da1cc54f8ffeff74072973e7c43f71b9d0462d', '89b825ae127bd5bd55d506b0ecc199f78a73c911cdda2deceb9d412d3e71e587'),
    'unseen-family-tax': (451, 'c6445eafbb864f37fc44fe8eaeecfcb6968c2e33848fc754125deb7b5932cb17', '827d06fc624721a91a85b11355ec7dbf5087619cc947261ca8d3df8bf978965b'),
    'unseen-genus-cmn': (951, '27c75c749a9a3ca29de31f567f52c24be9aa7d097ba6b573ddccf26c3018d5a5', 'a7b46664bfb90da9761dcbf82186298e629ae27bd23949aae9a4358fae6308d1'),
    'unseen-genus-sci': (951, 'b8042b073a610f3578f33c49d158db2b7f2981ef423d563466f0982165ecab66', 'a9fd48d27fb327c25061c37f355131ce6cbc447382c803362c98e9f1a0b4d564'),
    'unseen-genus-tax': (951, 'ecd46d487b255b0d23d84f937c51514d9aeac985a30f76020a0a07b8f9cfcf87', '39478ac7b582e170457a78d0a32285b944b7ec429c1bc8a97c1b32e960c4de44'),
    'unseen-species-cmn': (1255, 'c5361fbca40e0d9c51e1bb5f6a3253ca884786d9077671962ffc97f8e0cfee25', '57959150ed55d9490d2412a460a6489706b6ae1333b10f453ac171703529b2e6'),
    'unseen-species-sci': (1255, '65fd1ae4a2569f944efbcf31660dda5baf5579f1da13ebf185f95b73ed9d1a27', '71dee29f08a3336330693a3741fc6607e7e20e47297386fe0403efa9d70139ca'),
    'unseen-species-tax': (1255, '684567ae8d640a600def2cf808237967b75b94841f5dff3c29a0648005ec3e4a', 'e18db9dba52301687accfdc131ea0f384666ef98938454cf02c0c9986636ca91')
}
EXPECTED_LENGTHS = {k: v[0] for k, v in _hashes.items()}
EXPECTED_FIRST_ITEM_AUDIO_SHA256 = {k: v[1] for k, v in _hashes.items()}
ANNOTATIONS_SHA256 = {k: v[2] for k, v in _hashes.items()}


@pytest.fixture
def ds() -> BeansZero:
    """Load BeansZero dataset for testing."""
    _ds = BeansZero(split="test", streaming=True, backend='pandas')
    return _ds

def create_dataset_hashes(split_name: str) -> tuple[int, str, str]:
    ds = BeansZero(split=split_name, streaming=False, backend='pandas')

    # Compute first item audio hash
    first_sample = ds[0]
    first_audio = first_sample["audio"].tobytes()
    first_audio_hash = create_hash(first_audio)

    # Compute annotations hash
    df = ds._data.unwrap.sort_index(axis=0).sort_index(axis=1)
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    annotations_hash = create_hash(csv_bytes)

    return len(ds), first_audio_hash, annotations_hash


def create_all_split_hashes():
    results = {}
    for split in SPLITS:
        results[split] = create_dataset_hashes(split)
    return results


@pytest.mark.skipif(
    EXPECTED_LENGTHS is None,
    reason="Hash values not yet computed. Run hash computation first."
)
def test_dataset_integrity() -> None:
    """Test the dataset snapshot."""
    all_hashes = create_all_split_hashes()
    for split in SPLITS:
        len_h, first_audio_h, annotations_h = all_hashes[split]
        expected_len = EXPECTED_LENGTHS[split]
        assert len_h == expected_len, f"Dataset length for split '{split}' does not match expected value."
        assert (
            first_audio_h == EXPECTED_FIRST_ITEM_AUDIO_SHA256[split]
        ), f"First item audio hash for split '{split}' does not match expected value."
        assert (
            annotations_h == ANNOTATIONS_SHA256[split]
        ), f"Annotations hash for split '{split}' does not match expected value."


def test_columns_property(ds: BeansZero) -> None:
    """Test the columns property."""
    cols = ds.columns
    for col in EXPECTED_COLS:
        assert col in cols, f"Expected column '{col}' not found in dataset columns."


def test_construction_from_config() -> None:
    """Test the from_config class method."""
    config = {
        "dataset_name": "beans_zero",
        "split": "cbi",
        "streaming": True,
        "backend": "pandas",
    }
    config = DatasetConfig.model_validate(config)
    ds, _ = BeansZero.from_config(config)
    assert isinstance(ds, BeansZero), "from_config did not return a BeansZero instance."


def test_transforms_in_from_config() -> None:
    """Test construction with transforms in from_config."""
    config = {
        "dataset_name": "beans_zero",
        "split": "watkins",
        "streaming": True,
        "backend": "pandas",
        "transformations": [{
            "type": "label_from_feature",
            "feature": "output",
            "output_feature": "label"
        }]
    }
    config = DatasetConfig.model_validate(config)
    ds, metadata = BeansZero.from_config(config)

    assert "label_from_feature" in metadata, "Transformations metadata not returned."
    assert "label" in ds.columns, "Transformed feature 'label' not found in dataset columns."


def test_available_splits(ds: BeansZero) -> None:
    """Test the available_splits method."""
    splits = ds.available_splits
    for split in SPLITS:
        assert split in splits, f"Expected split '{split}' not found in available splits."


def test_split_lookup_error() -> None:
    """Test that an invalid split raises a LookupError."""
    with pytest.raises(LookupError):
        BeansZero(split="invalid_split", streaming=False, backend='pandas')


def test_streaming_iter(ds: BeansZero) -> None:
    # iterate through first 3 samples
    for i, sample in enumerate(ds):
        if i >= 3:
            break
        assert "audio" in sample, "Sample does not contain 'audio' key."
        assert "instruction" in sample, "Sample does not contain 'instruction' key."


def test_random_samples() -> None:
    """Test random samples from the dataset."""
    ds = BeansZero(split="test", streaming=False, backend='polars')
    import random

    n = len(ds)
    rng = random.Random()
    sample_indices = [rng.randrange(n) for _ in range(min(2, n))]

    for idx in sample_indices:
        item = ds[idx]
        assert "audio" in item, f"[{idx}] missing 'audio' key"
        audio = item["audio"]

        assert len(audio) >= 10, f"[{idx}] audio too short (length={len(audio)})"
        assert not np.any(np.isnan(audio)), f"[{idx}] audio contains NaN values"
        assert not np.all(audio == 0), f"[{idx}] audio is all zeros"


# if __name__ == "__main__":
#     all_hashes = create_all_split_hashes()
#     print(all_hashes)
