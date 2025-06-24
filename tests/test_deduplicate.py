import pandas as pd

from esp_data.transforms import Deduplicate

def test_deduplicate() -> None:
    df = pd.DataFrame({
        "species": ["bee", "bee", "butterfly", "bee"],
        "count": [10, 10, 5, 10]
    })

    transform = Deduplicate(subset=["species"], keep_first=True)
    deduplicated_df, _ = transform(df)

    expected_df = pd.DataFrame({
        "species": ["bee", "butterfly"],
        "count": [10, 5]
    })

    pd.testing.assert_frame_equal(deduplicated_df.reset_index(drop=True), expected_df)
