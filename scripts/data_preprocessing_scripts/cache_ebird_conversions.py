"""
Cache eBird taxonomy bidirectional mappings.

Reads each CSV from gs://esp-data-ingestion/taxonomy/ebird_taxonomy_raw/,
produces two JSON dicts per year:
  - sci_name_to_ebird_<year>.json   : { scientific_name -> ebird_code }
  - ebird_to_sci_name_<year>.json   : { ebird_code -> scientific_name }
and uploads them to gs://esp-data-ingestion/taxonomy/ebird_taxonomy_cached/.
"""

import json
import re

import pandas as pd

from esp_data.io import filesystem

RAW_PATH = "esp-data-ingestion/taxonomy/ebird_taxonomy_raw"
CACHED_PATH = "esp-data-ingestion/taxonomy/ebird_taxonomy_cached"


def main() -> None:
    fs = filesystem("gcs")
    csv_files = sorted(fs.glob(f"{RAW_PATH}/*.csv"))

    if not csv_files:
        print("No CSV files found in gs://esp-data-ingestion/taxonomy/ebird_taxonomy_raw/")
        return

    for gcs_path in csv_files:
        filename = gcs_path.split("/")[-1]
        match = re.search(r"\d{4}", filename)
        if not match:
            print(f"Skipping {filename}: could not extract year")
            continue
        year = match.group()

        print(f"Processing {filename} (year={year}) ...")
        df = pd.read_csv(f"gs://{gcs_path}")

        valid = df[["SCI_NAME", "SPECIES_CODE"]].dropna()
        sci_to_ebird: dict[str, str] = dict(
            zip(valid["SCI_NAME"], valid["SPECIES_CODE"], strict=True)
        )
        ebird_to_sci: dict[str, str] = dict(
            zip(valid["SPECIES_CODE"], valid["SCI_NAME"], strict=True)
        )

        for name, data in [
            (f"sci_name_to_ebird_{year}.json", sci_to_ebird),
            (f"ebird_to_sci_name_{year}.json", ebird_to_sci),
        ]:
            out_path = f"{CACHED_PATH}/{name}"
            with fs.open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Wrote gs://{out_path} ({len(data)} entries)")

    print("Done.")


if __name__ == "__main__":
    main()
