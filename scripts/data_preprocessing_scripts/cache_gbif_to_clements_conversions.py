"""
Cache GBIF canonical name → Clements scientific name mappings.

For each eBird taxonomy year (2021–2025):
  - Load EBirdConverter to get the set of Clements scientific names
  - Load GBIFConverter (default GCS path)
  - For each Clements name, resolve it through GBIF; if successful, record
    {gbif_canonical_name: clements_name}
  - Write the result to:
    gs://esp-data-ingestion/taxonomy/ebird_taxonomy_cached/gbif_to_clements_{year}.json

This is useful when dataset labels use GBIF canonical names but the target
vocabulary is Clements (eBird).
"""

import json

from esp_data.discover import EBirdConverter, GBIFConverter
from esp_data.io import filesystem

CACHED_PATH = "esp-data-ingestion/taxonomy/ebird_taxonomy_cached"
YEARS = range(2021, 2026)


def main() -> None:
    print("Loading GBIFConverter ...")
    gbif_converter = GBIFConverter()

    fs = filesystem("gcs")

    for year in YEARS:
        print(f"\nProcessing year {year} ...")
        ebird_converter = EBirdConverter(year=year)

        gbif_to_clements: dict[str, str] = {}
        failed = 0

        for clements_name in ebird_converter.sci_to_ebird:
            info, ok = gbif_converter(clements_name)
            if ok:
                gbif_canonical = info["canonicalName"]
                gbif_to_clements[gbif_canonical] = clements_name
            else:
                failed += 1

        out_path = f"{CACHED_PATH}/gbif_to_clements_{year}.json"
        with fs.open(out_path, "w") as f:
            json.dump(gbif_to_clements, f, indent=2)

        total = len(ebird_converter.sci_to_ebird)
        print(
            f"  Wrote gs://{out_path} "
            f"({len(gbif_to_clements)} entries, {failed}/{total} Clements names unresolved in GBIF)"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
