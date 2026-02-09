"""
This script caches all possible outputs with ok==True of the gbif_taxonomy
converter, to allow for faster lookups.
"""

import pandas as pd

from esp_data.discover import GBIFConverter

CACHE_PATH = "gs://esp-ml-datasets/gbif_taxonomy/v0.1.0/gbif_animals_converter_cache.json"

cached = {}
converter = GBIFConverter()
to_lookup = list(converter.df_by_canonical_name.index)

issues = []

for i, lookup in enumerate(to_lookup):
    if i % 100 == 0:
        print(f"{i} / {len(to_lookup)} completed")
    result, ok = converter(lookup)
    if not ok:
        issues.append(lookup)
        continue
    cached[lookup] = result

print(f"Found {len(issues)} issues in {len(to_lookup)} records")

pd.DataFrame.from_dict(cached, orient="index").to_json(CACHE_PATH, indent=2)

# # check it
# result = pd.read_json(CACHE_PATH).to_dict(orient="index")
# breakpoint()
