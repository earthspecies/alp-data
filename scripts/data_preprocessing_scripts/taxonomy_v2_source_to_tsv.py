"""
Script to convert GBIF backbone source to csv for simple queries

Assumes Darwin Core Archive source has been downloaded from
https://www.gbif.org/dataset/d7dddbf4-2cf0-4f39-9b2a-bb099caae36c
and unzipped
"""

import csv
import os
import sys

import pandas as pd

SOURCE_DIR = "/mnt/home/taxonomy/data/backbone"

csv.field_size_limit(sys.maxsize)

# Input file from GBIF backbone taxonomy
input_file = os.path.join(SOURCE_DIR, "Taxon.tsv")
output_file = os.path.join(SOURCE_DIR, "gbif_animals.tsv")

# Build a dictionary of vernacular names by taxonID
vernacular_names = {}
vern_fp = os.path.join(SOURCE_DIR, "VernacularName.tsv")
with open(vern_fp, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["language"] != "en":
            continue
        taxon_id = int(row["taxonID"])
        vernacular_name = row["vernacularName"]
        if taxon_id not in vernacular_names:
            vernacular_names[taxon_id] = set()
        vernacular_names[taxon_id].add(vernacular_name.lower())

df = pd.read_csv(
    input_file,
    sep="\t",
    quoting=csv.QUOTE_NONE,  # this fixes a parsing issue where \n and \t
    # characters are enclosed in quotes in the original file.
    engine="python",  # REQUIRED
    escapechar="\\",  # optional but often needed
)

df = df[df["kingdom"] == "Animalia"]
df = df.set_index("taxonID")
df = df[df["taxonRank"].isin(["species", "subspecies", "variety", "form"])]
tokeep = [
    "acceptedNameUsageID",
    "parentNameUsageID",  # keep so we can look up subspecies
    "canonicalName",
    "taxonRank",
    "taxonomicStatus",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
]
df = df[tokeep]


def get_vernacular_name(taxon_id: int) -> str:
    """
    Gets csv of vernacular names for a taxon id

    Returns
    --------
    str csv of vernacular names
    """
    taxon_id = int(taxon_id)
    if taxon_id not in vernacular_names:
        return ""
    else:
        return ",".join(sorted(vernacular_names[taxon_id]))


print("getting vernacular names")
df["vernacularName"] = df.index.map(get_vernacular_name)
print("done getting vernacular names")
df.to_csv(output_file, sep="\t")

# df = pd.read_csv(output_file, sep='\t', index_col='taxonID')
