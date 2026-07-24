"""GBIF taxonomy for the 7 Madeira odontocete species (light; run on dev VM).
Fixes source typos, resolves GBIF canonicalName + common + ranks, writes
species_taxonomy.csv for the (Slurm) build step to join."""
import pandas as pd
from esp_data.discover.gbif_taxonomy import GBIFConverter

SCI_NAME_FIX = {
    "Globicephala macrorhyncus": "Globicephala macrorhynchus",
    "Steno bredanesis": "Steno bredanensis",
    "Tursiop truncatus": "Tursiops truncatus",
}
META = "/mnt/home/esp-data-dev/scratch_madeira/dl/Metadata.xlsx"
OUT = "/mnt/home/esp-data-dev/scratch_madeira/species_taxonomy.csv"

df = pd.read_excel(META, sheet_name="Metadata")
conv = GBIFConverter()
rows = []
for (sp, common, code), _ in df.groupby(["Species", "Common name", "Species code"]):
    fixed = SCI_NAME_FIX.get(sp.strip(), sp.strip())
    info = conv(fixed) or {}
    rows.append({
        "src_species": sp, "species_code": str(code).strip(),
        "src_common": common, "sci_fixed": fixed,
        "canonical_name": info.get("canonicalName", fixed),
        "gbif_id": info.get("taxonID") or info.get("usageKey"),
        "kingdom": info.get("kingdom"), "phylum": info.get("phylum"),
        "class": info.get("class"), "order": info.get("order"),
        "family": info.get("family"), "genus": info.get("genus"),
        "vernacular": info.get("vernacularName"),
    })
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out[["src_species","species_code","canonical_name","gbif_id","family"]].to_string())
print("\nwrote", OUT)
