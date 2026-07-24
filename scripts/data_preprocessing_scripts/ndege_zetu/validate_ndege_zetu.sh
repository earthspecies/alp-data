#!/usr/bin/env bash
#SBATCH --job-name=ndege-validate
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm
set -euo pipefail
cd "${HOME}/esp-data-dev"
export CLOUDSDK_CONFIG="$(mktemp -d)"   # attached SA for gsutil + gcsfs
GCS="gs://esp-data-ingestion/ndege-zetu/v0.1.0"

echo "=== 1. audio-present cross-check (mp3 originals + 16k/32k wav mirrors) ==="
uv run python - <<PY
import subprocess, sys
from io import StringIO
import pandas as pd
GCS="${GCS}"
def names(d):
    out=subprocess.run(["gsutil","ls",f"{GCS}/{d}"],capture_output=True,text=True,check=True).stdout
    return {l.rsplit("/",1)[-1] for l in out.splitlines() if l.strip().lower().endswith((".mp3",".wav"))}
L={d:names(d) for d in ("audio","audio_16k","audio_32k")}
for d in L: print(f"{d}/: {len(L[d])} files")
blob=subprocess.run(["gsutil","cat",f"{GCS}/ndege_zetu_all.csv"],capture_output=True,text=True,check=True).stdout
df=pd.read_csv(StringIO(blob),keep_default_na=False,na_values=[""])
miss=0
for _,r in df.iterrows():
    for col,d in (("audio_fp","audio"),("16khz_path","audio_16k"),("32khz_path","audio_32k")):
        if str(r[col]).rsplit("/",1)[-1] not in L[d]: print("MISSING",d,r[col]); miss+=1
print("missing:",miss); sys.exit(1 if miss else 0)
PY

echo "=== 2. dataset smoke-load ==="
uv run python - <<'PY'
import numpy as np
from esp_data.datasets import NdegeZetu
COLS=["Selection","Begin Time (s)","End Time (s)","Low Freq (Hz)","High Freq (Hz)","Species","Presence"]
for split in ["test","val","train"]:
    for sr in [16000,32000]:
        ds=NdegeZetu(split=split,sample_rate=sr)
        it=ds[0]; a,st=it["audio"],it["selection_table"]
        assert isinstance(a,np.ndarray) and a.ndim==1 and a.size>0
        assert it["sample_rate"]==sr
        assert float(it["audio_duration_sec"])>0
        assert list(st.columns)==COLS, list(st.columns)
        print(f"{split}@{sr}: n={len(ds)} audio={a.shape} dur_col={it['audio_duration_sec']} n_species={it['n_species']}")

ds=NdegeZetu(split="all",sample_rate=16000)
# a positive (multi-species) recording: check weak boxes span the clip
pos=[i for i in range(len(ds)) if int(ds._data[i]["n_species"])>=2]
it=ds[pos[0]]; st=it["selection_table"]
assert (st["End Time (s)"].astype(float)>0).all() and (st["Begin Time (s)"].astype(float)==0).all()
print(f"positive {it['sound_name']}: {len(st)} weak boxes, fg='{it['foreground_species']}' bg='{it['background_species']}' presence={sorted(st['Presence'].unique())}")

# windowed read (as window_annotations would drive it) inherits clip labels
row=dict(ds._data[pos[0]]); row["window_start_sec"],row["window_end_sec"]=10.0,20.0
out=ds._process(row)
assert abs(out["audio"].size/out["sample_rate"]-10.0)<0.3
print(f"windowed 10s read -> {out['audio'].size} samples, {len(out['selection_table'])} inherited weak labels")

# a negative recording -> empty selection table
neg=[i for i in range(len(ds)) if int(ds._data[i]["n_species"])==0]
e=ds[neg[0]]; assert len(e["selection_table"])==0
print(f"negative {e['sound_name']}: selection_table rows={len(e['selection_table'])}")
print("labels:",len(ds.get_available_labels()),"species; e.g.",ds.get_available_labels()[:3])
print("SMOKE OK")
PY
echo "Done."
