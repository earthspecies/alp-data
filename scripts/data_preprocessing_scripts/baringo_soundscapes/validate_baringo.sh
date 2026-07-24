#!/usr/bin/env bash
#SBATCH --job-name=baringo-validate
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm
set -euo pipefail
cd "${HOME}/esp-data-dev"
export CLOUDSDK_CONFIG="$(mktemp -d)"
GCS="gs://esp-data-ingestion/baringo-soundscapes/v0.1.0"

echo "=== 1. audio-present cross-check ==="
uv run python - <<PY
import subprocess, sys
from io import StringIO
import pandas as pd
GCS="${GCS}"
def names(d):
    out=subprocess.run(["gsutil","ls",f"{GCS}/{d}"],capture_output=True,text=True,check=True).stdout
    return {l.rsplit("/",1)[-1] for l in out.splitlines() if l.strip().endswith(".wav")}
L={d:names(d) for d in ("audio_16k","audio_32k")}
for d in L: print(f"{d}/: {len(L[d])} wavs")
blob=subprocess.run(["gsutil","cat",f"{GCS}/baringo_soundscapes_all.csv"],capture_output=True,text=True,check=True).stdout
df=pd.read_csv(StringIO(blob),keep_default_na=False,na_values=[""])
miss=0
for _,r in df.iterrows():
    for col,d in (("16khz_path","audio_16k"),("32khz_path","audio_32k")):
        if str(r[col]).rsplit("/",1)[-1] not in L[d]: print("MISSING",d,r[col]); miss+=1
print("missing:",miss); sys.exit(1 if miss else 0)
PY

echo "=== 2. dataset smoke-load ==="
uv run python - <<'PY'
import numpy as np
from esp_data.datasets import BaringoSoundscapes
COLS=["Selection","Begin Time (s)","End Time (s)","Low Freq (Hz)","High Freq (Hz)","Species","eBird_Code"]
for sr in [16000,32000]:
    ds=BaringoSoundscapes(split="all",sample_rate=sr)
    assert "audio_duration" in ds.columns, "missing audio_duration column"
    it=ds[0]; a,st=it["audio"],it["selection_table"]
    assert isinstance(a,np.ndarray) and a.ndim==1 and a.size>0
    assert it["sample_rate"]==sr and float(it["audio_duration"])>0
    assert list(st.columns)==COLS, list(st.columns)
    assert (st["Begin Time (s)"].astype(float)==st["End Time (s)"].astype(float)).all(), "not centerpoints"
    print(f"all@{sr}: n={len(ds)} audio={a.shape} dur={it['audio_duration']} n_events={it['n_events']} n_species={it['n_species']}")

ds=BaringoSoundscapes(split="all",sample_rate=32000)
# windowed read into a ~1h recording; centerpoints inside the window are kept
row=dict(ds._data[0]); row["window_start_sec"],row["window_end_sec"]=100.0,110.0
out=ds._process(row)
assert abs(out["audio"].size/out["sample_rate"]-10.0)<0.3
st=out["selection_table"]
print(f"windowed 100-110s: {out['audio'].size} samples, {len(st)} centerpoints in-window (times {sorted(round(float(x),1) for x in st['Begin Time (s)'])[:5]})")
print("labels:",len(ds.get_available_labels()),"species; e.g.",ds.get_available_labels()[:3])
print("SMOKE OK")
PY
echo "Done."
