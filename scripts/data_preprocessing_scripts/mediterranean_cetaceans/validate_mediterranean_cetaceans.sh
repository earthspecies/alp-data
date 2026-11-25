#!/usr/bin/env bash
#SBATCH --job-name=medcet-validate
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
GCS="gs://esp-data-ingestion/mediterranean-cetaceans/v0.1.0"

echo "=== 1. audio-present cross-check ==="
uv run python - <<PY
import subprocess, sys
from io import StringIO
import pandas as pd
GCS="${GCS}"
listings={}
for d in ("audio_16k","audio_32k"):
    out=subprocess.run(["gsutil","ls","-r",f"{GCS}/{d}"],capture_output=True,text=True,check=True).stdout
    listings[d]={l.split(f"{GCS}/{d}/",1)[-1] for l in out.splitlines() if l.strip().endswith(".wav")}
    print(f"{d}/: {len(listings[d])} wavs on GCS")
blob=subprocess.run(["gsutil","cat",f"{GCS}/mediterranean_cetaceans_all.csv"],capture_output=True,text=True,check=True).stdout
df=pd.read_csv(StringIO(blob),keep_default_na=False,na_values=[""])
miss=0
for _,r in df.iterrows():
    for d,col in (("audio_16k","16khz_path"),("audio_32k","32khz_path")):
        rel=str(r[col]).split(f"{d}/",1)[-1]
        if rel not in listings[d]: print("MISSING",d,rel); miss+=1
print("rows:",len(df),"missing:",miss); sys.exit(1 if miss else 0)
PY

echo "=== 2. dataset smoke-load ==="
uv run python - <<'PY'
import numpy as np
from alp_data.datasets import MediterraneanCetaceans
COLS=["Selection","Begin Time (s)","End Time (s)","Low Freq (Hz)","High Freq (Hz)","Species","signalType"]
for sr in [16000,32000]:
    ds=MediterraneanCetaceans(split="all",sample_rate=sr)
    it=ds[0]; a,st=it["audio"],it["selection_table"]
    assert isinstance(a,np.ndarray) and a.ndim==1 and a.size>0
    assert it["sample_rate"]==sr
    assert list(st.columns)==COLS, list(st.columns)
    print(f"all@{sr}: n={len(ds)} audio={a.shape} dur={a.size/sr:.1f}s events0={len(st)} species={it['species']}")

ds=MediterraneanCetaceans(split="all",sample_rate=32000)
# a recording WITH events
nz=[i for i in range(len(ds)) if int(ds._data[i]["n_events"])>0]
r=ds[nz[0]]; st=r["selection_table"]
print(f"non-empty rec {r['sound_name']} ({r['species']}): {len(st)} events, "
      f"signalTypes={sorted(st['signalType'].unique())}, "
      f"begin<=dur ok={bool((st['Begin Time (s)']<=r['audio_duration']+0.5).all())}")
# windowed 10s seek
row=dict(ds._data[nz[0]]); row["window_start_sec"],row["window_end_sec"]=0.0,10.0
out=ds._process(row)
assert abs(out["audio"].size/out["sample_rate"]-10.0)<0.3, out["audio"].size
print(f"windowed 10s -> {out['audio'].size} samples, {len(out['selection_table'])} events in-window")
# an empty (pure-negative) recording parses to 0-row table
empties=[i for i in range(len(ds)) if int(ds._data[i]["n_events"])==0]
if empties:
    e=ds[empties[0]]; assert len(e["selection_table"])==0
    print(f"empty rec {e['sound_name']}: rows={len(e['selection_table'])} (negative OK)")
print("labels:",ds.get_available_labels())
print("SMOKE OK")
PY
echo "Done."
