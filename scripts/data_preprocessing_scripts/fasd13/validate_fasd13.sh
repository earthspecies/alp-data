#!/usr/bin/env bash
#SBATCH --job-name=fasd13-validate
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm
set -euo pipefail
cd "${HOME}/esp-data-dev"
export CLOUDSDK_CONFIG="$(mktemp -d)"   # attached SA for gsutil + gcsfs
export GCE_METADATA_MTLS_MODE=none
GCS="gs://esp-data-ingestion/fasd13/v0.1.0"

echo "=== 1. audio-present cross-check ==="
uv run python - <<PY
import subprocess, sys
from io import StringIO
import pandas as pd
GCS="${GCS}"
listings={}
for d in ("audio_16k","audio_32k","support_16k","support_32k"):
    out=subprocess.run(["gsutil","ls","-r",f"{GCS}/{d}"],capture_output=True,text=True,check=True).stdout
    listings[d]={l.split(f"{GCS}/{d}/",1)[-1] for l in out.splitlines() if l.strip().endswith(".wav")}
    print(f"{d}/: {len(listings[d])} wavs on GCS")

blob=subprocess.run(["gsutil","cat",f"{GCS}/fasd13_all.csv"],capture_output=True,text=True,check=True).stdout
df=pd.read_csv(StringIO(blob),keep_default_na=False,na_values=[""])
sup=subprocess.run(["gsutil","cat",f"{GCS}/fasd13_support.csv"],capture_output=True,text=True,check=True).stdout
ds=pd.read_csv(StringIO(sup),keep_default_na=False,na_values=[""])

miss=0
for _,r in df.iterrows():
    for d,col in (("audio_16k","16khz_path"),("audio_32k","32khz_path")):
        rel=str(r[col]).split(f"{d}/",1)[-1]
        if rel not in listings[d]: print("MISSING",d,rel); miss+=1
for _,r in ds.iterrows():
    for d,col in (("support_16k","support_16khz_path"),("support_32k","support_32khz_path")):
        rel=str(r[col]).split(f"{d}/",1)[-1]
        if rel not in listings[d]: print("MISSING",d,rel); miss+=1
print(f"recordings: {len(df)}  support clips: {len(ds)}  missing: {miss}")

# Published per-sub-dataset figures must match what we ingested.
PUB={"AS":(12,0.20,162),"CC":(10,10.00,2200),"GS":(7,38.33,85),"HA":(12,1.10,628),
     "HG":(9,72.00,483),"HW":(10,2.79,1565),"JS":(4,0.23,924),"KD":(12,2.00,883),
     "MS":(10,1.67,1369),"PM":(4,6.42,2032),"RG":(2,1.50,34),"RS":(7,1.87,552),
     "RW":(10,5.00,398)}
bad=0
for code,(nf,hr,nev) in PUB.items():
    sub=df[df.subdataset==code]
    got=(len(sub), sub.audio_duration.sum()/3600, int(sub.n_pos.sum()))
    ok = got[0]==nf and abs(got[1]-hr)<0.05 and got[2]==nev
    print(f"  {code}: files {got[0]}/{nf}  dur {got[1]:.2f}/{hr:.2f} h  events {got[2]}/{nev}  {'ok' if ok else 'MISMATCH'}")
    bad += 0 if ok else 1
print("sub-dataset checks failed:", bad)
sys.exit(1 if (miss or bad) else 0)
PY

echo "=== 2. dataset smoke-load ==="
uv run python - <<'PY'
import numpy as np, pandas as pd
from alp_data.datasets import FASD13
from alp_data.datasets.fasd13 import SUBDATASET_CODES

ST=["Selection","Begin Time (s)","End Time (s)","Q","Label","event_index"]
for sr in (16000,32000):
    ds=FASD13(split="AS",sample_rate=sr)
    it=ds[0]; a=it["audio"]
    assert isinstance(a,np.ndarray) and a.ndim==1 and a.size>0
    assert it["sample_rate"]==sr, it["sample_rate"]
    assert list(it["selection_table"].columns)==ST, list(it["selection_table"].columns)
    print(f"AS@{sr}: n={len(ds)} audio={a.shape} dur={a.size/sr:.1f}s events={len(it['selection_table'])}")

ds=FASD13(split="all",sample_rate=32000)
rows=list(ds._data)
codes={r["subdataset"] for r in rows}
print(f"all: {len(ds)} recordings across {len(codes)} sub-datasets")
assert len(ds)==109, len(ds)
assert set(ds.available_splits)=={"all",*SUBDATASET_CODES}
assert codes==set(SUBDATASET_CODES), codes ^ set(SUBDATASET_CODES)

# every recording must expose 5 usable shots
short=[(r["subdataset"],r["sound_name"],r["n_shots_available"]) for r in rows if int(r["n_shots_available"])<5]
print("recordings with <5 shots:",len(short))
assert not short, short

# windowed lazy read
row=dict(ds._data[0]); row["window_start_sec"],row["window_end_sec"]=10.0,20.0
out=ds._process(row)
assert abs(out["audio"].size/out["sample_rate"]-10.0)<0.3, out["audio"].size
print(f"windowed 10s -> {out['audio'].size} samples, {len(out['selection_table'])} events in row table")

# multi-audio mode via support clips
sup=pd.read_csv("gs://esp-data-ingestion/fasd13/v0.1.0/fasd13_support.csv",
                keep_default_na=False,na_values=[""])
first=dict(ds._data[0])
paths=sup[(sup.subdataset==first["subdataset"])&(sup.sound_name==first["sound_name"])]\
      .sort_values("shot_index")["support_32khz_path"].tolist()
assert len(paths)==5, paths
row=dict(first); row["window_start_sec"],row["window_end_sec"]=20.0,30.0
row["support_32khz_paths"]=paths[:4]
out=ds._process(row)
assert len(out["audios"])==5 and "audio" not in out
print("multi-audio 4-shot ->",[a.size for a in out["audios"]])
print("labels:",ds.get_available_labels())
print("SMOKE OK")
PY
echo "Done."
