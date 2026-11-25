#!/usr/bin/env bash
#SBATCH --job-name=ien-weak-validate
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm
set -euo pipefail
cd "${HOME}/esp-data-dev"
export CLOUDSDK_CONFIG="$(mktemp -d)"   # attached SA for gsutil + gcsfs

echo "=== audio-present cross-check (typeB + background) ==="
uv run python - <<'PY'
import subprocess, sys
from io import StringIO
import pandas as pd
BASE="gs://esp-data-ingestion/indian-fauna/v0.1.0"
def check(sub, csv):
    gcs=f"{BASE}/{sub}"
    listings={}
    for d in ("audio_16k","audio_32k"):
        out=subprocess.run(["gsutil","ls","-r",f"{gcs}/{d}"],capture_output=True,text=True,check=True).stdout
        listings[d]={l.split(f"{gcs}/{d}/",1)[-1] for l in out.splitlines() if l.strip().endswith(".wav")}
    blob=subprocess.run(["gsutil","cat",f"{gcs}/{csv}"],capture_output=True,text=True,check=True).stdout
    df=pd.read_csv(StringIO(blob),keep_default_na=False,na_values=[""])
    miss=sum(1 for _,r in df.iterrows() for d,c in (("audio_16k","16khz_path"),("audio_32k","32khz_path"))
             if str(r[c]).split(f"{d}/",1)[-1] not in listings[d])
    print(f"{sub}: {len(df)} rows, 16k={len(listings['audio_16k'])} 32k={len(listings['audio_32k'])} missing={miss}")
    return miss
m=check("typeB","indian_fauna_weak_all.csv")+check("background","indian_fauna_background_all.csv")
sys.exit(1 if m else 0)
PY

echo "=== dataset smoke-load ==="
uv run python - <<'PY'
import numpy as np
from alp_data.datasets import IndianFaunaWeak, IndianFaunaBackground
COLS=["Selection","Begin Time (s)","End Time (s)","Low Freq (Hz)","High Freq (Hz)","Species","Presence"]
for cls in (IndianFaunaWeak, IndianFaunaBackground):
    for sr in [16000,32000]:
        ds=cls(split="all",sample_rate=sr)
        it=ds[0]; a,st=it["audio"],it["selection_table"]
        assert isinstance(a,np.ndarray) and a.ndim==1 and a.size>0 and it["sample_rate"]==sr
        assert list(st.columns)==COLS, list(st.columns)
        print(f"{cls.__name__}@{sr}: n={len(ds)} audio={a.shape} dur={a.size/sr:.1f}s "
              f"fg='{it['foreground_species']}' n_species={it['n_species']}")
    labs=ds.get_available_labels()
    print(f"  {cls.__name__} labels: {len(labs)} sample={labs[:5]}")
print("SMOKE OK")
PY
echo "Done."
