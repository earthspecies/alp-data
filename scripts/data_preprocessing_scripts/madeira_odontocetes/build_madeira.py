"""Build the Madeira odontocete eval dataset (weak, clip-level).

Windows each 288 kHz single-species-encounter WAV into non-overlapping 10 s clips,
resamples to 16 kHz + 32 kHz, and emits a clip-level manifest carrying ALL the
Metadata.xlsx fields plus GBIF-canonical species + cleaned call-type multilabel.
Pure pandas/soundfile/librosa (no GCS) -> run on Slurm cpu. Outputs to a local
NFS root mirroring the GCS layout (audio_16k/, audio_32k/, *.csv); GCS staging is
a later `gsutil rsync` step.
"""
import argparse, datetime as dt, math, re
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, librosa

CLIP_S = 10
SRS = {16000: "audio_16k", 32000: "audio_32k"}
CALL_FIX = {"burstpulsed": "BurstPulse", "burstpulse": "BurstPulse"}  # merge dup

def dur_to_s(v):
    if isinstance(v, (dt.time, dt.datetime)):
        return v.hour*3600 + v.minute*60 + v.second + v.microsecond/1e6
    if isinstance(v, pd.Timedelta): return v.total_seconds()
    p = str(v).split(":")
    return int(p[0])*3600+int(p[1])*60+float(p[2]) if len(p)==3 else float("nan")

def clean_calltypes(v):
    if pd.isna(v): return ""
    out = []
    for t in re.split(r"[;,]", str(v)):
        t = t.strip()
        if not t: continue
        t = CALL_FIX.get(t.lower(), t)
        if t not in out: out.append(t)
    return "; ".join(sorted(out))

SNAKE = {
 "Species":"src_species","Common name":"src_common","Species code":"species_code",
 "WAV file name":"encounter_id","Date":"date","Latitude":"latitude","Longitude":"longitude",
 "Recording start (hh:mm:ss)":"rec_start","Recording end (hh:mm:ss)":"rec_end",
 "Sea state":"sea_state","Hydrophone depth (meters)":"hydrophone_depth_m",
 "Hydrophone sampling rate (kHz)":"hydrophone_sr_khz","Group size":"group_size",
 "Calves":"calves","Group behaviour":"group_behaviour","Boat presence":"boat_presence",
 "Boat type":"boat_type","Boat activity":"boat_activity","Notes":"notes",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/home/david_earthspecies_org/esp-data-dev/scratch_madeira/raw/Dataset")
    ap.add_argument("--meta", default="/home/david_earthspecies_org/esp-data-dev/scratch_madeira/dl/Metadata.xlsx")
    ap.add_argument("--tax", default="/home/david_earthspecies_org/esp-data-dev/scratch_madeira/species_taxonomy.csv")
    ap.add_argument("--out", default="/home/david_earthspecies_org/esp-data-dev/scratch_madeira/dataset")
    a = ap.parse_args()
    out = Path(a.out)
    for sub in SRS.values(): (out/sub).mkdir(parents=True, exist_ok=True)
    tax = pd.read_csv(a.tax).set_index("species_code")
    md = pd.read_excel(a.meta, sheet_name="Metadata")
    rows = []
    for _, m in md.iterrows():
        code = str(m["Species code"]).strip()
        enc = str(m["WAV file name"]).strip()
        wav = Path(a.raw)/code/f"{enc}.wav"
        if not wav.exists():
            print(f"!! missing {wav}"); continue
        info = sf.info(str(wav)); sr0 = info.samplerate
        n_clips = info.frames // (sr0*CLIP_S)
        t = tax.loc[code]
        base = {SNAKE[k]: m[k] for k in SNAKE if k in md.columns}
        base.update(dict(
            canonical_name=t["canonical_name"], species=t["canonical_name"],
            species_common=t["species_common"], family=t["family"], order=t["order"],
            taxon_class=t["class"], phylum=t["phylum"], kingdom=t["kingdom"],
            call_type=clean_calltypes(m["Call types"]), presence=1,
            source_sample_rate=sr0, encounter_duration_s=round(dur_to_s(m["Duration (mm:ss.ms)"]),2),
        ))
        for c in range(int(n_clips)):
            start = c*sr0*CLIP_S
            y, _ = sf.read(str(wav), start=start, frames=sr0*CLIP_S, dtype="float32")
            if y.ndim > 1: y = y.mean(axis=1)
            fn = f"{enc}_c{c:03d}"
            paths = {}
            for tsr, sub in SRS.items():
                yr = librosa.resample(y, orig_sr=sr0, target_sr=tsr, res_type="kaiser_best")
                sf.write(str(out/sub/f"{fn}.wav"), yr, tsr, subtype="PCM_16")
                paths[f"{tsr//1000}khz_path"] = f"{sub}/{fn}.wav"
            r = dict(base); r.update(dict(
                fn=fn, clip_index=c, clip_start_sec=c*CLIP_S, audio_duration=float(CLIP_S),
                sample_rate=32000, audio_fp=paths["32khz_path"],
                **{"16khz_path":paths["16khz_path"], "32khz_path":paths["32khz_path"]},
            ))
            rows.append(r)
        print(f"  {enc} ({t['canonical_name']}): {int(n_clips)} clips")
    df = pd.DataFrame(rows)
    df.to_csv(out/"all.csv", index=False)
    df.to_csv(out/"eval.csv", index=False)   # eval-only benchmark = full set
    pd.Series(sorted(df["canonical_name"].unique())).to_csv(out/"species_labels.csv", index=False, header=["canonical_name"])
    ct = sorted({t for v in df["call_type"] for t in v.split("; ") if t})
    (out/"call_type_vocab.txt").write_text("\n".join(ct))
    print(f"\nTOTAL {len(df)} clips, {df['canonical_name'].nunique()} species")
    print("per-species clips:\n", df.groupby("canonical_name").size().to_string())
    print("call-type vocab:", ct)
    print("wrote", out/"all.csv")

if __name__ == "__main__":
    main()
