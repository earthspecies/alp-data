import re

files = [
    "esp_data/datasets/birdeep.py",
    "esp_data/datasets/wabad.py",
    "esp_data/datasets/xeno_canto_annotated_jeantet_23.py"
]

def fix_file(filepath):
    with open(filepath, "r") as f:
        text = f.read()

    # We want to replace the conflict section with the combined logic.
    # The combined logic: check for presampled, fallback to logic that checks window.
    # Actually, we can just replace the whole conflicting block manually.
    
    if "birdeep.py" in filepath:
        text = re.sub(
            r'<<<<<<< HEAD.*?=======\s+audio, sr = read_audio\(audio_path\)\s+>>>>>>> origin/beans-pro',
            '''        window_start = row.get("window_start_sec")
        window_end = row.get("window_end_sec")

        if window_start is not None and window_end is not None:
            audio, sr = read_audio(
                audio_path,
                start_time=float(window_start),
                end_time=float(window_end),
            )
        else:
            audio, sr = read_audio(audio_path)''', text, flags=re.DOTALL
        )
        text = re.sub(
            r'<<<<<<< HEAD\s+raw_st = row\.get\("selection_table"\).*?=======\s+st = pd.read_csv\(StringIO\(row\["selection_table"\]\), sep="\\\\t"\)\s+audio_dur = len\(audio\) / float\(sr\)\s+st = st\[st\["Begin Time \(s\)"\] < audio_dur\]\.copy\(\)\s+>>>>>>> origin/beans-pro',
            '''        raw_st = row.get("selection_table")
        if raw_st is not None:
            if isinstance(raw_st, str):
                st = pd.read_csv(StringIO(raw_st), sep="\\t")
            elif isinstance(raw_st, pd.DataFrame):
                st = raw_st
            else:
                st = pd.DataFrame()

            # Clip events outside audio
            audio_dur = len(audio) / float(sr)
            if "Begin Time (s)" in st.columns:
                st = st[st["Begin Time (s)"] < audio_dur].copy()

            row["selection_table"] = st''', text, flags=re.DOTALL
        )
        text = re.sub(
            r'<<<<<<< HEAD\s+row\["sample_rate"\] = sample_rate\s+=======\s+row\["sample_rate"\] = sr\s+row\["selection_table"\] = st\s+>>>>>>> origin/beans-pro',
            '''        row["sample_rate"] = sr''', text, flags=re.DOTALL
        )
    
    elif "wabad.py" in filepath:
        text = re.sub(
            r'<<<<<<< HEAD\s+When the row contains.*?=======\s+.*?Parameters.*?Returns.*?dict\[str, Any\]\s+The processed row\.\s+"""\s+use_presampled = False.*?audio, sr = read_audio\(audio_path\)\s+>>>>>>> origin/beans-pro',
            '''
        When the row contains ``window_start_sec`` / ``window_end_sec``
        (set by the ``window_annotations`` transform), only the
        corresponding audio segment is loaded from disk/GCS instead of
        the full recording.

        Parameters
        ----------
        row : dict[str, Any]
            A dictionary representing a single row of the dataset.

        Returns
        -------
        dict[str, Any]
            The processed row.
        """
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            if path_column in row and row[path_column] is not None and row[path_column] != "":
                audio_path = anypath(self.data_root) / row[path_column]
                use_presampled = True

        if not use_presampled:
            audio_path = anypath(self.data_root) / row[self._originals_path_column]

        window_start = row.get("window_start_sec")
        window_end = row.get("window_end_sec")

        if window_start is not None and window_end is not None:
            audio, sr = read_audio(
                audio_path, start_time=float(window_start), end_time=float(window_end)
            )
        else:
            audio, sr = read_audio(audio_path)
''', text, flags=re.DOTALL
        )
        text = re.sub(
            r'<<<<<<< HEAD\s+row\["audio"\] = audio.*?=======\s+st = pd.read_csv\(StringIO\(row\["selection_table"\]\), sep="\\\\t"\).*?row\["selection_table"\] = st\s+>>>>>>> origin/beans-pro',
            '''        row["audio"] = audio
        row["sample_rate"] = sr

        raw_st = row.get("selection_table")
        if raw_st is not None:
            if isinstance(raw_st, str):
                st = pd.read_csv(StringIO(raw_st), sep="\\t")
            elif isinstance(raw_st, pd.DataFrame):
                st = raw_st
            else:
                st = pd.DataFrame()

            audio_dur = len(audio) / float(sr)
            if "Begin Time (s)" in st.columns:
                st = st[st["Begin Time (s)"] < audio_dur].copy()
            row["selection_table"] = st''', text, flags=re.DOTALL
        )

    elif "xeno_canto_annotated_jeantet_23.py" in filepath:
        text = re.sub(
            r'<<<<<<< HEAD.*?=======\s+audio, sr = read_audio\(audio_path\)\s+>>>>>>> origin/beans-pro',
            '''        window_start = row.get("window_start_sec")
        window_end = row.get("window_end_sec")

        # Read either the full recording or a requested sub-window.
        if window_start is not None and window_end is not None:
            audio, sr = read_audio(
                audio_path,
                start_time=float(window_start),
                end_time=float(window_end),
            )
        else:
            audio, sr = read_audio(audio_path)''', text, flags=re.DOTALL
        )

    with open(filepath, "w") as f:
        f.write(text)

for fp in files:
    fix_file(fp)

