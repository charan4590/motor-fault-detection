"""
Step 3: Write demo CSVs into data/samples/ for the app.

Each CSV has a single column "vibration". The windows come from 3 HP
recordings (test data) and are left out of the final models' training in
train.py, so the demo runs on data the model has never seen.
"""
import os
import pandas as pd

from prepare_data import (FILES, WINDOW_SIZE, DEMO_FILES, DEMO_WINDOW_INDEX,
                          load_signal, make_windows)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "samples")
os.makedirs(OUT_DIR, exist_ok=True)

# 12 kHz samples (2048 values each), one per demo file
for file_number in DEMO_FILES:
    label, load, size = FILES[file_number]
    window = make_windows(load_signal(file_number))[DEMO_WINDOW_INDEX]
    name = label.lower().replace(" ", "_")
    if size != "none":
        name += f"_{size.replace('0.', '0_')}in"      # e.g. inner_race_fault_0_007in
    path = os.path.join(OUT_DIR, f"sample_{name}.csv")
    pd.DataFrame({"vibration": window}).to_csv(path, index=False)
    print(f"Saved {os.path.basename(path)}  ({len(window)} samples, 12 kHz, {file_number}.mat, {load} HP)")

# A real 48 kHz sample: the ORIGINAL (not downsampled) recording of the same
# Normal demo window. 2048 samples at 12 kHz = 4 x 2048 = 8192 samples at 48 kHz.
# Use it in the app with "Sampling rate" set to 48000.
raw_48k = load_signal(100, to_12khz=False)
start = DEMO_WINDOW_INDEX * WINDOW_SIZE * 4
segment = raw_48k[start: start + WINDOW_SIZE * 4]
path = os.path.join(OUT_DIR, "sample_normal_48kHz.csv")
pd.DataFrame({"vibration": segment}).to_csv(path, index=False)
print(f"Saved {os.path.basename(path)}  ({len(segment)} samples, 48 kHz, 100.mat, 3 HP)")
