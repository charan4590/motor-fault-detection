"""
Step 1: Load the CWRU .mat files and cut each signal into windows.

Each window is 2048 samples long (about 0.17 seconds at 12 kHz).
Every window gets:
  - a class label (Normal / Inner Race / Ball / Outer Race)
  - the motor load (0-3 HP)
  - the fault size in inches (0.007, 0.014, 0.021), or "none" for Normal
"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import resample_poly

# Folder holding the downloaded .mat files
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Number of samples in one window
WINDOW_SIZE = 2048

# Sampling rate of the accelerometers (samples per second)
SAMPLING_RATE = 12000

# File number -> (class name, motor load in HP, fault size in inches)
# Files for each fault type are listed in load order 0, 1, 2, 3 HP.
FILES = {}
for numbers, label, size in [
    ([97, 98, 99, 100], "Normal", "none"),
    ([105, 106, 107, 108], "Inner Race Fault", "0.007"),
    ([118, 119, 120, 121], "Ball Fault", "0.007"),
    ([130, 131, 132, 133], "Outer Race Fault", "0.007"),
    ([169, 170, 171, 172], "Inner Race Fault", "0.014"),
    ([185, 186, 187, 188], "Ball Fault", "0.014"),
    ([197, 198, 199, 200], "Outer Race Fault", "0.014"),
    ([209, 210, 211, 212], "Inner Race Fault", "0.021"),
    ([222, 223, 224, 225], "Ball Fault", "0.021"),
    ([234, 235, 236, 237], "Outer Race Fault", "0.021"),
]:
    for load, number in enumerate(numbers):
        FILES[number] = (label, load, size)

# Fixed order of class names and fault sizes (used for plots and reports)
CLASSES = ["Normal", "Inner Race Fault", "Ball Fault", "Outer Race Fault"]
FAULT_SIZES = ["0.007", "0.014", "0.021"]

# Demo samples for the app: one window (index 10) from several 3 HP files.
# 3 HP is the test set of the load-held-out evaluation, and the final models
# are trained WITHOUT these windows, so the live demo uses unseen data.
DEMO_FILES = [100, 108, 121, 133, 200, 212]
DEMO_WINDOW_INDEX = 10


def load_signal(file_number, sensor="DE", to_12khz=True):
    """Load one vibration signal from a .mat file.

    sensor="DE" -> drive-end accelerometer (used for training)
    sensor="FE" -> fan-end accelerometer (used to test a different sensor position)
    to_12khz=False returns the Normal files at their original 48 kHz rate
    """
    path = os.path.join(RAW_DIR, f"{file_number}.mat")
    mat = sio.loadmat(path)

    # Find every variable whose name ends with e.g. "_DE_time"
    keys = [k for k in mat.keys() if k.endswith(f"_{sensor}_time")]
    if not keys:
        raise ValueError(f"No _{sensor}_time variable found in {path}")

    # Some files hold more than one (e.g. 99.mat also contains X098_DE_time).
    # Prefer the key that contains this file's own number, e.g. "X099_DE_time".
    key = keys[0]
    for k in keys:
        if k.startswith(f"X{file_number:03d}_"):
            key = k

    # The signal is stored as a column (N x 1); flatten it to a 1-D array
    signal = mat[key].flatten()

    # The Normal (healthy) files were recorded at 48 kHz, not 12 kHz.
    # (Evidence: their spectrum shows the shaft-speed peak at 1/4 of the expected
    # frequency when read as 12 kHz, and they are 4x longer than the fault files.)
    # Downsample them by 4 so EVERY class is at 12 kHz. resample_poly also applies
    # an anti-aliasing low-pass filter.
    if FILES[file_number][0] == "Normal" and to_12khz:
        signal = resample_poly(signal, up=1, down=4)
    return signal


def make_windows(signal, window_size=WINDOW_SIZE):
    """Cut a signal into non-overlapping windows. Leftover samples are dropped."""
    n_windows = len(signal) // window_size
    return signal[: n_windows * window_size].reshape(n_windows, window_size)


def load_all_windows(sensor="DE"):
    """Return (windows, info) for all files.

    windows: 2-D array, one row per window
    info:    table with label, load, fault_size, file, window_index per window
    """
    all_windows, rows = [], []
    for file_number, (label, load, size) in FILES.items():
        windows = make_windows(load_signal(file_number, sensor))
        all_windows.append(windows)
        for i in range(len(windows)):
            rows.append({"label": label, "load": load, "fault_size": size,
                         "file": file_number, "window_index": i})
    return np.vstack(all_windows), pd.DataFrame(rows)


if __name__ == "__main__":
    windows, info = load_all_windows()
    print(f"Total windows: {len(windows)} (each {WINDOW_SIZE} samples) from {len(FILES)} files")
    print(pd.crosstab(info["label"], info["fault_size"], margins=True))
