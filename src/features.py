"""
Step 2: Turn each 2048-sample window into 14 numbers (features).

extract_features() is the ONE shared function used by both
training (src/train.py) and the web app (app.py), so the model always
sees features computed exactly the same way.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# Order of the feature columns (the model expects exactly this order)
FEATURE_NAMES = [
    "mean", "rms", "variance", "std", "peak", "peak_to_peak",
    "crest_factor", "kurtosis", "skewness", "dominant_freq",
    "band1_energy", "band2_energy", "band3_energy", "band4_energy",
]


def compute_fft(window, sampling_rate):
    """Return (frequencies, magnitudes) of the one-sided FFT of a window."""
    magnitudes = np.abs(np.fft.rfft(window)) / len(window)
    freqs = np.fft.rfftfreq(len(window), d=1.0 / sampling_rate)
    return freqs, magnitudes


def extract_features(window, sampling_rate=12000):
    """Compute the feature list for one window of vibration samples."""
    window = np.asarray(window, dtype=float)

    # ---- Time-domain features ----
    mean = np.mean(window)                      # average value
    rms = np.sqrt(np.mean(window ** 2))         # overall vibration energy level
    variance = np.var(window)                   # spread of values
    std = np.std(window)                        # spread (same units as signal)
    peak = np.max(np.abs(window))               # largest absolute spike
    peak_to_peak = np.max(window) - np.min(window)  # top to bottom distance
    crest_factor = peak / rms if rms > 0 else 0.0   # how "spiky" the signal is
    kurt = kurtosis(window)                     # high when sharp impacts exist
    skewness = skew(window)                     # asymmetry of the values

    # ---- Frequency-domain features ----
    freqs, mags = compute_fft(window, sampling_rate)
    # Dominant frequency = frequency with the biggest magnitude (skip 0 Hz / DC)
    dominant_freq = freqs[1:][np.argmax(mags[1:])]

    # Split the spectrum (0 to 6 kHz) into 4 equal bands and sum energy in each
    power = mags ** 2
    bands = np.array_split(power, 4)
    band_energies = [np.sum(b) for b in bands]

    return [mean, rms, variance, std, peak, peak_to_peak, crest_factor,
            kurt, skewness, dominant_freq] + band_energies


def build_feature_table(windows, info, sampling_rate=12000):
    """Compute features for every window and return them with the window info."""
    rows = [extract_features(w, sampling_rate) for w in windows]
    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return pd.concat([df, info.reset_index(drop=True)], axis=1)


if __name__ == "__main__":
    from prepare_data import load_all_windows, SAMPLING_RATE

    windows, info = load_all_windows()
    df = build_feature_table(windows, info, SAMPLING_RATE)

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows x {len(FEATURE_NAMES)} features to data/features.csv")
