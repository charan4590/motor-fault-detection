"""
Streamlit web app: upload a vibration CSV and get a bearing fault prediction.

Run with:  streamlit run app.py
"""
import os
import sys
import json
from fractions import Fraction
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import joblib
import matplotlib.pyplot as plt
import streamlit as st

# Let Python find our own src/ folder so we reuse the SAME feature function
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from features import extract_features, compute_fft, FEATURE_NAMES  # noqa: E402

SAMPLING_RATE = 12000   # the model was trained on 12 kHz data
WINDOW_SIZE = 2048      # and on windows of 2048 samples
MIN_CONFIDENCE = 0.60   # below this, the result is shown as "uncertain"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
INCH_TO_MM = 25.4


@st.cache_resource
def load_models():
    """Load the trained models, scalers and feature ranges once."""
    def load(name):
        return joblib.load(os.path.join(MODELS_DIR, name))
    with open(os.path.join(MODELS_DIR, "feature_ranges.json")) as f:
        ranges = json.load(f)
    return (load("model.joblib"), load("scaler.joblib"),
            load("severity_model.joblib"), load("severity_scaler.joblib"), ranges)


def to_12khz(signal, rate):
    """Convert a signal recorded at any sampling rate to 12 kHz (the model's rate)."""
    if rate == SAMPLING_RATE:
        return signal
    ratio = Fraction(SAMPLING_RATE, int(rate)).limit_denominator(1000)
    return resample_poly(signal, ratio.numerator, ratio.denominator)


def unfamiliar_features(features, ranges):
    """Names of features that fall well outside what the model saw in training.

    A 10% margin is allowed beyond the training min/max.
    """
    outside = set()
    for i, name in enumerate(FEATURE_NAMES):
        low, high = ranges[name]
        margin = 0.1 * (high - low)
        if np.any(features[:, i] < low - margin) or np.any(features[:, i] > high + margin):
            outside.add(name)
    return sorted(outside)


def read_signal(uploaded_file):
    """Read the uploaded CSV into a 1-D numpy array. Raises ValueError if bad."""
    try:
        df = pd.read_csv(uploaded_file)
        # If the "header" is actually a number, the file has no header row:
        # read it again so that first value is kept as a sample
        if pd.to_numeric(pd.Series(df.columns), errors="coerce").notna().all():
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=None)
    except Exception:
        raise ValueError("Could not read the file. Please upload a valid CSV file.")
    if df.shape[1] != 1:
        raise ValueError(f"The CSV must have exactly one column, but it has {df.shape[1]}.")
    values = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    if values.isna().any():
        raise ValueError("The column contains empty or non-numeric values.")
    if len(values) < WINDOW_SIZE:
        raise ValueError(f"Need at least {WINDOW_SIZE} samples, but the file has only {len(values)}.")
    return values.to_numpy(dtype=float)


def window_features(signal):
    """Split the signal into 2048-sample windows and compute features for each."""
    n_windows = len(signal) // WINDOW_SIZE
    windows = signal[: n_windows * WINDOW_SIZE].reshape(n_windows, WINDOW_SIZE)
    return np.array([extract_features(w, SAMPLING_RATE) for w in windows])


def predict(features, model, scaler):
    """Predict each window and average the probabilities over all windows."""
    probabilities = model.predict_proba(scaler.transform(features)).mean(axis=0)
    best = np.argmax(probabilities)
    return model.classes_[best], probabilities[best], probabilities


# ---------------------------- Page layout ----------------------------
st.set_page_config(page_title="Motor Fault Detection AI", layout="wide")
st.title("Motor Fault Detection AI")
st.write("Upload a CSV file with **one column** of motor vibration samples "
         "(acceleration in g, from a drive-end accelerometer). "
         "The model classifies the bearing as Normal, Inner Race Fault, "
         "Ball Fault or Outer Race Fault, and estimates the fault size.")

model, scaler, sev_model, sev_scaler, ranges = load_models()
rate = st.number_input("Sampling rate of your data (Hz)", min_value=1000,
                       max_value=200000, value=SAMPLING_RATE, step=1000,
                       help="Data at other rates is converted to 12 kHz before prediction.")
uploaded = st.file_uploader("Upload vibration CSV", type=["csv"])

if uploaded is not None:
    try:
        raw = read_signal(uploaded)
        signal = to_12khz(raw, rate)
        if len(signal) < WINDOW_SIZE:
            raise ValueError(f"After converting to 12 kHz the signal has only {len(signal)} "
                             f"samples; at least {WINDOW_SIZE} are needed. "
                             f"Upload at least {int(np.ceil(WINDOW_SIZE * rate / SAMPLING_RATE))} samples.")
    except ValueError as err:
        st.error(f"Sorry, this file can't be used: {err}")
        st.stop()
    if rate != SAMPLING_RATE:
        st.info(f"Converted {len(raw)} samples at {rate} Hz to {len(signal)} samples at 12 kHz.")

    # ---- Plots: waveform and FFT ----
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Vibration waveform")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(np.arange(len(signal)) / SAMPLING_RATE * 1000, signal, linewidth=0.6)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        st.pyplot(fig)
    with col2:
        st.subheader("Frequency spectrum (FFT)")
        freqs, mags = compute_fft(signal[:WINDOW_SIZE], SAMPLING_RATE)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(freqs, mags, linewidth=0.6, color="#c05621")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        fig.tight_layout()
        st.pyplot(fig)

    # ---- Key numbers (from the first 2048-sample window) ----
    feats = dict(zip(["mean", "rms", "variance", "std", "peak", "peak_to_peak",
                      "crest_factor", "kurtosis", "skewness", "dominant_freq"],
                     extract_features(signal[:WINDOW_SIZE], SAMPLING_RATE)))
    m1, m2, m3 = st.columns(3)
    m1.metric("RMS", f"{feats['rms']:.4f}")
    m2.metric("Peak", f"{feats['peak']:.4f}")
    m3.metric("Dominant frequency", f"{feats['dominant_freq']:.1f} Hz")

    # ---- Prediction ----
    features = window_features(signal)
    n_windows = len(features)
    label, confidence, probs = predict(features, model, scaler)
    st.subheader("Prediction")
    st.write(f"Predicted condition: **{label}** — confidence **{confidence * 100:.1f}%** "
             f"(averaged over {n_windows} window{'s' if n_windows > 1 else ''})")

    # Fault size (severity) is only estimated when a fault is predicted
    if label != "Normal":
        size, size_conf, _ = predict(features, sev_model, sev_scaler)
        st.write(f"Estimated fault size: **{size} inch** ({float(size) * INCH_TO_MM:.2f} mm) "
                 f"— confidence **{size_conf * 100:.1f}%**")

    # Safety checks: is this signal similar to what the model was trained on?
    strange = unfamiliar_features(features, ranges)
    if strange:
        st.warning("⚠️ Result uncertain: this signal is unlike the training data "
                   f"(out of range: {', '.join(strange)}). Check the sampling rate and "
                   "sensor position, or it may be a fault type the model does not know. "
                   "Manual inspection recommended.")
    elif confidence < MIN_CONFIDENCE:
        st.warning(f"⚠️ Result uncertain: confidence is below {MIN_CONFIDENCE * 100:.0f}%. "
                   "Manual inspection recommended.")
    elif label == "Normal":
        st.success("✅ Motor healthy")
    else:
        st.error(f"⚠️ Motor needs inspection — {label} detected")

    st.caption("Probability for each class:")
    st.dataframe(pd.DataFrame({"Class": model.classes_,
                               "Probability (%)": np.round(probs * 100, 1)}),
                 hide_index=True)
