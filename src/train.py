"""
Step 4: Train the Random Forest models and test them in several ways.

Fault-TYPE model (Normal / Inner Race / Ball / Outer Race):
  (a) Random split: 80% of windows for training, 20% for testing (stratified)
  (b) Load held out: train on loads 0-2 HP, test on 3 HP
  (c) Fault size held out: e.g. train without any 0.014" faults, test on 0.014"
  (d) Noise test: add random noise to the test windows and check accuracy
      (model trained on clean data vs model trained with extra noisy copies)
  (e) Other sensor: train on the drive-end sensor, test on the fan-end sensor

Fault-SIZE (severity) model, only for faulty windows (0.007 / 0.014 / 0.021 inch):
  (f) Random 80/20 split and (g) load held out

Outputs go to results/ (plots + metrics.json) and models/.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw plots to files, no window needed
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

from features import FEATURE_NAMES, extract_features, compute_fft, build_feature_table
from prepare_data import (CLASSES, FAULT_SIZES, FILES, SAMPLING_RATE, WINDOW_SIZE,
                          DEMO_FILES, DEMO_WINDOW_INDEX, load_signal, load_all_windows)

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

NOISE_LEVELS_DB = [20, 10, 5, 0]   # signal-to-noise ratios for the noise test
SIZE_NAMES = [f'{s}"' for s in FAULT_SIZES]


# ----------------------------- helpers -----------------------------

def train_model(X_train, y_train):
    """Scale the features, then fit a Random Forest. Returns (scaler, model)."""
    scaler = StandardScaler()                 # makes every feature mean 0, std 1
    X_scaled = scaler.fit_transform(X_train)  # learn scaling from training data only
    # class_weight="balanced" so every class counts equally during training
    model = RandomForestClassifier(n_estimators=100, random_state=42,
                                   class_weight="balanced")
    model.fit(X_scaled, y_train)
    return scaler, model


def add_noise(window, snr_db, rng):
    """Add random (Gaussian) noise so the signal-to-noise ratio is snr_db.

    SNR 20 dB = noise has 1% of the signal power (a little noise)
    SNR  0 dB = noise is as strong as the signal itself (very noisy)
    """
    signal_power = np.mean(window ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return window + rng.normal(0, np.sqrt(noise_power), size=len(window))


def noisy_features(windows, snr_values, rng):
    """Features of noisy copies of the windows (one SNR value per window)."""
    return np.array([extract_features(add_noise(w, s, rng), SAMPLING_RATE)
                     for w, s in zip(windows, snr_values)])


def train_with_noise(windows, X, y, rng):
    """Train on the clean windows PLUS one noisy copy of each (SNR 0-20 dB).

    This "data augmentation" teaches the model what noisy signals look like.
    Returns (scaler, model, X_used) where X_used is all training features.
    """
    snrs = rng.uniform(0, 20, size=len(windows))
    X_all = np.vstack([X, noisy_features(windows, snrs, rng)])
    y_all = np.concatenate([y, y])
    scaler, model = train_model(X_all, y_all)
    return scaler, model, X_all


def evaluate(name, title, scaler, model, X_test, y_test, labels, display_labels=None,
             save_plot=True):
    """Print accuracy + report, optionally save a confusion matrix PNG, return metrics."""
    y_pred = model.predict(scaler.transform(X_test))
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=labels, digits=4,
                                   zero_division=0)
    report_dict = classification_report(y_test, y_pred, labels=labels,
                                        output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"\n===== {title} =====")
    print(f"Test windows: {len(y_test)}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(report)

    if save_plot:
        # Confusion matrix: rows = true class, columns = predicted class
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(cm, display_labels=display_labels or labels).plot(
            ax=ax, cmap="Blues", colorbar=False, xticks_rotation=20)
        ax.set_title(f"{title}\nAccuracy = {acc * 100:.2f}%")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{name}.png"), dpi=150)
        plt.close(fig)

    return {
        "test_windows": int(len(y_test)),
        "accuracy": round(float(acc), 4),
        "per_class": {str(c): {"precision": round(report_dict[str(c)]["precision"], 4),
                               "recall": round(report_dict[str(c)]["recall"], 4),
                               "f1": round(report_dict[str(c)]["f1-score"], 4),
                               "support": int(report_dict[str(c)]["support"])}
                      for c in labels},
        "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
        "confusion_matrix": cm.tolist(),
    }


def plot_feature_importance(model):
    """Bar chart of how much each feature helped the Random Forest."""
    order = np.argsort(model.feature_importances_)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(FEATURE_NAMES)[order], model.feature_importances_[order],
            color="#2b6cb0")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importance (final fault-type model)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)


def plot_noise_results(noise_results):
    """Line chart: accuracy vs noise level for both models."""
    levels = ["clean"] + [f"{d} dB" for d in NOISE_LEVELS_DB]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label, color, dy in [("clean_trained", "Trained on clean data", "#c05621", -14),
                                  ("noise_trained", "Trained with noisy copies", "#2b6cb0", 7)]:
        accs = [noise_results[key][lv] * 100 for lv in levels]
        ax.plot(levels, accs, marker="o", label=label, color=color)
        for x, a in zip(levels, accs):
            ax.annotate(f"{a:.1f}", (x, a), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=8, color=color)
    ax.set_xlabel("Test signal quality (signal-to-noise ratio; lower = noisier)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Fault-type accuracy on noisy test signals")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "noise_robustness.png"), dpi=150)
    plt.close(fig)


def plot_signals_per_class():
    """One time-domain plot and one FFT plot per class (0.007" files at 0 HP)."""
    for file_number, (label, load, size) in FILES.items():
        if load != 0 or size not in ("none", "0.007"):
            continue
        window = load_signal(file_number)[:WINDOW_SIZE]
        slug = label.lower().replace(" ", "_")
        time_ms = np.arange(WINDOW_SIZE) / SAMPLING_RATE * 1000

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(time_ms, window, linewidth=0.7)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Acceleration (g)")
        ax.set_title(f"{label} - time domain (file {file_number}.mat, 0 HP)")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"time_{slug}.png"), dpi=150)
        plt.close(fig)

        freqs, mags = compute_fft(window, SAMPLING_RATE)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(freqs, mags, linewidth=0.7, color="#c05621")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title(f"{label} - FFT (file {file_number}.mat, 0 HP)")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"fft_{slug}.png"), dpi=150)
        plt.close(fig)


# ------------------------------- main -------------------------------

if __name__ == "__main__":
    # Clean features (from features.py) and the raw windows (needed to add noise)
    df = pd.read_csv(os.path.join(ROOT, "data", "features.csv"),
                     dtype={"fault_size": str})
    windows, info = load_all_windows("DE")
    # Make sure row i of features.csv is window i
    assert (df["file"].values == info["file"].values).all()
    assert (df["window_index"].values == info["window_index"].values).all()

    X = df[FEATURE_NAMES].values
    y = df["label"].values
    loads = df["load"].values
    sizes = df["fault_size"].values
    is_fault = y != "Normal"

    metrics = {
        "total_windows": int(len(df)),
        "total_files": len(FILES),
        "windows_per_class": {c: int(np.sum(y == c)) for c in CLASSES},
        "windows_per_fault_size": {s: int(np.sum(sizes == s)) for s in FAULT_SIZES},
    }

    # ---- (a) Stratified 80/20 random split ----
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.2,
                                           stratify=y, random_state=42)
    scaler_a, model_a = train_model(X[idx_train], y[idx_train])
    metrics["a_random_split"] = evaluate(
        "random_split", "(a) Fault type - random 80/20 split",
        scaler_a, model_a, X[idx_test], y[idx_test], CLASSES)

    # ---- (b) Load held out: train on 0-2 HP, test on 3 HP ----
    train_mask = loads <= 2
    scaler_b, model_b = train_model(X[train_mask], y[train_mask])
    metrics["b_load_held_out"] = evaluate(
        "load_held_out", "(b) Fault type - train 0-2 HP, test 3 HP",
        scaler_b, model_b, X[~train_mask], y[~train_mask], CLASSES)

    # ---- (c) Fault size held out: the model never sees this crack size ----
    metrics["c_fault_size_held_out"] = {}
    for held_out in FAULT_SIZES:
        test_mask = sizes == held_out           # only faulty windows of that size
        train_mask = ~test_mask                 # Normal + the other two sizes
        scaler_c, model_c = train_model(X[train_mask], y[train_mask])
        metrics["c_fault_size_held_out"][held_out] = evaluate(
            f"size_held_out_{held_out.replace('0.', '')}",
            f'(c) Fault type - trained without {held_out}" faults, tested on them',
            scaler_c, model_c, X[test_mask], y[test_mask], CLASSES)

    # ---- (d) Noise test on the test set of split (a) ----
    rng_train = np.random.default_rng(0)        # fixed seeds = repeatable results
    scaler_d, model_d, _ = train_with_noise(windows[idx_train], X[idx_train],
                                            y[idx_train], rng_train)
    noise_results = {"clean_trained": {}, "noise_trained": {}}
    for level in ["clean"] + NOISE_LEVELS_DB:
        if level == "clean":
            X_test, key = X[idx_test], "clean"
        else:
            rng_test = np.random.default_rng(100 + level)
            X_test = noisy_features(windows[idx_test], [level] * len(idx_test), rng_test)
            key = f"{level} dB"
        for name, sc, md in [("clean_trained", scaler_a, model_a),
                             ("noise_trained", scaler_d, model_d)]:
            acc = accuracy_score(y[idx_test], md.predict(sc.transform(X_test)))
            noise_results[name][key] = round(float(acc), 4)
    print("\n===== (d) Noise test (accuracy on test set of split a) =====")
    print(pd.DataFrame(noise_results))
    metrics["d_noise_test"] = noise_results
    plot_noise_results(noise_results)

    # ---- (e) Different sensor: fan-end (FE) accelerometer, same recordings ----
    fe_windows, fe_info = load_all_windows("FE")
    fe_df = build_feature_table(fe_windows, fe_info, SAMPLING_RATE)
    X_fe, y_fe = fe_df[FEATURE_NAMES].values, fe_df["label"].values
    scaler_e, model_e = train_model(X, y)       # all drive-end windows
    metrics["e_other_sensor"] = evaluate(
        "other_sensor", "(e) Fault type - train drive-end sensor, test fan-end sensor",
        scaler_e, model_e, X_fe, y_fe, CLASSES)

    # ---- (f), (g) Fault size (severity) model, faulty windows only ----
    Xf, sf, loads_f = X[is_fault], sizes[is_fault], loads[is_fault]
    tr, te = train_test_split(np.arange(len(sf)), test_size=0.2, stratify=sf,
                              random_state=42)
    sc, md = train_model(Xf[tr], sf[tr])
    metrics["f_severity_random_split"] = evaluate(
        "severity_random_split", "(f) Fault size - random 80/20 split",
        sc, md, Xf[te], sf[te], FAULT_SIZES, SIZE_NAMES)
    sc, md = train_model(Xf[loads_f <= 2], sf[loads_f <= 2])
    metrics["g_severity_load_held_out"] = evaluate(
        "severity_load_held_out", "(g) Fault size - train 0-2 HP, test 3 HP",
        sc, md, Xf[loads_f == 3], sf[loads_f == 3], FAULT_SIZES, SIZE_NAMES)

    # ---- Final models used by the app ----
    # Trained on all windows EXCEPT the demo windows (so the demo is unseen data),
    # with one extra noisy copy of each window (noise augmentation).
    is_demo = (df["file"].isin(DEMO_FILES) & (df["window_index"] == DEMO_WINDOW_INDEX)).values
    keep = ~is_demo
    rng_final = np.random.default_rng(1)
    final_scaler, final_model, X_used = train_with_noise(windows[keep], X[keep], y[keep],
                                                         rng_final)
    keep_f = keep & is_fault
    sev_scaler, sev_model, _ = train_with_noise(windows[keep_f], X[keep_f], sizes[keep_f],
                                                rng_final)
    joblib.dump(final_model, os.path.join(MODELS_DIR, "model.joblib"))
    joblib.dump(final_scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(sev_model, os.path.join(MODELS_DIR, "severity_model.joblib"))
    joblib.dump(sev_scaler, os.path.join(MODELS_DIR, "severity_scaler.joblib"))

    # Min / max of every feature the final model saw. The app warns the user
    # if an uploaded signal falls far outside these ranges ("unfamiliar signal").
    ranges = {f: [float(X_used[:, i].min()), float(X_used[:, i].max())]
              for i, f in enumerate(FEATURE_NAMES)}
    with open(os.path.join(MODELS_DIR, "feature_ranges.json"), "w") as f:
        json.dump(ranges, f, indent=2)

    metrics["final_model_training_windows"] = int(keep.sum())
    metrics["final_model_training_rows_with_noisy_copies"] = int(len(X_used))
    plot_feature_importance(final_model)
    plot_signals_per_class()

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved models/ (fault-type + severity models, scaler, feature ranges), "
          "results/*.png and results/metrics.json")
