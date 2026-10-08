# AI-Based Motor Vibration Fault Detection for Industries

A software prototype that analyses a motor's vibration signal and tells you:

1. **Fault type:** Normal · Inner Race Fault · Ball Fault · Outer Race Fault
2. **Fault size (severity):** 0.007" (0.18 mm), 0.014" (0.36 mm) or 0.021" (0.53 mm) crack, when a fault is found

It uses public lab data (the CWRU Bearing Data Center), hand-crafted signal features and Random Forest classifiers, plus a Streamlit web app where you upload a vibration CSV and get a prediction.

> This is a software prototype only. It works on recorded data files. It is **not** connected to any sensor or motor and does **not** do real-time monitoring.

---

## 1. Dataset

Case Western Reserve University (CWRU) Bearing Data Center, drive-end accelerometer, 12 kHz fault data, motor loads 0–3 HP. **40 files:**

| Class | 0.007" fault | 0.014" fault | 0.021" fault |
|---|---|---|---|
| Normal (no fault) | 97, 98, 99, 100 | | |
| Inner Race Fault | 105–108 | 169–172 | 209–212 |
| Ball Fault | 118–121 | 185–188 | 222–225 |
| Outer Race Fault (centred @ 6:00) | 130–133 | 197–200 | 234–237 |

Within each group, the files are for loads 0, 1, 2 and 3 HP in order. Files are downloaded from `https://engineering.case.edu/sites/default/files/<number>.mat` into `data/raw/`.

**Data checks done in this project**
- In each file the code uses the variable whose name ends with `_DE_time` (drive-end signal), with no hardcoded key names. `99.mat` contains two such variables (`X098_DE_time` and `X099_DE_time`), so the loader picks the one matching the file number.
- **The Normal files are recorded at 48 kHz, not 12 kHz.** Read as 12 kHz, their spectrum shows peaks at ¼, ½ and ¾ of the shaft speed. Read as 48 kHz, those same peaks are the normal 1×, 2× and 3× shaft harmonics, and the recording lengths match the 10 s fault recordings. So the Normal files are **downsampled from 48 kHz to 12 kHz** (`scipy.signal.resample_poly`, which includes an anti-aliasing filter) before use. Without this fix, a classifier can tell "Normal" apart partly by the sampling-rate difference rather than by the bearing condition.

After windowing: **2331 windows**. Normal 206, Inner Race 709, Ball 708, Outer Race 708. By fault size: 709 / 708 / 708.

## 2. How it works

```
.mat files ─► (Normal: 48→12 kHz) ─► 2048-sample windows ─► 14 features ─► StandardScaler ─► Random Forest ─► fault type
                                                                                          └─► Random Forest ─► fault size
```

1. **Windowing** (`src/prepare_data.py`): each signal is cut into non-overlapping windows of 2048 samples (≈ 0.17 s at 12 kHz). Each window keeps its class, load and fault size.
2. **Features** (`src/features.py`): for every window we compute
   - time domain: mean, RMS, variance, standard deviation, peak, peak-to-peak, crest factor, kurtosis, skewness
   - frequency domain: FFT dominant frequency, and spectral energy in 4 equal bands (≈ 0–1.5, 1.5–3, 3–4.5, 4.5–6 kHz)

   The same `extract_features()` function is used by training and by the app.
3. **Training** (`src/train.py`): `StandardScaler` + `RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")`. Balanced class weights are used because Normal has fewer windows than each fault class.
   - **Noise augmentation:** the final models are trained on every clean window plus one noisy copy of it, with random noise at a signal-to-noise ratio of 0–20 dB. This makes them much more robust to noisy signals (see test d).
4. **App** (`app.py`), which:
   - shows the waveform, FFT, RMS, peak, dominant frequency, predicted fault type and confidence, estimated fault size, and a status message
   - accepts **any sampling rate**: you enter the rate, and the signal is converted to 12 kHz
   - shows **"Result uncertain"** instead of a confident answer when the signal is unlike anything in training (any feature far outside the training range) or when confidence is below 60%

## 3. Setup

Python 3.9+.

```bash
pip install -r requirements.txt
```

Download the 40 `.mat` files listed above into `data/raw/` (macOS/Linux):

```bash
cd data/raw && for n in 97 98 99 100 105 106 107 108 118 119 120 121 130 131 132 133 169 170 171 172 185 186 187 188 197 198 199 200 209 210 211 212 222 223 224 225 234 235 236 237; do curl -L -o $n.mat https://engineering.case.edu/sites/default/files/$n.mat; done && cd ../..
```

## 4. Run each step

From the project folder:

```bash
python3 src/prepare_data.py      # windows per class and fault size
python3 src/features.py          # writes data/features.csv
python3 src/make_samples.py      # writes data/samples/*.csv (demo files)
python3 src/train.py             # trains, runs all tests, writes models/ and results/ (~20 s)
python3 -m streamlit run app.py  # web app at http://localhost:8501
```

In the app, upload any file from `data/samples/`. For `sample_normal_48kHz.csv`, first set "Sampling rate" to **48000**.

## 5. Results

All numbers come from `results/metrics.json` (produced by `python3 src/train.py`).

### Main results

| Test | What it checks | Test windows | Accuracy | Macro F1 |
|---|---|---|---|---|
| (a) Fault type, random 80/20 split | Standard accuracy | 467 | **99.57%** | 0.9965 |
| (b) Fault type, train 0–2 HP, test 3 HP | Works at a motor load it never saw | 591 | **98.82%** | 0.9901 |
| (f) Fault size, random 80/20 split | Severity estimation | 425 | **99.76%** | 0.9976 |
| (g) Fault size, train 0–2 HP, test 3 HP | Severity at an unseen load | 532 | **99.25%** | 0.9925 |

Per-class results:
- **(a):** precision / recall / F1 are 1.00 for Normal. Inner Race recall is 0.993, Ball precision 0.986, Outer Race recall 0.993.
- **(b):** Normal is 1.00. Ball precision is 0.962, Outer Race recall 0.966; the remaining scores are ≥ 0.98.

### Robustness tests

**(d) Noise:** random noise added to the 467 test windows of split (a).

| Signal-to-noise ratio | Clean | 20 dB | 10 dB | 5 dB | 0 dB (noise as strong as signal) |
|---|---|---|---|---|---|
| Model trained on clean data | 99.57% | 99.36% | 87.79% | 81.58% | 65.10% |
| Model trained with noisy copies (**used in the app**) | 99.36% | 99.36% | 98.93% | 97.86% | **92.51%** |

Noise augmentation costs 0.21 points on clean data and gains 27.4 points at 0 dB.

**(c) Unseen fault size:** train without one crack size, then test on that size.

| Fault size left out | Test windows | Accuracy |
|---|---|---|
| 0.007" | 709 | 39.92% |
| 0.014" | 708 | 32.06% |
| 0.021" | 708 | 63.14% |

**(e) Different sensor position:** train on the drive-end sensor, test on the fan-end sensor in the same recordings. 2331 windows, **19.61%** accuracy. Normal is still recognised (recall 94.7%), but the fault types are mostly confused.

Tests (c) and (e) show real weaknesses. The model only recognises fault sizes and sensor positions that appear in its training data (see Limitations). We also tried normalising each window's amplitude to improve (c) and (e), but it did not help consistently, so it was not adopted.

**Most important features** (final fault-type model): peak-to-peak (0.160), band 3 energy (0.132), variance (0.120), std (0.118), band 2 energy (0.115), peak (0.112).

### App check (demo samples)
The final models were trained on 2325 windows (4650 rows including the noisy copies). The demo windows in `data/samples/` were left out, so they are unseen data. All are 3 HP recordings. Results in the Streamlit app:

| Sample file | Predicted type | Predicted size | Status |
|---|---|---|---|
| sample_normal.csv | Normal (100.0%) | – | Motor healthy |
| sample_inner_race_fault_0_007in.csv | Inner Race Fault (100.0%) | 0.007" (94.0%) | Motor needs inspection |
| sample_ball_fault_0_007in.csv | Ball Fault (100.0%) | 0.007" (100.0%) | Motor needs inspection |
| sample_outer_race_fault_0_007in.csv | Outer Race Fault (100.0%) | 0.007" (100.0%) | Motor needs inspection |
| sample_outer_race_fault_0_014in.csv | Outer Race Fault (100.0%) | 0.014" (100.0%) | Motor needs inspection |
| sample_inner_race_fault_0_021in.csv | Inner Race Fault (100.0%) | 0.021" (100.0%) | Motor needs inspection |
| sample_normal_48kHz.csv (rate set to 48000) | Normal (100.0%) | – | Motor healthy |

Safety-check tests in the app:
- A pure 50 Hz sine wave gave **"Result uncertain"** (unfamiliar signal).
- The healthy sample made 100× louder also gave **"Result uncertain"**. Without the check, the model would have called it "Outer Race Fault" at 83% confidence.
- The 48 kHz sample uploaded with the rate wrongly left at 12000 was still called Normal, but only at 69.5% confidence, and the check did **not** flag it. Entering the correct sampling rate matters.

### Output files (`results/`)
- Confusion matrices:
  - `confusion_matrix_random_split.png`, `confusion_matrix_load_held_out.png`
  - `confusion_matrix_severity_random_split.png`, `confusion_matrix_severity_load_held_out.png`
  - `confusion_matrix_size_held_out_007/014/021.png`, `confusion_matrix_other_sensor.png`
- `noise_robustness.png`, `feature_importance.png`
- `time_<class>.png` and `fft_<class>.png` for each class (0.007" / Normal files at 0 HP)
- `metrics.json`

## 6. Project structure

```
data/raw/            40 CWRU .mat files
data/features.csv    features for every window
data/samples/        demo CSVs (unseen by the final models)
src/prepare_data.py  load signals, 48→12 kHz fix for Normal, windowing
src/features.py      shared feature extraction
src/make_samples.py  writes the demo CSVs
src/train.py         training, all tests, plots, final models
models/              model.joblib, scaler.joblib, severity_model.joblib,
                     severity_scaler.joblib, feature_ranges.json
results/             plots + metrics.json
app.py               Streamlit app
```

## 7. Limitations and future scope

**What this version improved:**
- More data: 3 crack sizes instead of 1 (40 files, 2331 windows).
- Severity: the fault size is now estimated (99.76% / 99.25%).
- Noise: noise augmentation keeps accuracy at 92.5% even when the noise is as strong as the signal (up from 65.1%).
- Any sampling rate is accepted and converted to 12 kHz.
- Unfamiliar signals are flagged as "uncertain" instead of being given a confident wrong answer.
- Normal data is corrected to the right sampling rate (48 → 12 kHz).

**Limitations that remain:**
- **Software prototype only:** it analyses recorded data. There is no sensor, no hardware and no real-time monitoring.
- **Lab data from one test rig:** it has not been tested on real industrial motors, which have different sizes, speeds, mountings and background noise.
- **Unseen fault sizes:** the model only recognises crack sizes it was trained on (32–63% on an unseen size, test c).
- **Sensor position:** a model trained on the drive-end sensor does not transfer to the fan-end sensor (19.6%, test e). A new installation would need training data from the same sensor position.
- **Only bearing faults:** other faults (misalignment, unbalance, looseness) are not classified. The "uncertain" check can catch some unusual signals, but it is a simple range check and not guaranteed.
- **Wrong sampling rate:** the user must enter the correct rate. A wrong rate is not always detected.

**Future scope:**
- An accelerometer (e.g. ADXL345 / MPU6050) with an ESP32 microcontroller to capture vibration from a real motor for real-time monitoring.
- An IoT/cloud dashboard for remote monitoring, alerts and trend history.
- Physics-based features (envelope spectrum at the bearing defect frequencies BPFI, BPFO and BSF) to generalise better to new fault sizes and sensor positions.
- Training on more datasets and on real industrial data, and adding more fault types.
