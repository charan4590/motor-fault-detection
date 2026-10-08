# Presentation Content: AI-Based Motor Vibration Fault Detection for Industries

10 slides. Every number comes from `results/metrics.json` and the app test run. Image file names refer to the `results/` folder.

---

## Slide 1: Title
**AI-Based Motor Vibration Fault Detection for Industries**
- Detecting bearing fault **type** and **severity** from vibration signals using Machine Learning
- Software prototype on the public CWRU bearing dataset
- Name / Roll No. / Department / College / Date

*Image: none (or `fft_inner_race_fault.png` as a faded background)*

---

## Slide 2: Introduction
- Electric motors drive pumps, fans, compressors and conveyors in almost every industry.
- Bearings are one of the most common causes of motor failure.
- A damaged bearing changes the motor's vibration: it adds sharp impacts and new frequency components.
- Machine Learning can learn these patterns and recognise the fault automatically.

*Images: `time_normal.png` next to `time_inner_race_fault.png` (healthy vs faulty waveform)*

---

## Slide 3: Problem Statement
- Manual vibration analysis needs an expert and takes time.
- Unexpected bearing failure causes downtime, repair cost and safety risks.
- **Goal:** a software system that takes a vibration signal and reports:
  - the bearing condition (**Normal, Inner Race Fault, Ball Fault, Outer Race Fault**)
  - the fault size (**0.007" / 0.014" / 0.021"**)
  - a confidence score, and a warning when the result is uncertain

---

## Slide 4: Proposed Solution
- Data: CWRU Bearing Data Center. 40 recordings: drive-end sensor, 12 kHz, 3 fault sizes, motor loads 0–3 HP.
- Data correction: the Normal recordings turned out to be 48 kHz (verified from the shaft-speed peaks), so they were downsampled to 12 kHz.
- 2331 windows of 2048 samples, with 14 statistical and frequency features per window.
- Two Random Forest models: one for fault type, one for fault size.
- Trained with extra noisy copies of the data, so the models still work on noisy signals.
- Streamlit web app: upload a CSV at any sampling rate and get the waveform, FFT, prediction, fault size and status.

---

## Slide 5: System Architecture
```
Vibration CSV / CWRU .mat
   ↓
Convert to 12 kHz (if needed)
   ↓
Windowing (2048 samples)
   ↓
Feature extraction (9 time-domain + 5 frequency-domain)
   ↓
StandardScaler
   ↓
Random Forest #1 → fault type      Random Forest #2 → fault size
   ↓
Safety check (unfamiliar signal / low confidence → "Result uncertain")
   ↓
"Motor healthy" / "Motor needs inspection"
```

*Image: draw the block diagram above in PowerPoint (SmartArt / shapes)*

---

## Slide 6: Algorithm
1. Load the drive-end signal (`*_DE_time`) from each .mat file. Downsample the Normal files from 48 kHz to 12 kHz.
2. Split into 2048-sample windows, each labelled with class, load and fault size.
3. Compute features: mean, RMS, variance, std. deviation, peak, peak-to-peak, crest factor, kurtosis, skewness, FFT dominant frequency, and energy in 4 frequency bands.
4. Standardise the features (mean 0, std 1).
5. Train Random Forests (100 trees, random_state = 42, balanced class weights) on clean windows plus noisy copies (SNR 0–20 dB).
6. Evaluate on random splits, an unseen motor load, noisy signals, an unseen fault size and a different sensor.
7. In the app: features → scaler → model → class with the highest probability, plus the safety check.

*Image: `feature_importance.png` (top features: peak-to-peak 0.160, band 3 energy 0.132, variance 0.120)*

---

## Slide 7: Results

| Test | Test windows | Accuracy |
|---|---|---|
| Fault type, random 80/20 split | 467 | **99.57%** |
| Fault type, train 0–2 HP, test unseen 3 HP | 591 | **98.82%** |
| Fault size, random 80/20 split | 425 | **99.76%** |
| Fault size, unseen 3 HP | 532 | **99.25%** |

Noise test (noise as strong as the signal, 0 dB): **65.1% → 92.5%** with noise-augmented training.

Honest weaknesses:
- Unseen crack size: 32–63%
- Different sensor position (fan-end): 19.6%

The model needs training data from the same fault sizes and sensor position it will be used on.

*Images: `confusion_matrix_load_held_out.png` and `noise_robustness.png` side by side*

---

## Slide 8: Demo
- Run `python3 -m streamlit run app.py` and upload CSVs from `data/samples/`.
- The demo windows were **left out** of training (unseen 3 HP data).

| Sample | Prediction | Status |
|---|---|---|
| sample_normal.csv | Normal (100.0%) | Motor healthy |
| sample_inner_race_fault_0_007in.csv | Inner Race, 0.007" | Motor needs inspection |
| sample_outer_race_fault_0_014in.csv | Outer Race, 0.014" | Motor needs inspection |
| sample_inner_race_fault_0_021in.csv | Inner Race, 0.021" | Motor needs inspection |
| sample_normal_48kHz.csv (rate = 48000) | Normal (100.0%) | Motor healthy |

- Safety check: a 50 Hz sine wave, or a healthy signal made 100× louder, shows **"Result uncertain"** instead of a confident wrong answer.

*Image: a screenshot of the app showing a fault prediction (take this yourself before the presentation)*

---

## Slide 9: Advantages
- Identifies the fault **type** (inner race / ball / outer race) and **size**, not just "faulty".
- Robust to noise: 92.5% accuracy even when the noise is as strong as the signal.
- Accepts data at any sampling rate.
- Warns when it is unsure instead of giving a confident wrong answer.
- Lightweight: Random Forests on 14 features train in about 20 seconds on a laptop, with no GPU.
- Explainable: each feature (RMS, kurtosis, band energy) has a physical meaning.
- Tested honestly, including tests where it fails, so its limits are known.

---

## Slide 10: Limitations & Future Scope
**Limitations**
- Software prototype on recorded lab data from one test rig. No hardware or real-time monitoring yet.
- Does not generalise to unseen crack sizes (32–63%) or a different sensor position (19.6%).
- Only bearing faults are classified.

**Future scope**
- **Hardware:** accelerometer + ESP32 to capture vibration from a real motor for real-time monitoring.
- **IoT / cloud dashboard:** remote monitoring, alerts and trend history.
- Physics-based features (envelope analysis at bearing defect frequencies) to generalise to new sizes and sensors.
- Train on real industrial data and more fault types (misalignment, unbalance).

*Image: `confusion_matrix_other_sensor.png` (optional, shows the sensor-position limitation)*
