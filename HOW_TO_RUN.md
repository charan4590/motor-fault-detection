# How to Run: Motor Fault Detection AI

A step-by-step guide to run this project on your own computer (Windows, macOS or Linux).
No hardware is needed. The trained models and demo files are already included, so you can run the app in about 5 minutes.

---

## What you need

| Software | Version | Download |
|---|---|---|
| **Python** | 3.10 – 3.13 (3.12 recommended) | https://www.python.org/downloads/ |
| **Git** (optional) | any | https://git-scm.com/downloads |

> **Windows users:** when installing Python, tick **"Add python.exe to PATH"** on the first screen.

Check Python is installed by opening a terminal (Windows: **Command Prompt**; Mac: **Terminal**):

```bash
python --version
```
On Mac/Linux, use `python3 --version` instead. You should see `Python 3.10` or newer.

---

## Step 1: Get the project

**Option A: with Git**
```bash
git clone https://github.com/charan4590/motor-fault-detection.git
cd motor-fault-detection
```

**Option B: without Git**
1. Open https://github.com/charan4590/motor-fault-detection
2. Click the green **Code** button, then **Download ZIP**
3. Unzip it, then open a terminal **inside** the unzipped `motor-fault-detection` folder

---

## Step 2: Create a virtual environment (one time)

This keeps the project's packages separate from the rest of your computer.

**Windows**
```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` at the start of your terminal line.
> Every time you open a new terminal to use the project, run the **activate** line again.

---

## Step 3: Install the packages (one time)

```bash
pip install -r requirements.txt
```
This takes 1–3 minutes.

---

## Step 4: Start the app

```bash
python -m streamlit run app.py
```

Your browser opens **http://localhost:8501** automatically. If it doesn't, open that address yourself.
> If Streamlit asks for an email address the first time, just press **Enter** to skip.

---

## Step 5: Try a prediction

1. Leave **Sampling rate** at **12000**.
2. Click **Browse files** and open the project's **`data/samples/`** folder.
3. Upload any file. The file name tells you the correct answer, so you can check the AI:

| File | Expected result |
|---|---|
| `sample_normal.csv` | Normal → ✅ **Motor healthy** |
| `sample_inner_race_fault_0_007in.csv` | Inner Race Fault, 0.007 inch → ⚠️ **Motor needs inspection** |
| `sample_ball_fault_0_007in.csv` | Ball Fault, 0.007 inch → ⚠️ **Motor needs inspection** |
| `sample_outer_race_fault_0_007in.csv` | Outer Race Fault, 0.007 inch → ⚠️ **Motor needs inspection** |
| `sample_outer_race_fault_0_014in.csv` | Outer Race Fault, 0.014 inch → ⚠️ **Motor needs inspection** |
| `sample_inner_race_fault_0_021in.csv` | Inner Race Fault, 0.021 inch → ⚠️ **Motor needs inspection** |
| `sample_normal_48kHz.csv` | First change **Sampling rate** to **48000**, then upload → ✅ **Motor healthy** |

The app shows the vibration waveform, the frequency spectrum (FFT), RMS, peak, dominant frequency, the predicted fault with its confidence, and the estimated fault size.

**Using your own data:** upload a CSV with **one column** of vibration values (acceleration in g, from a drive-end accelerometer), at least 2048 values, and enter the correct sampling rate.

---

## Step 6: Stop the app

Go back to the terminal and press **Ctrl + C**.

---

## Optional: retrain the model from the raw data

Only needed if you want to reproduce the training. The raw dataset (40 files, ~136 MB) is not stored on GitHub, so download it first.

**1. Download the CWRU data into `data/raw/`**

macOS / Linux:
```bash
cd data/raw
for n in 97 98 99 100 105 106 107 108 118 119 120 121 130 131 132 133 169 170 171 172 185 186 187 188 197 198 199 200 209 210 211 212 222 223 224 225 234 235 236 237; do curl -L -o $n.mat https://engineering.case.edu/sites/default/files/$n.mat; done
cd ../..
```

Windows (PowerShell):
```powershell
cd data\raw
foreach ($n in 97,98,99,100,105,106,107,108,118,119,120,121,130,131,132,133,169,170,171,172,185,186,187,188,197,198,199,200,209,210,211,212,222,223,224,225,234,235,236,237) { Invoke-WebRequest "https://engineering.case.edu/sites/default/files/$n.mat" -OutFile "$n.mat" }
cd ..\..
```

**2. Run the pipeline in order** (with the virtual environment active):
```bash
python src/prepare_data.py
python src/features.py
python src/make_samples.py
python src/train.py
```
`train.py` takes about 20 seconds. It prints all accuracies and saves the models to `models/` and the charts and `metrics.json` to `results/`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` / `python3` not found | Install Python (see above). On Windows, reinstall with **"Add python.exe to PATH"** ticked. |
| `streamlit: command not found` | Use `python -m streamlit run app.py` (not `streamlit run app.py`). |
| `No module named ...` | Activate the virtual environment (Step 2), then run `pip install -r requirements.txt` again. |
| `No such file or directory: app.py` | Your terminal isn't in the project folder. Use `cd` to go into `motor-fault-detection`. |
| `Port 8501 is already in use` | The app is already running in another terminal. Close it with Ctrl + C, or open http://localhost:8501. |
| Error installing `scikit-learn==1.6.1` | Your Python is too new (3.14+). Install Python 3.12 and repeat from Step 2. |
| Windows: "running scripts is disabled" when activating | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell once, or use **Command Prompt** instead. |

---

For details on the method and the full results, see [README.md](README.md).
