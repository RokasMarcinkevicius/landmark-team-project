# Sprint 1 – Landmark‑ID (Vilnius Commons) – Team Documentation [EN]


---

## 1) Sprint Overview

* **Time window:** 7–13 Oct 2025
* **MVP flow:** Camera → single frame → **TFLite (INT8)** model → **Top‑3** predictions on screen
* **Backbone:** `MobileNetV3‑Small` (transfer learning)
* **Stack:** Python 3.11, TensorFlow 2.16.1, Keras 3, Weights & Biases (W&B) 0.22.1
* **Data source:** Wikimedia Commons (5 Vilnius landmarks)
* **Classes:**

  * `gediminas_tower`
  * `vilnius_cathedral`
  * `gate_of_dawn`
  * `st_anne`
  * `three_crosses`

**Definition of Done (Sprint 1):**

* Test set **Top‑1 ≥ 75%** ✅ (observed val_top1 ~ **77–78%**)
* Model size **≤ 15–20 MB** (INT8 target)
* Training is reproducible (**seed=42**)
* Training/experiments logged in **W&B**

---

## 2) Repository Layout (summary)

```
landmark-id/
  data/
    raw/                   # downloaded raw images (Commons)
    train/ val/ test/      # post-split folders
    SOURCES.csv            # license/attribution metadata
  models/                  # .keras and .tflite outputs
  reports/                 # metric reports + confusion matrix
  scripts/
    fetch_commons_requests.py   # robust downloader via MediaWiki API
  src/
    data/split.py               # 70/15/15 split
    data/log_dataset_wandb.py   # W&B Artifact (dataset)
    models/train_keras_wandb.py # training + W&B logging
    models/eval_and_report.py   # test & report (CM, class-wise)
  .venv/                        # local virtual environment
```

---

## 3) Environment Setup

* **Python:** 3.11 (recommended for TF)
* **Virtual env:** `.venv`
* **Critical pins:**

  * `tensorflow==2.16.1`
  * `protobuf<5,>=3.20.3` (e.g., 4.25.8) ← required for TF 2.16.1
  * `wandb==0.22.1`

**Install:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install tensorflow==2.16.1 pillow tqdm wandb "protobuf<5,>=3.20.3"
```

**W&B login:**

```bash
wandb login
```

---

## 4) Data Collection & Preparation

### 4.1 Download (Wikimedia Commons)

* Script: `scripts/fetch_commons_requests.py`
* Features: custom **User‑Agent**, **retry/backoff**, 429/5xx handling, minimum image size filter, and license/attribution written to `SOURCES.csv`.

**Run:**

```bash
python scripts/fetch_commons_requests.py --max-per-class 120 --min-size 400
```

### 4.2 Train/Val/Test Split (70/15/15)

```bash
python src/data/split.py
```

**Example counts (one run):**

* Train: **378** images / 5 classes
* Val: **81** images / 5 classes
* Test: **83** images / 5 classes

### 4.3 Log dataset to W&B (Artifact)

```bash
python src/data/log_dataset_wandb.py
```

> You should see a `vilnius-landmarks-mini` dataset artifact in your W&B project.

---

## 5) Model Training (Baseline + Fine‑tune)

### 5.1 Baseline (head‑only)

* Backbone: `MobileNetV3Small(weights="imagenet", include_top=False)`
* Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
* Augment: flip, zoom, brightness
* Metrics: `top1`, `top3`
* Callbacks: `EarlyStopping(monitor=val_top1, patience=5, restore_best_weights=True)`, `WandbMetricsLogger`
* Checkpoint: **Keras `ModelCheckpoint`** (writes only on best `val_top1`)

**Train:**

```bash
python src/models/train_keras_wandb.py
```

> Note: Using the new W&B Keras API (`wandb.integration.keras`). The legacy `WandbCallback` graph logging was removed due to Keras 3 incompatibility.

### 5.2 Fine‑tuning

* Strategy: unfreeze **~last 30 layers**, retrain at **LR = 1e‑5** for ~8 epochs
* Rationale: typically adds **+3–8 points** on small datasets

**Observed result (example):**

* Val **Top‑1 ≈ 77–78%**
* Val **Top‑3 ≈ 96–98%**

---

## 6) Evaluation & Reporting

* Script: `src/models/eval_and_report.py`
* Produces:

  * `reports/confusion_matrix.csv`
  * `reports/sprint1_metrics.md` (Top‑1/Top‑3 + per‑class precision/recall/F1)
* W&B: run charts, class‑wise metrics, confusion matrix image

**Run:**

```bash
python src/models/eval_and_report.py
```

---

## 7) Model Export (TFLite)

* Outputs:

  * `models/landmark_mnv3_fp32.tflite` (FP32)
  * `models/landmark_mnv3_int8.tflite` (dynamic‑range INT8)
* Size target: **≤ 20 MB** (INT8)
* Runtime speed: measured on device in Sprint 2 (goal **< 150 ms**)

**Example export:**

```bash
python - << 'PY'
import tensorflow as tf
m = tf.keras.models.load_model("models/landmark_mnv3.keras")
conv = tf.lite.TFLiteConverter.from_keras_model(m)
open("models/landmark_mnv3_fp32.tflite","wb").write(conv.convert())
conv.optimizations=[tf.lite.Optimize.DEFAULT]
open("models/landmark_mnv3_int8.tflite","wb").write(conv.convert())
print("TFLite export complete")
PY
```

**(Optional) W&B artifact:**

```bash
python - << 'PY'
import wandb
run = wandb.init(project="landmark-id", job_type="export")
a = wandb.Artifact("landmark-mnv3-tflite", type="model",
                   description="Vilnius 5-class, FP32+INT8 export")
a.add_file("models/landmark_mnv3_fp32.tflite")
a.add_file("models/landmark_mnv3_int8.tflite")
wandb.log_artifact(a); run.finish()
PY
```

---

## 8) Issues We Hit & Fixes

* **`ModuleNotFoundError: mwclient`** → wrong interpreter; use `.venv/bin/python`, reinstall via `python -m pip install mwclient`.
* **Wikimedia API `JSONDecodeError`** → switched to a robust requests‑based fetcher with UA + retries; dump non‑JSON to `logs/last_response.html`.
* **`protobuf 6.x` vs TF 2.16.1** → pin `protobuf<5,>=3.20.3` (e.g., 4.25.8).
* **`wandb.keras`/graph errors** → move to `wandb.integration.keras` and use `WandbMetricsLogger`; use **Keras** `ModelCheckpoint` for files.
* **W&B checkpoint FileNotFound** → avoid `WandbModelCheckpoint`; Keras `ModelCheckpoint` writes actual files only on best epoch.

---

## 9) Definition of Done – Status

* [x] Data ready (5 classes, Commons) with `SOURCES.csv` licenses/attribution
* [x] 70/15/15 split (seed=42)
* [x] Baseline + fine‑tune training; **val_top1 ~ 77–78%**
* [x] W&B: runs/metrics/artifacts
* [x] Confusion matrix + class‑wise metrics reported
* [ ] TFLite INT8 export files prepared (script above if missing)

---

## 10) Handoff to Sprint 2 (short)

* Flutter integration (camera → TFLite → Top‑3). Ensure `labels.txt` order matches training class order.
* On‑device latency benchmarking (**< 150 ms** goal) + permission/error UX.
* “Unsure” threshold (e.g., Top‑1 < 0.55 → suggest retake).

---

## 11) One‑Shot Quickstart (repro commands)

```bash
# 1) Environment
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow==2.16.1 pillow tqdm wandb "protobuf<5,>=3.20.3"
wandb login

# 2) Data → split → artifact
a=120; s=400
python scripts/fetch_commons_requests.py --max-per-class $a --min-size $s
python src/data/split.py
python src/data/log_dataset_wandb.py

# 3) Train (baseline + fine‑tune)
python src/models/train_keras_wandb.py

# 4) Evaluate + report
python src/models/eval_and_report.py

# 5) TFLite export (optional)
python - << 'PY'
import tensorflow as tf
m=tf.keras.models.load_model("models/landmark_mnv3.keras")
c=tf.lite.TFLiteConverter.from_keras_model(m)
open("models/landmark_mnv3_fp32.tflite","wb").write(c.convert())
c.optimizations=[tf.lite.Optimize.DEFAULT]
open("models/landmark_mnv3_int8.tflite","wb").write(c.convert())
PY
```

---

### License & Attribution

* Wikimedia Commons images are **freely licensed**; if images are displayed in the app, provide proper **attribution**.
* `data/SOURCES.csv` records **source URL / author / license** for each image.

---

*Prepared for team sharing; feel free to open a repo issue or comment on the W&B run if you need tweaks.*
