# Early Prediction of Obstructive Sleep Apnea Using Multi-Signal Deep Learning with Attention Mechanisms

**B.Tech Final Year Project Report**  
**Author:** Ippili Yaswanth Kumar (B122053)  
**Supervisor:** Dr. Puspanjali Mohapatra  
**Institution:** International Institute of Information Technology, Bhubaneswar  
**Year:** May 2026

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Abstract

Obstructive Sleep Apnea (OSA) is a serious respiratory condition that causes breathing to stop and start repeatedly during sleep. Current diagnostic and treatment methods are **reactive** — they only respond *after* an apnea event has already occurred, exposing patients to intermittent hypoxia and cardiovascular stress.

This project proposes a **proactive forecasting system** that predicts an impending apnea event **30 seconds in advance** using raw nasal airflow and blood oxygen saturation (SpO₂) signals from the Sleep Heart Health Study (SHHS) dataset.

We engineered and rigorously compared two deep learning architectures:
- A foundational **1D-CNN-BiLSTM** model
- An advanced **1D-CNN-BiLSTM with Multi-Head Attention** mechanism

The attention-enhanced model achieved state-of-the-art performance:
- **Overall Accuracy:** 91.04%
- **Sensitivity (Recall):** **99.58%** (critical for patient safety)
- **Specificity:** 82.51%

This enables truly preventive interventions such as preemptive CPAP pressure ramp-up or gentle wearable haptic alerts — potentially eliminating apnea events before they cause physiological harm.

**Keywords:** Obstructive Sleep Apnea, Early Prediction, CNN-BiLSTM, Multi-Head Attention, Airflow, SpO₂, Proactive Healthcare.

---

## 🚀 Key Highlights & Results

| Metric                  | Base Model (1D-CNN-BiLSTM) | **Attention-Enhanced Model** |
|-------------------------|----------------------------|------------------------------|
| **Accuracy**            | 89.36%                     | **91.04%**                   |
| **Sensitivity**         | 93.28%                     | **99.58%**                   |
| **Specificity**         | 85.44%                     | 82.51%                       |
| **Parameters**          | ~129k                      | ~166k                        |
| **Prediction Horizon**  | 30 seconds                 | 30 seconds                   |
| **Input Window**        | 90 seconds @ 2 Hz          | 90 seconds @ 2 Hz            |

### Why This Matters
- **99.58% sensitivity** means the model almost **never misses** an impending apnea — the most important metric for any medical early-warning system.
- The Multi-Head Attention mechanism functions as a **"temporal spotlight"**, automatically focusing on the subtle pre-apneic breathing pattern changes that precede airway collapse.
- Models are **lightweight** and optimized for edge deployment (wearables, smart CPAP machines).

---

## 🧠 Methodology

### Data Pipeline (Reproducible & Leakage-Free)
1. **Dataset**: Sleep Heart Health Study (SHHS) — 40 diverse patients (EDF + XML annotations).
2. **Signal Selection**: Nasal Airflow + SpO₂ only (minimal hardware requirement).
3. **Preprocessing**:
   - Aggressive downsampling to **2 Hz** (removes noise while preserving clinically relevant patterns).
   - **Split-Channel Normalization** (key innovation):
     - Airflow → Local Z-score normalization
     - SpO₂ → Absolute scaling (preserves true oxygen desaturation magnitude)
   - Strict 90-second historical window → 30-second future labeling (ratio ≥ 30% apnea = positive class).
   - **Patient-wise GroupShuffleSplit** (zero data leakage between train/test).

### Neural Network Architectures

**Base Model** (`First_Implementation.py`)
```
Input (180 timesteps × 2 channels)
→ Conv1D(64,5) + BN + ReLU + MaxPool + Dropout(0.2)
→ Conv1D(128,3) + BN + ReLU + MaxPool + Dropout(0.3)
→ BiLSTM(64, return_sequences=True) + Dropout(0.3)
→ GlobalMaxPooling1D
→ Dense(32, ReLU) + L2 reg
→ Dense(1, Sigmoid)
```

**Attention-Enhanced Model** (`Second_Implementation.py`) ← **Recommended**
```
... (same CNN + BiLSTM blocks)
→ MultiHeadAttention(num_heads=2, key_dim=32)
→ GlobalMaxPooling1D
→ Dense(64, ReLU) + Dropout(0.3) + L2 reg
→ Dense(1, Sigmoid)
```

**Training Optimizations**:
- Mixed-precision (`float16`) on NVIDIA T4 GPU
- Adam optimizer + EarlyStopping (monitor `val_auc`) + ReduceLROnPlateau
- Dynamic threshold selection via **Youden's J statistic** on ROC curve

---

## 📁 Recommended Project Structure

```bash
osa-early-prediction/
├── README.md
├── First_Implementation.py          # Base CNN-BiLSTM (baseline)
├── Second_Implementation.py         # Attention model (best results)
├── requirements.txt
├── report/
│   ├── main.tex                     # Full LaTeX source
│   └── figures/                     # Training curves, confusion matrices, architecture diagrams
├── data/                            # (gitignored) Place SHHS EDF + XML files here
└── LICENSE
```

---

## 🛠️ Getting Started

### 1. Prerequisites
```bash
pip install tensorflow mne numpy scikit-learn matplotlib seaborn
```

Or use the provided `requirements.txt` (create it with the above packages).

### 2. Dataset Access
- Apply for access to the **Sleep Heart Health Study (SHHS)** at: https://sleepdata.org/datasets/shhs
- Download a subset of EDF files + corresponding XML annotation files.
- Recommended: Start with 30–40 patients for quick experimentation.

### 3. Run in Google Colab (Recommended — Zero Setup)

1. Upload both `.py` files to a new Colab notebook.
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Organize data as:
   ```
   /content/drive/MyDrive/SHHS_Dataset/
   ├── edfs/          # *.edf files
   └── annotations/   # matching *-nsrr.xml or *-profusion.xml files
   ```
4. Run `Second_Implementation.py` — it will automatically:
   - Extract & preprocess data
   - Train the attention model
   - Generate training curves + confusion matrix
   - Print final metrics with optimal threshold

**Expected runtime on T4 GPU**: ~25–40 minutes for 40 patients.

### 4. Local Execution
Update the paths at the top of the scripts:
```python
EDF_DIR = "/path/to/your/edfs/"
ANNOT_DIR = "/path/to/your/annotations/"
```

Then simply:
```bash
python Second_Implementation.py
```

---

## 📊 Generated Outputs

Both scripts automatically produce:
- Training/Validation Loss & AUC curves
- Confusion Matrix with optimal ROC threshold
- Full classification report (Precision, Recall, F1 per class)

Example output from Attention model:
```
Optimal ROC Threshold: 0.473
Accuracy:    91.04%
Sensitivity: 99.58%
Specificity: 82.51%
```

---

## 🌟 Real-World Impact & Applications

This work shifts OSA management from **reactive** to **proactive**:

| Application                  | How It Works                                                                 |
|-----------------------------|----------------------------------------------------------------------------------|
| **Smart APAP Machines**     | Preemptively ramp up pressure 30s before collapse — prevents hypoxia entirely   |
| **Wearable Alerts**         | Smart rings/watches deliver gentle haptic vibration to prompt position change   |
| **At-Home Screening**       | Low-cost dual-sensor device + mobile app for rapid triage (reduces PSG backlog) |
| **Telemedicine**            | Overnight prediction report for physicians with severity scoring                |

---

## 🔮 Future Work

- **Edge Deployment**: INT8 quantization + TensorFlow Lite for ESP32 / Raspberry Pi / smart rings.
- **Longer Horizon**: Extend prediction to 60–120 seconds.
- **Multiclass**: Add thoracic effort channel to distinguish Obstructive vs Central Apnea.
- **Personalization**: Federated Learning for patient-specific fine-tuning while preserving privacy.
- **Explainability**: Integrate SHAP / Grad-CAM for regulatory (FDA/CE) approval.

---

## 📚 Key References

- Vaswani et al. (2017). *Attention is All You Need*. NeurIPS.
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory*. Neural Computation.
- Quan et al. (1997). *The Sleep Heart Health Study: design, rationale, and methods*. Sleep.
- Full bibliography available in the LaTeX report (`report/main.tex`).

---

## 📜 License & Disclaimer

This project is released under the **MIT License**.

**Important Disclaimer**: This is a **research prototype** developed for academic purposes. It has **not** undergone clinical validation or regulatory approval (FDA 510(k) / CE marking). Do **not** use for actual patient diagnosis or treatment without proper medical oversight and regulatory clearance.

---

## 👨‍🎓 Contact & Citation

**Author**: Ippili Yaswanth Kumar  
**Email**: b122053@iiit-bh.ac.in
**Institution**: IIIT Bhubaneswar, Department of Computer Science & Engineering(CSE)

If you use this work in your research, please cite the accompanying B.Tech project report:

> Ippili Yaswanth Kumar (2026). *Early Prediction of Obstructive Sleep Apnea Using Multi-Signal Deep Learning with Attention Mechanisms*. B.Tech Final Project Report, International Institute of Information Technology, Bhubaneswar.

---

**Made with ❤️ for better sleep and preventive healthcare.**

*Last updated: May 2026*
