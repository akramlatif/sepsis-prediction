# Early Sepsis Onset Prediction

> Predicting sepsis onset **6 hours before** physician diagnosis using ICU time-series data from the [PhysioNet Sepsis Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/).

---

## Overview

This project implements a complete machine-learning pipeline that addresses the key challenges of clinical sepsis prediction:

| Challenge | Solution |
|-----------|----------|
| **Irregular time-series** | Forward-fill imputation + missingness indicators |
| **High missing-value rates** | Median imputation (fit on train only) + binary flags |
| **Label noise** | cleanlab-based noise detection and removal |
| **Temporal data leakage** | Patient-level splits, backward-only rolling features |
| **Class imbalance** | `scale_pos_weight`, AUPRC evaluation, threshold analysis |
| **Poor calibration** | Isotonic regression calibration |

Two models are trained and compared:
1. **XGBoost** — gradient-boosted tree baseline  
2. **LightGBM** — advanced gradient-boosting model

---

## Project Structure

```
sepsis_prediction/
├── data/                        # Place .psv files here
├── src/
│   ├── load_data.py             # Load all .psv files into one DataFrame
│   ├── preprocessing.py         # Imputation, missingness flags, normalization
│   ├── label_engineering.py     # 6-hour-ahead labels, patient-level split
│   ├── label_noise.py           # Noise detection via cleanlab
│   ├── features.py              # Rolling statistics, temporal features
│   ├── model.py                 # XGBoost + LightGBM training
│   ├── calibration.py           # Isotonic regression calibration
│   └── evaluate.py              # Metrics, plots, threshold analysis
├── notebooks/
│   └── full_pipeline.ipynb      # Interactive walkthrough
├── outputs/                     # Generated plots and results
│   ├── confusion_matrix_*.png
│   ├── calibration_curve.png
│   └── results.txt
├── requirements.txt
├── README.md
└── main.py                      # Run the entire pipeline
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download the PhysioNet 2019 Sepsis Challenge data from:  
https://physionet.org/content/challenge-2019/1.0.0/

You need to register for a PhysioNet account and sign the data use agreement.

### 3. Place data files

Copy all `.psv` patient files (e.g., `p000001.psv`, `p000002.psv`, ...) into the `data/` directory:

```
data/
├── p000001.psv
├── p000002.psv
├── p000003.psv
└── ...
```

### 4. Run the pipeline

```bash
python main.py
```

The pipeline will:
1. Load all patient files
2. Create 6-hour-ahead prediction labels
3. Perform a patient-level 80/20 train/test split
4. Engineer rolling temporal features
5. Preprocess (impute, normalize) — fit on train only
6. Detect and remove noisy labels using cleanlab
7. Train XGBoost and LightGBM models with early stopping
8. Calibrate probability outputs
9. Evaluate and save results

---

## Outputs

After running `main.py`, the `outputs/` directory will contain:

| File | Description |
|------|-------------|
| `confusion_matrix_xgboost.png` | Confusion matrix for XGBoost |
| `confusion_matrix_lightgbm.png` | Confusion matrix for LightGBM |
| `calibration_curve.png` | Calibration curves for both models |
| `results.txt` | Summary of AUROC, AUPRC, and runtime |
| `xgboost_model.joblib` | Saved XGBoost model |
| `lightgbm_model.joblib` | Saved LightGBM model |

---

## Key Design Decisions

### No Data Leakage
- **Patient-level split**: No patient appears in both the train and test sets.
- **Scaler fit on train only**: `StandardScaler` is fit exclusively on training data.
- **Backward-only rolling features**: `.shift(1)` is applied before `.rolling()` to exclude the current hour.
- **Post-diagnosis rows dropped**: All data at and after the first SepsisLabel=1 hour is discarded.

### Label Engineering
- The prediction target (`label_6h`) is set to 1 for the 6 hours *before* the first recorded sepsis onset.
- This creates a clinically meaningful early-warning window.

### Label Noise Handling
- cleanlab identifies likely mislabelled training samples using confident learning.
- Only training labels are inspected and cleaned — test labels are untouched.

### Reproducibility
- `random_state=42` is set on every model, split, and randomised component.

---

## Interactive Notebook

For a step-by-step walkthrough with inline plots, open:

```bash
jupyter notebook notebooks/full_pipeline.ipynb
```

---

## License

This project is for educational and research purposes.  
The PhysioNet data is subject to its own [data use agreement](https://physionet.org/content/challenge-2019/1.0.0/).
