# High-Performance Encrypted Network Traffic Recognition System

**IT41033 – Nature Inspired Algorithms | Mini Project | Intake 12/13 | Horizon Campus 2026**

A machine-learning pipeline that classifies encrypted (HTTPS/TLS/QUIC) network flows into
six behavioral application profiles **without decrypting payloads** — using only
statistical/behavioral flow features (packet length distributions, inter-arrival timings,
burst analytics and PCA embeddings).

## Target Classes

| Code | Class                        |
|------|-------------------------------|
| L    | Live Video Streaming          |
| P    | Buffered Video Player         |
| M    | Music / Audio Streaming       |
| U    | File Upload                   |
| D    | File Download                 |
| W    | General Transactional (Web)   |

## Project Structure

```
project/
├── data/                       # raw / processed data (downloaded, not committed)
├── models/                     # saved trained models + scaler + PCA + encoders
├── results/                    # metrics, confusion matrices, plots, reports
├── notebooks/
│   └── exploration.ipynb       # optional EDA notebook
├── src/
│   ├── __init__.py
│   ├── config.py                # all paths & hyperparameters
│   ├── data_loader.py           # downloads & loads the dataset
│   ├── preprocessing.py         # cleaning, splitting, scaling
│   ├── dimensionality_reduction.py  # PCA step
│   ├── models.py                # RandomForest / XGBoost / DNN builders
│   ├── train.py                 # trains all 3 models w/ cross-validation
│   ├── evaluate.py              # metrics, confusion matrices, comparison report
│   └── utils.py                 # logging / helper functions
├── main.py                      # end-to-end pipeline entry point
├── requirements.txt
└── README.md
```

## Pipeline (matches the project proposal, Section 4)

1. **Data Ingestion** (`data_loader.py`) – downloads `HTTPS-clf-dataset.csv` (145,671 flows)
   from the GitHub release and loads it.
2. **Cleaning** (`preprocessing.py`) – drops raw array/string columns, fixes types, handles
   missing/infinite values, encodes labels.
3. **Dimensionality Reduction** (`dimensionality_reduction.py`) – standardizes features and
   applies PCA to reduce ~76 raw features while retaining ≥95% variance.
4. **Model Training** (`train.py`, `models.py`) – trains and tunes:
   - Random Forest Classifier
   - XGBoost Classifier
   - Deep Neural Network (Keras/TensorFlow, multi-layer with dropout & batch-norm)
   Each is evaluated with stratified K-Fold cross-validation.
5. **Evaluation** (`evaluate.py`) – accuracy, precision/recall/F1 (macro & weighted),
   confusion matrices, ROC-AUC (OvR), and a side-by-side model comparison report.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline (download → clean → PCA → train all 3 models → evaluate):

```bash
python main.py --all
```

Or run individual stages:

```bash
python main.py --download          # only download the dataset
python main.py --preprocess        # clean + split + scale + PCA
python main.py --train rf          # train Random Forest only
python main.py --train xgb         # train XGBoost only
python main.py --train dnn         # train Deep Neural Network only
python main.py --evaluate          # evaluate all trained models & produce comparison report
```

Useful flags:
```bash
python main.py --all --sample 20000   # run on a 20,000-row subsample for a quick smoke test
python main.py --all --no-download    # skip download if dataset.csv already exists in data/
```

## Outputs

After a full run, `results/` contains:
- `model_comparison.csv` – accuracy / precision / recall / F1 for all 3 models
- `confusion_matrix_<model>.png` – confusion matrix heatmaps
- `pca_variance.png` – explained variance plot
- `classification_report_<model>.txt` – full sklearn classification reports
- `dnn_training_history.png` – loss/accuracy curves for the neural network

`models/` contains the persisted `scaler.pkl`, `pca.pkl`, `label_encoder.pkl`,
`random_forest.pkl`, `xgboost.json`, and `dnn_model.keras`.

## Dataset

Source: https://github.com/Ashansen-Corder/HTTPS-clf-dataset.csv/releases/download/v1.0/HTTPS-clf-dataset.csv

145,671 labelled encrypted network flows with 76 behavioral/statistical features
(volumetric, packet-length distribution, inter-arrival timing, burst analytics, and
pre-computed PCA embeddings) across the six traffic classes above.

## Group Members

- ITBIN-2313-0104 — Ashan Senanayaka
- ITBIN-2313-0131 — J.W.D.A. Williyamge
- ITBIN-2313-0115 — Kavindu Thathsara
- ITBIN-2313-0098 — M.A.S. Sandamali
- ITBIN-2313-0085 — Nethmi Priyanjala
---

## Note on Submission Files & Dataset

Due to the **5MB file size limit on LMS**, large binary files, dataset CSVs, and local virtual environment dependencies have been excluded from this submission ZIP package:

- **`data/HTTPS-clf-dataset.csv`** (Raw Dataset - ~184 MB)
- **`data/processed_splits.npz`** (Preprocessed NPZ Splits - ~5.4 MB)
- **`.venv/`** (Local Python Virtual Environment)

### Access Complete Dataset & Code
The full source code, raw dataset files, preprocessed splits, and complete commit history are accessible on our official GitHub Repository:
👉 **GitHub Repository:** [https://github.com/Ashansen-Corder/-Encrypted-Network-Traffic-Recognition-System.git](https://github.com/Ashansen-Corder/-Encrypted-Network-Traffic-Recognition-System.git)