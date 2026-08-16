"""
Central configuration for the Encrypted Network Traffic Recognition System.
All paths, constants, and hyperparameters live here so every module stays in sync.
"""
import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

RAW_CSV_PATH = os.path.join(DATA_DIR, "HTTPS-clf-dataset.csv")
DATASET_URL = (
    "https://github.com/Ashansen-Corder/HTTPS-clf-dataset.csv/"
    "releases/download/v1.0/HTTPS-clf-dataset.csv"
)

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
PCA_PATH = os.path.join(MODELS_DIR, "pca.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_columns.json")

RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost.json")
DNN_MODEL_PATH = os.path.join(MODELS_DIR, "dnn_model.keras")

SPLIT_CACHE_PATH = os.path.join(DATA_DIR, "processed_splits.npz")

for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Target / label
# --------------------------------------------------------------------------- #
LABEL_COLUMN = "TYPE"
CLASS_NAMES = {
    "L": "Live Video Streaming",
    "P": "Buffered Video Player",
    "M": "Music/Audio Player",
    "U": "File Upload",
    "D": "File Download",
    "W": "General Transactional Traffic",
}

# --------------------------------------------------------------------------- #
# Columns to drop.
# The raw CSV stores some fields as stringified python arrays (e.g. per-packet
# length sequences). These are not scalar features and are dropped in favour of
# the pre-engineered scalar statistics (mean/max/std/percentiles) and the
# pre-computed PCA embeddings that summarise them, matching Section 3.2 of the
# project proposal.
# --------------------------------------------------------------------------- #
NON_FEATURE_COLUMNS = [
    "Unnamed: 0",
    "DBI_BRST_BYTES",
    "DBI_BRST_PACKETS",
    "PKT_LENGTHS",
    "PPI_PKT_DIRECTIONS",
    "PKT_TIMES",
    "DBI_BRST_TIME_START",
    "DBI_BRST_TIME_STOP",
    "DBI_BRST_DURATION",
    "DBI_BRST_INTERVALS",
    "TIME_INTERVALS",
]

# --------------------------------------------------------------------------- #
# Preprocessing / split
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10          # fraction of the remaining training data used for validation (DNN only)
CV_FOLDS = 5              # stratified K-fold cross-validation folds
PCA_VARIANCE_TARGET = 0.95  # retain components explaining >=95% variance

# --------------------------------------------------------------------------- #
# Random Forest hyperparameters
# --------------------------------------------------------------------------- #
RF_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
RF_DEFAULT_PARAMS = dict(
    n_estimators=400,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)

# --------------------------------------------------------------------------- #
# XGBoost hyperparameters
# --------------------------------------------------------------------------- #
XGB_PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [6, 10],
    "learning_rate": [0.05, 0.1],
}
XGB_DEFAULT_PARAMS = dict(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# --------------------------------------------------------------------------- #
# Deep Neural Network hyperparameters
# --------------------------------------------------------------------------- #
DNN_PARAMS = dict(
    hidden_layers=[256, 128, 64],
    dropout_rate=0.3,
    l2_reg=1e-4,
    learning_rate=1e-3,
    batch_size=256,
    epochs=60,
    patience=8,  # early stopping patience
)

# --------------------------------------------------------------------------- #
# Search mode: "quick" uses default params (fast), "grid" runs GridSearchCV
# --------------------------------------------------------------------------- #
SEARCH_MODE = os.environ.get("NIA_SEARCH_MODE", "quick")
