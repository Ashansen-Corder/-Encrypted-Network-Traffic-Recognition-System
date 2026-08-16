"""
Stage 4.2 — Cleaning & preparing the information.

Handles missing/infinite values, encodes the TYPE label, splits into
train/validation/test sets, and standardizes features (zero mean, unit variance).
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import config
from .utils import get_logger, timer

logger = get_logger(__name__)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities with NaN, then impute NaNs with the column median."""
    with timer("Cleaning dataframe (inf/NaN handling)", logger):
        df = df.copy()
        feature_cols = [c for c in df.columns if c != config.LABEL_COLUMN]

        n_before = len(df)
        df = df.dropna(subset=[config.LABEL_COLUMN])
        df = df[df[config.LABEL_COLUMN].isin(config.CLASS_NAMES.keys())]
        if len(df) != n_before:
            logger.info(f"Dropped {n_before - len(df)} rows with missing/unknown labels.")

        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        n_missing = df[feature_cols].isna().sum().sum()
        if n_missing > 0:
            logger.info(f"Imputing {n_missing} missing feature values with column medians.")
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

        # Drop constant columns (zero variance) - they add no information.
        variances = df[feature_cols].var()
        constant_cols = variances[variances == 0].index.tolist()
        if constant_cols:
            logger.info(f"Dropping {len(constant_cols)} zero-variance columns: {constant_cols}")
            df = df.drop(columns=constant_cols)

    return df


def encode_labels(y: pd.Series, encoder: LabelEncoder = None, fit: bool = True):
    if fit:
        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y)
    else:
        y_enc = encoder.transform(y)
    return y_enc, encoder


def split_data(df: pd.DataFrame):
    """Stratified train / validation / test split."""
    feature_cols = [c for c in df.columns if c != config.LABEL_COLUMN]
    X = df[feature_cols].values
    y_raw = df[config.LABEL_COLUMN].values

    y_enc, label_encoder = encode_labels(pd.Series(y_raw), fit=True)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_enc, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y_enc
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE, stratify=y_train_full,
    )

    logger.info(
        f"Split sizes -> train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}"
    )
    return (X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, label_encoder)


def scale_features(X_train, X_val, X_test, save: bool = True):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    if save:
        joblib.dump(scaler, config.SCALER_PATH)
        logger.info(f"Saved scaler -> {config.SCALER_PATH}")

    return X_train_s, X_val_s, X_test_s, scaler


def save_label_encoder(label_encoder: LabelEncoder):
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)
    logger.info(f"Saved label encoder -> {config.LABEL_ENCODER_PATH}")


def save_feature_list(feature_cols):
    with open(config.FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"Saved feature column list -> {config.FEATURE_LIST_PATH}")


def preprocess_pipeline(df: pd.DataFrame):
    """Full cleaning -> split -> scale pipeline. Returns everything downstream stages need."""
    df = clean_dataframe(df)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, label_encoder = split_data(df)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    save_label_encoder(label_encoder)
    save_feature_list(feature_cols)

    return {
        "X_train": X_train_s, "X_val": X_val_s, "X_test": X_test_s,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_cols": feature_cols,
        "label_encoder": label_encoder,
        "scaler": scaler,
    }
