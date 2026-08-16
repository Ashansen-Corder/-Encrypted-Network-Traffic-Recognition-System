"""
Stage 4.1 — Data Ingestion.

Downloads the HTTPS-clf-dataset.csv release (if not already present) and loads
it into a pandas DataFrame, keeping only genuine scalar feature columns plus
the TYPE label.
"""
import os
import urllib.request

import pandas as pd

from . import config
from .utils import get_logger, timer

logger = get_logger(__name__)


def download_dataset(url: str = config.DATASET_URL, dest: str = config.RAW_CSV_PATH,
                      force: bool = False) -> str:
    """Download the dataset CSV from the GitHub release if it isn't cached locally."""
    if os.path.exists(dest) and not force:
        logger.info(f"Dataset already present at {dest}, skipping download.")
        return dest

    with timer(f"Downloading dataset from {url}", logger):
        def _progress(block_num, block_size, total_size):
            if total_size <= 0:
                return
            downloaded = block_num * block_size
            pct = min(100, downloaded * 100 / total_size)
            print(f"\r  {pct:5.1f}%  ({downloaded/1e6:8.1f} MB / {total_size/1e6:8.1f} MB)",
                  end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
    return dest


def load_raw_dataframe(csv_path: str = config.RAW_CSV_PATH, nrows: int = None) -> pd.DataFrame:
    """
    Load the raw CSV.

    `nrows` can be used for quick smoke tests on a subsample. Note that the
    dataset is grouped by class, so a plain `nrows` head-read would only
    return one class — instead we load the full file and take a stratified
    random sample of `nrows` rows across all classes.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python main.py --download` first, "
            f"or pass --no-download only if the file already exists."
        )

    with timer(f"Loading raw CSV ({csv_path})", logger):
        df = pd.read_csv(csv_path)

    if nrows is not None and nrows < len(df):
        frac = nrows / len(df)
        sampled_parts = [
            group.sample(frac=frac, random_state=config.RANDOM_STATE)
            for _, group in df.groupby(config.LABEL_COLUMN)
        ]
        df = pd.concat(sampled_parts).sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
        logger.info(f"Stratified subsample requested: {nrows} -> got {len(df)} rows across all classes.")

    logger.info(f"Loaded raw dataframe with shape {df.shape}")
    return df


def select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop non-feature / raw array-string columns (see config.NON_FEATURE_COLUMNS)
    and keep only genuine scalar numeric features plus the label column.
    """
    drop_cols = [c for c in config.NON_FEATURE_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)

    if config.LABEL_COLUMN not in df.columns:
        raise KeyError(f"Expected label column '{config.LABEL_COLUMN}' not found in dataset.")

    feature_cols = [c for c in df.columns if c != config.LABEL_COLUMN]
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        logger.warning(f"Dropping unexpected non-numeric feature columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    logger.info(f"Kept {df.shape[1] - 1} numeric feature columns + label column.")
    return df


def load_dataset(csv_path: str = config.RAW_CSV_PATH, nrows: int = None) -> pd.DataFrame:
    """Convenience wrapper: load raw CSV and immediately select feature columns."""
    df = load_raw_dataframe(csv_path, nrows=nrows)
    df = select_feature_columns(df)
    return df


if __name__ == "__main__":
    download_dataset()
    df = load_dataset()
    print(df.head())
    print(df[config.LABEL_COLUMN].value_counts())
