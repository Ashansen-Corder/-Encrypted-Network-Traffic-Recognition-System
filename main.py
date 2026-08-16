#!/usr/bin/env python3
"""
End-to-end CLI for the Encrypted Network Traffic Recognition System.

Examples
--------
Full pipeline (download -> preprocess -> PCA -> train all 3 models -> evaluate):
    python main.py --all

Quick smoke test on a 20k-row subsample:
    python main.py --all --sample 20000

Run stages independently:
    python main.py --download
    python main.py --preprocess
    python main.py --train rf
    python main.py --train xgb
    python main.py --train dnn
    python main.py --train all
    python main.py --evaluate
"""
import argparse
import json
import os

import joblib
import numpy as np

from src import config
from src.data_loader import download_dataset, load_dataset
from src.dimensionality_reduction import apply_pca, fit_pca
from src.evaluate import evaluate_all
from src.preprocessing import preprocess_pipeline
from src.train import train_all
from src.utils import get_logger, set_global_seed, timer

logger = get_logger("main")


def stage_download(force: bool = False):
    download_dataset(force=force)


def stage_preprocess(sample: int = None, use_pca: bool = True):
    df = load_dataset(nrows=sample)
    processed = preprocess_pipeline(df)

    if use_pca:
        pca = fit_pca(processed["X_train"])
        X_train, X_val, X_test = apply_pca(pca, processed["X_train"], processed["X_val"], processed["X_test"])
        processed["X_train"], processed["X_val"], processed["X_test"] = X_train, X_val, X_test
        processed["pca"] = pca

    np.savez_compressed(
        config.SPLIT_CACHE_PATH,
        X_train=processed["X_train"], X_val=processed["X_val"], X_test=processed["X_test"],
        y_train=processed["y_train"], y_val=processed["y_val"], y_test=processed["y_test"],
    )
    logger.info(f"Cached processed splits -> {config.SPLIT_CACHE_PATH}")
    return processed


def load_cached_splits():
    if not os.path.exists(config.SPLIT_CACHE_PATH):
        raise FileNotFoundError(
            "No cached processed splits found. Run `python main.py --preprocess` first."
        )
    npz = np.load(config.SPLIT_CACHE_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
    return {
        "X_train": npz["X_train"], "X_val": npz["X_val"], "X_test": npz["X_test"],
        "y_train": npz["y_train"], "y_val": npz["y_val"], "y_test": npz["y_test"],
        "label_encoder": label_encoder,
    }


def stage_train(which, processed=None):
    processed = processed or load_cached_splits()
    if which == "all":
        models_to_train = ("rf", "xgb", "dnn")
    else:
        models_to_train = (which,)
    return train_all(processed, models_to_train)


def load_trained_models(which=("rf", "xgb", "dnn")):
    import tensorflow as tf
    from xgboost import XGBClassifier

    models = {}
    if "rf" in which and os.path.exists(config.RF_MODEL_PATH):
        models["rf"] = joblib.load(config.RF_MODEL_PATH)
    if "xgb" in which and os.path.exists(config.XGB_MODEL_PATH):
        model = XGBClassifier()
        model.load_model(config.XGB_MODEL_PATH)
        models["xgb"] = model
    if "dnn" in which and os.path.exists(config.DNN_MODEL_PATH):
        models["dnn"] = tf.keras.models.load_model(config.DNN_MODEL_PATH)
    return models


def stage_evaluate(processed=None, trained_models=None):
    processed = processed or load_cached_splits()
    trained_models = trained_models or load_trained_models()
    if not trained_models:
        raise RuntimeError("No trained models found. Run `python main.py --train all` first.")
    return evaluate_all(trained_models, processed["X_test"], processed["y_test"], processed["label_encoder"])


def main():
    parser = argparse.ArgumentParser(description="Encrypted Network Traffic Recognition — pipeline runner")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline end to end.")
    parser.add_argument("--download", action="store_true", help="Download the dataset only.")
    parser.add_argument("--preprocess", action="store_true", help="Clean, split, scale, and PCA-transform the data.")
    parser.add_argument("--train", choices=["rf", "xgb", "dnn", "all"], help="Train the given model(s).")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate all trained models found on disk.")
    parser.add_argument("--sample", type=int, default=None, help="Use only N rows (quick smoke test).")
    parser.add_argument("--no-download", action="store_true", help="Skip the download step in --all.")
    parser.add_argument("--no-pca", action="store_true", help="Skip PCA and train on scaled raw features.")
    parser.add_argument("--search", choices=["quick", "grid"], default=None,
                         help="quick = default hyperparameters (fast); grid = GridSearchCV tuning (slow).")
    args = parser.parse_args()

    if args.search:
        config.SEARCH_MODE = args.search

    set_global_seed(config.RANDOM_STATE)

    if not any([args.all, args.download, args.preprocess, args.train, args.evaluate]):
        parser.print_help()
        return

    if args.all:
        with timer("FULL PIPELINE", logger):
            if not args.no_download:
                stage_download()
            processed = stage_preprocess(sample=args.sample, use_pca=not args.no_pca)
            trained = stage_train("all", processed=processed)
            evaluate_all(trained, processed["X_test"], processed["y_test"], processed["label_encoder"])
        return

    if args.download:
        stage_download()
    if args.preprocess:
        stage_preprocess(sample=args.sample, use_pca=not args.no_pca)
    if args.train:
        stage_train(args.train)
    if args.evaluate:
        stage_evaluate()


if __name__ == "__main__":
    main()
