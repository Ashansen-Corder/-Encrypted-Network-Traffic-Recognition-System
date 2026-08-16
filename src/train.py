"""
Stage 4.4 / 4.5 — Trying out methods + testing the system (cross-validation).

Trains Random Forest, XGBoost, and a Deep Neural Network, each validated with
stratified K-fold cross-validation on the training set, then fit on the full
training set and persisted to disk.
"""
import json
import os

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from . import config
from .models import build_dnn, build_random_forest, build_xgboost
from .utils import get_logger, set_global_seed, timer

logger = get_logger(__name__)


def _cross_validate(estimator, X, y, name: str):
    with timer(f"{name}: {config.CV_FOLDS}-fold stratified cross-validation", logger):
        skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        scores = cross_val_score(estimator, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    logger.info(f"{name} CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}  (folds={scores})")
    return scores


def train_random_forest(X_train, y_train):
    set_global_seed(config.RANDOM_STATE)

    if config.SEARCH_MODE == "grid":
        base = build_random_forest(params=dict(n_jobs=-1, random_state=config.RANDOM_STATE,
                                                 class_weight="balanced"))
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
        with timer("Random Forest: GridSearchCV", logger):
            search = GridSearchCV(base, config.RF_PARAM_GRID, cv=skf, scoring="accuracy",
                                   n_jobs=-1, verbose=1)
            search.fit(X_train, y_train)
        logger.info(f"Best RF params: {search.best_params_}")
        model = search.best_estimator_
    else:
        model = build_random_forest()
        _cross_validate(model, X_train, y_train, "Random Forest")
        with timer("Random Forest: final fit on full training set", logger):
            model.fit(X_train, y_train)

    joblib.dump(model, config.RF_MODEL_PATH)
    logger.info(f"Saved Random Forest model -> {config.RF_MODEL_PATH}")
    return model


def train_xgboost(X_train, y_train, num_classes: int):
    set_global_seed(config.RANDOM_STATE)

    if config.SEARCH_MODE == "grid":
        base = build_xgboost(num_classes, params=dict(
            tree_method="hist", eval_metric="mlogloss",
            random_state=config.RANDOM_STATE, n_jobs=-1))
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
        with timer("XGBoost: GridSearchCV", logger):
            search = GridSearchCV(base, config.XGB_PARAM_GRID, cv=skf, scoring="accuracy",
                                   n_jobs=-1, verbose=1)
            search.fit(X_train, y_train)
        logger.info(f"Best XGB params: {search.best_params_}")
        model = search.best_estimator_
    else:
        model = build_xgboost(num_classes)
        _cross_validate(model, X_train, y_train, "XGBoost")
        with timer("XGBoost: final fit on full training set", logger):
            model.fit(X_train, y_train)

    model.save_model(config.XGB_MODEL_PATH)
    logger.info(f"Saved XGBoost model -> {config.XGB_MODEL_PATH}")
    return model


def train_dnn(X_train, y_train, X_val, y_val, num_classes: int):
    """Train and evaluate the Deep Neural Network with training safeguards."""
    import tensorflow as tf

    set_global_seed(config.RANDOM_STATE)
    params = config.DNN_PARAMS

    model = build_dnn(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        params=params,
    )
    model.summary(print_fn=logger.info)

    # Save the best-performing model based on validation loss.
    checkpoint_path = os.path.join(
        config.RESULTS_DIR,
        "best_dnn.keras",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=params["patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
    ]

    with timer("Deep Neural Network: training", logger):
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=2,
        )

    # Evaluate the trained DNN on the validation set.
    val_loss, val_accuracy = model.evaluate(
        X_val,
        y_val,
        verbose=0,
    )

    logger.info(
        f"Deep Neural Network validation loss: {val_loss:.4f}"
    )
    logger.info(
        f"Deep Neural Network validation accuracy: {val_accuracy:.4f}"
    )

    # Save the final trained model.
    model.save(config.DNN_MODEL_PATH)
    logger.info(
        f"Saved DNN model -> {config.DNN_MODEL_PATH}"
    )

    # Save the training history for later comparison and visualization.
    history_path = os.path.join(
        config.RESULTS_DIR,
        "dnn_history.json",
    )

    with open(history_path, "w") as f:
        json.dump(
            {
                k: [float(v) for v in vals]
                for k, vals in history.history.items()
            },
            f,
            indent=2,
        )

    logger.info(
        f"Saved DNN training history -> {history_path}"
    )

    return model, history
    import tensorflow as tf

    set_global_seed(config.RANDOM_STATE)
    params = config.DNN_PARAMS
    model = build_dnn(input_dim=X_train.shape[1], num_classes=num_classes, params=params)
    model.summary(print_fn=logger.info)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=params["patience"], restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    with timer("Deep Neural Network: training", logger):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=2,
        )

    model.save(config.DNN_MODEL_PATH)
    logger.info(f"Saved DNN model -> {config.DNN_MODEL_PATH}")

    history_path = os.path.join(config.RESULTS_DIR, "dnn_history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    logger.info(f"Saved DNN training history -> {history_path}")

    return model, history


def train_all(processed: dict, models_to_train=("rf", "xgb", "dnn")):
    """Train the requested subset of models. `processed` is the dict returned by
    preprocessing.preprocess_pipeline (optionally PCA-transformed)."""
    X_train, y_train = processed["X_train"], processed["y_train"]
    X_val, y_val = processed["X_val"], processed["y_val"]
    num_classes = len(processed["label_encoder"].classes_)

    trained = {}
    if "rf" in models_to_train:
        trained["rf"] = train_random_forest(X_train, y_train)
    if "xgb" in models_to_train:
        trained["xgb"] = train_xgboost(X_train, y_train, num_classes)
    if "dnn" in models_to_train:
        trained["dnn"], trained["dnn_history"] = train_dnn(X_train, y_train, X_val, y_val, num_classes)

    return trained
