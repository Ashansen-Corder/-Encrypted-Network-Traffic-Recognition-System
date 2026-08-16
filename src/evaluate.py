"""
Stage 4.5 / final reporting — Testing the system.

Computes accuracy / precision / recall / F1, confusion matrices, and a
side-by-side comparison report across the three trained models.
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config
from .utils import get_logger

logger = get_logger(__name__)


def _predict(model, X, model_name: str):
    if model_name == "dnn":
        proba = model.predict(X, verbose=0)
        preds = np.argmax(proba, axis=1)
    else:
        preds = model.predict(X)
        proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return preds, proba


def evaluate_model(model, X_test, y_test, label_encoder, model_name: str) -> dict:
    class_names = list(label_encoder.classes_)
    preds, proba = _predict(model, X_test, model_name)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, preds),
        "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, preds, average="weighted", zero_division=0),
    }

    if proba is not None:
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC for {model_name}: {e}")
            metrics["roc_auc_ovr"] = np.nan

    report_txt = classification_report(y_test, preds, target_names=class_names, zero_division=0)
    report_path = os.path.join(config.RESULTS_DIR, f"classification_report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report — {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(report_txt)
    logger.info(f"Saved classification report -> {report_path}")

    plot_confusion_matrix(y_test, preds, class_names, model_name)

    logger.info(
        f"[{model_name}] accuracy={metrics['accuracy']:.4f} "
        f"f1_macro={metrics['f1_macro']:.4f} f1_weighted={metrics['f1_weighted']:.4f}"
    )
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{model_name} — Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"{model_name} — Confusion Matrix (normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    out_path = os.path.join(config.RESULTS_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix plot -> {out_path}")


def plot_dnn_history(history_path: str = None):
    history_path = history_path or os.path.join(config.RESULTS_DIR, "dnn_history.json")
    if not os.path.exists(history_path):
        logger.warning(f"No DNN history file found at {history_path}, skipping plot.")
        return
    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("DNN Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="val")
    axes[1].set_title("DNN Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(config.RESULTS_DIR, "dnn_training_history.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved DNN training history plot -> {out_path}")


def compare_models(all_metrics: list) -> pd.DataFrame:
    df = pd.DataFrame(all_metrics).set_index("model")
    df = df.sort_values("f1_macro", ascending=False)
    out_path = os.path.join(config.RESULTS_DIR, "model_comparison.csv")
    df.to_csv(out_path)
    logger.info(f"Saved model comparison table -> {out_path}")

    plt.figure(figsize=(8, 5))
    df[["accuracy", "f1_macro", "f1_weighted"]].plot(kind="bar", ax=plt.gca())
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    bar_path = os.path.join(config.RESULTS_DIR, "model_comparison.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    logger.info(f"Saved model comparison bar chart -> {bar_path}")

    logger.info("\n" + df.to_string())
    return df


def evaluate_all(models: dict, X_test, y_test, label_encoder) -> pd.DataFrame:
    all_metrics = []
    for name in ("rf", "xgb", "dnn"):
        if name in models:
            all_metrics.append(evaluate_model(models[name], X_test, y_test, label_encoder, name))
    if "dnn" in models:
        plot_dnn_history()
    return compare_models(all_metrics)
