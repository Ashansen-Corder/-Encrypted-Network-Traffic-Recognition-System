"""
Stage 4.3 — Making the information simple (Principal Component Analysis).

Fits PCA on the standardized training features, retaining enough components to
explain >= config.PCA_VARIANCE_TARGET of the total variance, and transforms
train/val/test consistently.
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from . import config
from .utils import get_logger, timer

logger = get_logger(__name__)


def fit_pca(X_train, variance_target: float = config.PCA_VARIANCE_TARGET, save: bool = True):
    with timer(f"Fitting PCA (target variance={variance_target})", logger):
        pca_full = PCA(random_state=config.RANDOM_STATE)
        pca_full.fit(X_train)
        cumulative = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative, variance_target) + 1)
        n_components = max(2, min(n_components, X_train.shape[1]))

        pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
        pca.fit(X_train)

    logger.info(
        f"Selected {n_components} components explaining "
        f"{np.sum(pca.explained_variance_ratio_):.4f} of variance "
        f"(reduced from {X_train.shape[1]} raw features)."
    )

    if save:
        joblib.dump(pca, config.PCA_PATH)
        logger.info(f"Saved PCA transformer -> {config.PCA_PATH}")
        plot_explained_variance(pca_full, n_components)

    return pca


def plot_explained_variance(pca_full: PCA, chosen_n: int, out_path: str = None):
    out_path = out_path or f"{config.RESULTS_DIR}/pca_variance.png"
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
    plt.axhline(config.PCA_VARIANCE_TARGET, color="red", linestyle="--",
                label=f"{config.PCA_VARIANCE_TARGET*100:.0f}% variance target")
    plt.axvline(chosen_n, color="green", linestyle="--", label=f"{chosen_n} components chosen")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA — Cumulative Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved PCA variance plot -> {out_path}")


def apply_pca(pca: PCA, *arrays):
    return [pca.transform(a) for a in arrays]
