"""Small shared helpers: logging, timing, reproducibility."""
import logging
import random
import sys
import time
from contextlib import contextmanager

import numpy as np


def get_logger(name: str = "nia") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


@contextmanager
def timer(label: str, logger: logging.Logger = None):
    logger = logger or get_logger()
    start = time.time()
    logger.info(f"START  | {label}")
    yield
    elapsed = time.time() - start
    logger.info(f"DONE   | {label} ({elapsed:.2f}s)")
