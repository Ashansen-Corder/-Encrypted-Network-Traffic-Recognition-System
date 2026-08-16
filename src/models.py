"""
Stage 4.4 — Trying out methods.

Builders for the three candidate classifiers described in the proposal:
  1. Random Forest Classifier
  2. XGBoost Classifier
  3. Deep Neural Network (Keras/TensorFlow)
"""
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from . import config


def build_random_forest(params: dict = None) -> RandomForestClassifier:
    params = params or config.RF_DEFAULT_PARAMS
    return RandomForestClassifier(**params)


def build_xgboost(num_classes: int, params: dict = None) -> XGBClassifier:
    params = dict(params or config.XGB_DEFAULT_PARAMS)
    params["num_class"] = num_classes
    params["objective"] = "multi:softprob"
    return XGBClassifier(**params)


def build_dnn(input_dim: int, num_classes: int, params: dict = None):
    """Multi-layer feed-forward network with dropout + batch normalization + L2 regularization."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    params = params or config.DNN_PARAMS
    reg = regularizers.l2(params["l2_reg"])

    model = models.Sequential(name="traffic_dnn")
    model.add(layers.Input(shape=(input_dim,)))

    for i, units in enumerate(params["hidden_layers"]):
        model.add(layers.Dense(units, kernel_regularizer=reg, name=f"dense_{i}"))
        model.add(layers.BatchNormalization(name=f"bn_{i}"))
        model.add(layers.Activation("relu", name=f"relu_{i}"))
        model.add(layers.Dropout(params["dropout_rate"], name=f"dropout_{i}"))

    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
