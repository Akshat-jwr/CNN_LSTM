"""CNN-LSTM architectures (kept as core model family) and lightweight tabular anchors."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@dataclass(frozen=True)
class CNNLSTMConfig:
    conv_filters: int = 32
    kernel_size: int = 3
    lstm_units: int = 32
    dropout: float = 0.30
    rec_dropout: float = 0.10
    l2_reg: float = 5e-4
    lr: float = 5e-4
    batch_size: int = 64
    epochs: int = 30
    task: str = "regression"  # "regression" | "classification"
    norm_type: str = "layer"  # "layer" | "batch"
    use_pooling: bool = True
    n_seeds: int = 3


def build_cnn_lstm(input_shape: tuple[int, int], cfg: CNNLSTMConfig) -> Model:
    """
    Build the CNN-LSTM hybrid model:
    Causal Conv1D -> Normalization -> (MaxPooling1D) -> Dropout -> LSTM -> Normalization -> Dropout -> Dense(1).
    """
    reg = regularizers.l2(cfg.l2_reg) if cfg.l2_reg > 0 else None
    norm_cls = LayerNormalization if cfg.norm_type == "layer" else BatchNormalization

    inputs = Input(shape=input_shape, name="inputs")
    x = Conv1D(
        filters=cfg.conv_filters,
        kernel_size=cfg.kernel_size,
        activation="relu",
        padding="causal",
        kernel_regularizer=reg,
        name="conv1d",
    )(inputs)
    x = norm_cls(name="norm_conv")(x)
    if cfg.use_pooling:
        x = MaxPooling1D(pool_size=2, name="maxpool")(x)
    x = Dropout(cfg.dropout, name="dropout_conv")(x)

    x = LSTM(
        units=cfg.lstm_units,
        return_sequences=False,
        dropout=cfg.dropout,
        recurrent_dropout=cfg.rec_dropout,
        kernel_regularizer=reg,
        name="lstm",
    )(x)
    x = norm_cls(name="norm_lstm")(x)
    x = Dropout(cfg.dropout, name="dropout_lstm")(x)

    if cfg.task == "classification":
        outputs = Dense(1, activation="sigmoid", name="output")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    elif cfg.task == "regression":
        outputs = Dense(1, activation="linear", name="output")(x)
        loss = Huber(delta=1.0)
        metrics = ["mae"]
    else:
        raise ValueError(f"Unknown task={cfg.task!r}")

    model = Model(inputs, outputs, name="cnn_lstm")
    model.compile(optimizer=Adam(learning_rate=cfg.lr, clipnorm=1.0), loss=loss, metrics=metrics)
    return model


def fit_predict_cnn_lstm_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cfg: CNNLSTMConfig,
    base_seed: int = 42,
    sample_weight: np.ndarray | None = None,
    val_fraction: float = 0.12,
) -> np.ndarray:
    """
    Fit a seed-averaged CNN-LSTM ensemble using strictly chronological trailing validation
    (with no random shuffle of validation) for EarlyStopping / ReduceLROnPlateau.
    """
    n_tr = len(X_train)
    n_val = max(32, int(n_tr * val_fraction))
    if n_tr - n_val < 64:
        raise ValueError(f"Training set too small ({n_tr}) for chronological val split.")

    X_fit, y_fit = X_train[:-n_val], y_train[:-n_val]
    X_es, y_es = X_train[-n_val:], y_train[-n_val:]
    sw_fit = sample_weight[:-n_val] if sample_weight is not None else None

    # Fit StandardScaler strictly on X_fit (reshape to 2D over samples*timesteps)
    n_s, seq_len, n_f = X_fit.shape
    sc = StandardScaler().fit(X_fit.reshape(-1, n_f))
    X_fit_s = sc.transform(X_fit.reshape(-1, n_f)).reshape(n_s, seq_len, n_f)
    X_es_s = sc.transform(X_es.reshape(-1, n_f)).reshape(len(X_es), seq_len, n_f)
    X_test_s = sc.transform(X_test.reshape(-1, n_f)).reshape(len(X_test), seq_len, n_f)

    preds = []
    for s_idx in range(cfg.n_seeds):
        tf.keras.backend.clear_session()
        set_global_seed(base_seed + s_idx)
        model = build_cnn_lstm((seq_len, n_f), cfg)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0),
        ]
        model.fit(
            X_fit_s,
            y_fit,
            validation_data=(X_es_s, y_es),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            shuffle=True,
            sample_weight=sw_fit,
            verbose=0,
            callbacks=callbacks,
        )
        p = model(X_test_s, training=False).numpy().reshape(-1)
        preds.append(p)

    return np.mean(preds, axis=0)


def fit_predict_linear_anchor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task: str = "regression",
    alpha: float = 100.0,
) -> np.ndarray:
    """
    Lightweight tabular anchor using the last timestep of each sequence (day `t` features)
    to verify whether signal exists before deep sequence modeling.
    """
    X_tr = X_train[:, -1, :]
    X_te = X_test[:, -1, :]
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    X_te_s = sc.transform(X_te)
    if task == "classification":
        clf = LogisticRegression(C=1.0 / max(alpha, 1e-3), max_iter=1000)
        clf.fit(X_tr_s, y_train.astype(int))
        return clf.predict_proba(X_te_s)[:, 1]
    reg = Ridge(alpha=alpha)
    reg.fit(X_tr_s, y_train)
    return reg.predict(X_te_s)


def fit_predict_lgbm_anchor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Shallow regularized LightGBM anchor on the last sequence timestep (day `t` features)
    for direct OOS comparison against CNN-LSTM.
    """
    X_tr = X_train[:, -1, :]
    X_te = X_test[:, -1, :]
    model = lgb.LGBMRegressor(
        n_estimators=120,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=150,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=5.0,
        reg_lambda=20.0,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X_tr, y_train)
    return model.predict(X_te)
