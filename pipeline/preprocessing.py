"""Preprocesamiento: split 80/20 y estandarización.

Invariante anti-fuga: el StandardScaler se AJUSTA solo con el conjunto de
entrenamiento y se aplica (transform) a test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class DatosPreparados:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    feature_names: list[str]


def split_y_escalar(df: pd.DataFrame) -> DatosPreparados:
    X = df[config.FEATURES].to_numpy(dtype=float)
    y = df[config.TARGET].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)   # ajustar SOLO en train
    X_test_s = scaler.transform(X_test)

    return DatosPreparados(
        X_train=X_train_s,
        X_test=X_test_s,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        feature_names=list(config.FEATURES),
    )
