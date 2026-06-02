"""Optimización: cross-validation y búsqueda de hiperparámetros.

- cross-validation k=5 (F1) sobre el mejor modelo.
- GridSearchCV optimizando ≥3 hiperparámetros del mejor modelo.
- Comparación base vs optimizado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score

from . import config
from .evaluation import metricas_modelo

# Grids con ≥3 hiperparámetros por modelo.
GRIDS: dict[str, dict] = {
    "Regresión Logística": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3, 4],
    },
}


def cross_validation_f1(modelo, X, y, k: int = config.CV_FOLDS) -> tuple[float, float]:
    """Media y desviación estándar del F1 en k-fold."""
    scores = cross_val_score(clone(modelo), X, y, cv=k, scoring="f1")
    return float(scores.mean()), float(scores.std())


def optimizar(nombre_modelo: str, modelo_base, X_train, y_train) -> GridSearchCV:
    """GridSearchCV (cv=5, scoring=f1) sobre el grid del modelo indicado."""
    grid = GRIDS[nombre_modelo]
    busqueda = GridSearchCV(
        clone(modelo_base), grid, cv=config.CV_FOLDS, scoring="f1", n_jobs=1)
    busqueda.fit(X_train, y_train)
    return busqueda


def comparar_base_vs_opt(modelo_base, modelo_opt, X_test, y_test) -> pd.DataFrame:
    """Tabla de métricas modelo base vs optimizado."""
    filas = {
        "Base": metricas_modelo(modelo_base, X_test, y_test),
        "Optimizado": metricas_modelo(modelo_opt, X_test, y_test),
    }
    return pd.DataFrame(filas).T.round(4)


def seleccionar_mejor(tabla_metricas: pd.DataFrame) -> str:
    """Nombre del mejor modelo según F1 (índice de la tabla ordenada)."""
    return str(tabla_metricas.index[0])
