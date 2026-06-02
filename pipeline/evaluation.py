"""Evaluación y visualización de modelos.

Métricas: Accuracy, Precisión, Recall, F1, AUC-ROC. Figuras: curva ROC
comparativa, matriz de confusión, feature importance. Funciones puras.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def metricas_modelo(modelo, X_test, y_test) -> dict:
    """Diccionario de métricas para un modelo entrenado."""
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precisión": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
    }


def tabla_metricas(modelos: dict, X_test, y_test) -> pd.DataFrame:
    """Tabla (modelos × métricas) ordenada por F1 descendente."""
    filas = {nombre: metricas_modelo(m, X_test, y_test)
             for nombre, m in modelos.items()}
    return pd.DataFrame(filas).T.sort_values("F1", ascending=False).round(4)


def fig_roc_comparativa(modelos: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for nombre, modelo in modelos.items():
        y_proba = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC comparativa")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_matriz_confusion(modelo, X_test, y_test, nombre: str):
    cm = confusion_matrix(y_test, modelo.predict(X_test))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    etiquetas = ["Se quedó (0)", "Renunció (1)"]
    ax.set_xticks([0, 1], labels=etiquetas)
    ax.set_yticks([0, 1], labels=etiquetas)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {nombre}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def fig_feature_importance(modelo, feature_names: list[str], nombre: str):
    """Para modelos con feature_importances_ (Random Forest, Gradient Boosting)."""
    if not hasattr(modelo, "feature_importances_"):
        raise ValueError(f"{nombre} no expone feature_importances_")
    importancias = modelo.feature_importances_
    orden = np.argsort(importancias)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(feature_names)[orden], importancias[orden], color="#55A868")
    ax.set_title(f"Feature importance — {nombre}")
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    return fig
