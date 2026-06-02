"""Análisis exploratorio.

Funciones puras: devuelven estadísticas (DataFrame/Series) o figuras
(matplotlib Figure). No llaman a plt.show() ni guardan a disco — eso lo decide
el llamador (notebook = mostrar, main.py = guardar).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config


def estadisticas_descriptivas(df: pd.DataFrame) -> pd.DataFrame:
    """Media, std, min, max y percentiles de todas las columnas."""
    return df.describe(percentiles=[0.25, 0.5, 0.75]).T


def correlaciones_top(df: pd.DataFrame, n: int = 3) -> pd.Series:
    """Las ``n`` correlaciones (en valor absoluto) más fuertes con el target."""
    corr = df.corr(numeric_only=True)[config.TARGET].drop(config.TARGET)
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(n)


def balance_clases(df: pd.DataFrame) -> pd.Series:
    """Proporción de cada clase del target (0/1)."""
    return df[config.TARGET].value_counts(normalize=True).sort_index()


def fig_heatmap_correlacion(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Heatmap de correlación entre variables")
    fig.tight_layout()
    return fig


def fig_distribucion_target(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 4))
    conteo = df[config.TARGET].value_counts().sort_index()
    ax.bar(["Se quedó (0)", "Renunció (1)"], conteo.values,
           color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(conteo.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("Distribución de la variable renuncia")
    ax.set_ylabel("Nº de empleados")
    fig.tight_layout()
    return fig


def fig_boxplots(df: pd.DataFrame):
    """Boxplots de salario y satisfacción separados por renuncia."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=df, x=config.TARGET, y="salario_mensual", ax=axes[0])
    axes[0].set_title("Salario mensual por renuncia")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Se quedó", "Renunció"])
    sns.boxplot(data=df, x=config.TARGET, y="satisfaccion_laboral", ax=axes[1])
    axes[1].set_title("Satisfacción laboral por renuncia")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Se quedó", "Renunció"])
    fig.tight_layout()
    return fig
