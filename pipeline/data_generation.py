"""Generación del dataset sintético con lógica de coherencia.

Regla OBLIGATORIA del enunciado: el dataset NO puede ser aleatorio puro. Aquí se
codifican relaciones realistas entre variables y entre las variables y la fuga
(`renuncia`). Las reglas se documentan en `REGLAS_COHERENCIA` para el informe.

Modelo de generación:
  1. Se muestrean features con dependencias realistas entre sí (p.ej. la
     satisfacción baja cuando hay muchas horas extra y la distancia es grande).
  2. La probabilidad de renuncia se deriva de un modelo logístico latente que
     combina los factores de riesgo; luego se muestrea Bernoulli(p) + ruido.
Así el target tiene señal real pero NO es perfectamente separable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Documentación legible de las reglas (para el informe / preguntas técnicas).
REGLAS_COHERENCIA = [
    "La satisfacción laboral baja con muchas horas extra y mayor distancia al trabajo, "
    "y sube con ascensos recientes y mejor salario.",
    "El salario crece con la edad, los años en la empresa y la última evaluación.",
    "La probabilidad de ascenso aumenta con la evaluación de desempeño y la antigüedad.",
    "La renuncia (target) aumenta con: baja satisfacción, muchas horas extra, salario "
    "bajo (relativo), larga distancia, baja evaluación, ausencia de ascenso, pocas "
    "capacitaciones y carga de proyectos extrema (muy pocos o demasiados).",
    "Se añade ruido aleatorio para que las clases no sean perfectamente separables.",
]


def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generar_dataset(n: int = config.N_REGISTROS,
                    seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Devuelve un DataFrame de ``n`` empleados con la lógica de coherencia."""
    rng = np.random.default_rng(seed)

    # ── Features base ───────────────────────────────────────────────────────
    edad = rng.integers(22, 61, size=n)

    # Años en la empresa: acotado por la edad (no se puede llevar más años que
    # los trabajables) y sesgado hacia valores bajos.
    max_años = np.minimum(20, edad - 21)
    años_en_empresa = np.floor(rng.beta(1.6, 3.0, size=n) * (max_años + 1)).astype(int)
    años_en_empresa = np.clip(años_en_empresa, 0, 20)

    # Evaluación de desempeño: beta centrada ~0.7.
    ultima_evaluacion = np.clip(rng.beta(6, 3, size=n), 0.0, 1.0)

    # Horas extra: muchos en 0-8, cola hacia 20.
    horas_extra = np.clip(rng.gamma(2.0, 3.0, size=n), 0, 20)

    distancia = np.clip(rng.gamma(2.2, 12.0, size=n), 1, 80)
    num_proyectos = rng.integers(1, 11, size=n)
    capacitaciones = rng.integers(0, 6, size=n)

    # Salario: coherente con edad, antigüedad y evaluación.
    salario = (
        900
        + 22.0 * (edad - 22)
        + 70.0 * años_en_empresa
        + 1400.0 * ultima_evaluacion
        + rng.normal(0, 300, size=n)
    )
    salario = np.clip(salario, 800, 6000)

    # Ascenso reciente: más probable con buena evaluación y antigüedad media-alta.
    z_asc = -1.2 + 2.5 * (ultima_evaluacion - 0.6) + 0.08 * años_en_empresa
    tiene_ascenso = (rng.random(n) < _sigmoide(z_asc)).astype(int)

    # Satisfacción laboral (1-5): derivada de otros factores → coherencia interna.
    sat_latente = (
        3.2
        - 0.09 * horas_extra
        - 0.012 * distancia
        + 0.7 * tiene_ascenso
        + 0.0003 * (salario - 2500)
        + rng.normal(0, 0.6, size=n)
    )
    satisfaccion = np.clip(np.rint(sat_latente), 1, 5).astype(int)

    # ── Probabilidad latente de renuncia ────────────────────────────────────
    sal_norm = (salario - salario.mean()) / (salario.std() + 1e-9)
    z = (
        -1.30
        + 0.90 * (3 - satisfaccion)              # baja satisfacción ↑
        + 1.80 * (horas_extra / 20.0)            # muchas horas extra ↑
        - 0.85 * sal_norm                        # salario bajo ↑
        + 1.00 * (distancia / 80.0)              # lejos ↑
        - 1.30 * (ultima_evaluacion - 0.5)       # baja evaluación ↑
        - 0.95 * tiene_ascenso                   # sin ascenso ↑
        - 0.22 * capacitaciones                  # pocas capacitaciones ↑
        + 0.60 * (np.abs(num_proyectos - 5) / 5) # carga extrema ↑
        + rng.normal(0, 0.30, size=n)            # ruido (no separable)
    )
    p_renuncia = _sigmoide(z)
    renuncia = (rng.random(n) < p_renuncia).astype(int)

    df = pd.DataFrame({
        "edad": edad,
        "años_en_empresa": años_en_empresa,
        "salario_mensual": np.round(salario, 2),
        "horas_extra_semana": np.round(horas_extra, 1),
        "satisfaccion_laboral": satisfaccion,
        "num_proyectos_año": num_proyectos,
        "distancia_casa_trabajo_km": np.round(distancia, 1),
        "ultima_evaluacion_desempeño": np.round(ultima_evaluacion, 3),
        "capacitaciones_recibidas": capacitaciones,
        "tiene_ascenso_ultimos_2_años": tiene_ascenso,
        config.TARGET: renuncia,
    })
    validar_rangos(df)
    return df


def validar_rangos(df: pd.DataFrame) -> None:
    """Falla ruidoso si alguna columna sale de su rango contractual."""
    for col, (lo, hi) in config.RANGOS.items():
        vmin, vmax = df[col].min(), df[col].max()
        if vmin < lo or vmax > hi:
            raise ValueError(
                f"Columna '{col}' fuera de rango [{lo}, {hi}]: observado [{vmin}, {vmax}]"
            )


def guardar_dataset(df: pd.DataFrame, ruta=config.DATASET_CSV) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8")
