"""Constantes y contratos del pipeline.

Centraliza la configuración reproducible: semilla, tamaños, nombres de columnas
y rutas. Cualquier módulo que necesite estos valores los importa de aquí.
"""
from __future__ import annotations

from pathlib import Path

# ── Reproducibilidad ────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_REGISTROS = 500          # ≥ 300 exigidos por el enunciado
TEST_SIZE = 0.20           # split 80/20
CV_FOLDS = 5               # k-fold cross-validation

# ── Esquema del dataset ─────────────────────────────────────────────────────
TARGET = "renuncia"

# Todas las features son numéricas → todas se estandarizan.
FEATURES: list[str] = [
    "edad",
    "años_en_empresa",
    "salario_mensual",
    "horas_extra_semana",
    "satisfaccion_laboral",
    "num_proyectos_año",
    "distancia_casa_trabajo_km",
    "ultima_evaluacion_desempeño",
    "capacitaciones_recibidas",
    "tiene_ascenso_ultimos_2_años",
]

# Rangos válidos (contrato de generación y validación).
RANGOS: dict[str, tuple[float, float]] = {
    "edad": (22, 60),
    "años_en_empresa": (0, 20),
    "salario_mensual": (800, 6000),
    "horas_extra_semana": (0, 20),
    "satisfaccion_laboral": (1, 5),
    "num_proyectos_año": (1, 10),
    "distancia_casa_trabajo_km": (1, 80),
    "ultima_evaluacion_desempeño": (0.0, 1.0),
    "capacitaciones_recibidas": (0, 5),
    "tiene_ascenso_ultimos_2_años": (0, 1),
}

# ── Rutas ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figuras"
DATASET_CSV = DATA_DIR / "empleados.csv"
