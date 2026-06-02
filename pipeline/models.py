"""Definición y entrenamiento de los 3 modelos.

- Regresión Logística  (base de referencia)
- Random Forest         (ensamble)
- Gradient Boosting     (sklearn.GradientBoostingClassifier; el enunciado admite
                         también XGBoost — se usa sklearn por reproducibilidad y
                         cero dependencias extra)
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from . import config


def construir_modelos() -> dict:
    """Devuelve los 3 modelos sin entrenar, con random_state fijo.

    Respuesta al desbalance (67/33) detectado en el EDA: se aplica
    ``class_weight='balanced'`` a los modelos que lo soportan nativamente
    (LogReg y Random Forest), penalizando más el error sobre la clase
    minoritaria (renuncia) → mayor recall. Gradient Boosting se deja por
    defecto, sirviendo además como punto de comparación del efecto del balanceo.
    """
    return {
        "Regresión Logística": LogisticRegression(
            random_state=config.RANDOM_STATE, max_iter=1000,
            class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            random_state=config.RANDOM_STATE, n_estimators=200,
            class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=config.RANDOM_STATE),
    }


def entrenar_modelos(X_train, y_train) -> dict:
    """Entrena los 3 modelos y los devuelve en un dict {nombre: modelo}."""
    modelos = construir_modelos()
    for modelo in modelos.values():
        modelo.fit(X_train, y_train)
    return modelos
