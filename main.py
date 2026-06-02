"""Punto de entrada del Ejercicio 1 - Aprendizaje Supervisado.

Orquesta el pipeline completo de punta a punta SIN Jupyter (reproducibilidad:
"corre de inicio a fin sin errores"). Genera el dataset, corre el EDA, entrena
y evalúa los 3 modelos, optimiza el mejor y guarda todas las figuras en
docs/figuras/. La lógica vive en pipeline/; aquí solo se orquesta y se persiste.
"""
from __future__ import annotations

import sys

# Consolas Windows (cp1252) no codifican '→'/tildes en stdout; forzar UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import matplotlib

matplotlib.use("Agg")  # backend headless: guardar figuras sin mostrar ventana

from pipeline import (
    config,
    data_generation,
    eda,
    evaluation,
    models,
    preprocessing,
    tuning,
)


def _guardar(fig, nombre: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.FIGURES_DIR / nombre
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    print(f"  figura → {ruta.relative_to(config.ROOT)}")


def main() -> None:
    print("=" * 70)
    print("  Ejercicio 1 — Aprendizaje Supervisado (predicción de renuncia)")
    print("=" * 70)

    # ── 1. Dataset ──────────────────────────────────────────────────────────
    print("\n[1] Generando dataset sintético…")
    df = data_generation.generar_dataset()
    data_generation.guardar_dataset(df)
    print(f"  {len(df)} registros → {config.DATASET_CSV.relative_to(config.ROOT)}")
    print("  Reglas de coherencia aplicadas:")
    for r in data_generation.REGLAS_COHERENCIA:
        print(f"    · {r}")

    # ── 2. EDA ──────────────────────────────────────────────────────────────
    print("\n[2] Análisis exploratorio…")
    print(eda.estadisticas_descriptivas(df).round(2).to_string())
    print("\n  Top 3 correlaciones con renuncia:")
    print(eda.correlaciones_top(df).round(3).to_string())
    balance = eda.balance_clases(df)
    print(f"\n  Balance de clases: {balance.round(3).to_dict()}")
    _guardar(eda.fig_heatmap_correlacion(df), "01_heatmap_correlacion.png")
    _guardar(eda.fig_distribucion_target(df), "02_distribucion_renuncia.png")
    _guardar(eda.fig_boxplots(df), "03_boxplots.png")

    # ── 3. Preprocesamiento ─────────────────────────────────────────────────
    print("\n[3] Split 80/20 + estandarización (fit solo en train)…")
    datos = preprocessing.split_y_escalar(df)
    print(f"  train={len(datos.y_train)}  test={len(datos.y_test)}")

    # ── 4. Modelado ─────────────────────────────────────────────────────────
    print("\n[4] Entrenando 3 modelos…")
    modelos = models.entrenar_modelos(datos.X_train, datos.y_train)
    tabla = evaluation.tabla_metricas(modelos, datos.X_test, datos.y_test)
    print(tabla.to_string())
    _guardar(evaluation.fig_roc_comparativa(modelos, datos.X_test, datos.y_test),
             "04_roc_comparativa.png")
    for nombre, modelo in modelos.items():
        slug = nombre.lower().replace(" ", "_").replace("ó", "o").replace("í", "i")
        _guardar(evaluation.fig_matriz_confusion(modelo, datos.X_test, datos.y_test, nombre),
                 f"05_confusion_{slug}.png")
        if hasattr(modelo, "feature_importances_"):
            _guardar(evaluation.fig_feature_importance(modelo, datos.feature_names, nombre),
                     f"06_importance_{slug}.png")

    # ── 5. Optimización ──────────────────────────────────────────────────────
    print("\n[5] Optimización del mejor modelo…")
    mejor = tuning.seleccionar_mejor(tabla)
    print(f"  Mejor modelo por F1: {mejor}")
    cv_mean, cv_std = tuning.cross_validation_f1(
        modelos[mejor], datos.X_train, datos.y_train)
    print(f"  CV (k={config.CV_FOLDS}) F1 = {cv_mean:.4f} ± {cv_std:.4f}")
    busqueda = tuning.optimizar(mejor, modelos[mejor], datos.X_train, datos.y_train)
    print(f"  Mejores hiperparámetros: {busqueda.best_params_}")
    comparacion = tuning.comparar_base_vs_opt(
        modelos[mejor], busqueda.best_estimator_, datos.X_test, datos.y_test)
    print("\n  Base vs Optimizado:")
    print(comparacion.to_string())

    print("\n✓ Pipeline completado sin errores.")


if __name__ == "__main__":
    main()
