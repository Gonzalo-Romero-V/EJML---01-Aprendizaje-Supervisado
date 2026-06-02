# Aprendizaje Supervisado — Predicción de renuncia de empleados

Ejercicio de clasificación binaria supervisada: a partir de un conjunto de datos
sintético y coherente sobre la permanencia de empleados, se entrenan y comparan
tres algoritmos para predecir si un empleado renunciará dentro de los próximos
seis meses, evaluando su valor predictivo con métricas objetivas.

## Contenido

- **Dataset sintético** (500 registros) generado por código con reglas de
  coherencia documentadas (`pipeline/data_generation.py`).
- **EDA**: estadísticas descriptivas, heatmap de correlaciones, distribución y
  balance de clases, boxplots.
- **Tres modelos** (scikit-learn): Regresión Logística, Random Forest y
  Gradient Boosting.
- **Evaluación**: Accuracy, Precisión, Recall, F1-Score y AUC-ROC; curva ROC
  comparativa, matriz de confusión e importancia de variables.
- **Validación y optimización**: validación cruzada (k=5) y búsqueda de
  hiperparámetros con `GridSearchCV`, comparando modelo base vs. optimizado.

## Estructura

```
.
├── main.py                 # Orquesta el pipeline completo de punta a punta
├── pipeline/               # Lógica del pipeline
│   ├── config.py           # Constantes reproducibles (semilla, esquema, rutas)
│   ├── data_generation.py  # Generación del dataset sintético
│   ├── preprocessing.py    # Split 80/20 + estandarización
│   ├── eda.py              # Análisis exploratorio y figuras
│   ├── models.py           # Definición y entrenamiento de los 3 modelos
│   ├── evaluation.py       # Métricas y visualizaciones
│   └── tuning.py           # Cross-validation y GridSearchCV
├── notebooks/
│   └── informe_supervisado.ipynb   # Análisis ejecutable en Jupyter
├── docs/figuras/           # Salida de figuras (generadas por main.py; no versionadas)
└── requirements.txt
```

## Instalación

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

## Uso

Ejecutar el pipeline completo (genera el dataset, corre el EDA, entrena y evalúa
los modelos, optimiza el mejor y guarda las figuras en `docs/figuras/`):

```bash
python main.py
```

También se puede recorrer paso a paso en el notebook
`notebooks/informe_supervisado.ipynb`.

## Reproducibilidad

Toda la aleatoriedad está fijada con `RANDOM_STATE = 42` (`pipeline/config.py`) y
las versiones de las dependencias están ancladas en `requirements.txt`. Los
artefactos derivados —el dataset y las figuras— no se versionan: se regeneran de
forma determinista al ejecutar `python main.py`, produciendo siempre los mismos
resultados (métricas y figuras idénticas).
