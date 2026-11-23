# Fase 4 — Modelado

**Notebook:** `modelos_clasicos.ipynb` (ML tradicional sobre Kaggle, Buena vs. Mala; solo train y val, **el test no se usa**).
**Código:** `../src/models_classic.py` (modelos y mallas), `../src/features.py` (histograma HSV), `../src/cnn.py` (CNN 32-64-128).
**Resultados:** `resultados_modelos_clasicos.csv` y `figuras/` (PDF vectorial). Los modelos entrenados se guardan en `../data/processed/models/` (no versionado).

**Qué hace:** línea base mayoritaria, regresión logística, SVM RBF y Random Forest con `GridSearchCV` y validación cruzada agrupada por objeto, más la línea base "solo borde" para medir cuánto del rendimiento puede venir del fondo.

**Pendiente:** clase *Regular* (tres clases), CNN entrenada, y la evaluación en fotos propias y en frutas de *Mixed*.

**Alimenta rúbrica:** 4, 5, 6.
Estado: 🟡 ML tradicional ejecutado en la tarea binaria; faltan *Regular* y la CNN.
