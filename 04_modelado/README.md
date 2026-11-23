# Fase 4 — Modelado

Todo se entrena y valida con `train` y `val` del dataset de Kaggle (Buena vs. Mala); **el test no se usa** hasta el final. Las métricas de aquí **no demuestran que se aprenda calidad**: el fondo ya predice la clase (ver el EDA y los controles "solo borde").

| Archivo | Qué es |
|---|---|
| `modelos_clasicos.ipynb` | línea base mayoritaria, regresión logística, SVM y Random Forest con `GridSearchCV` agrupado, y línea base "solo borde" |
| `entrenar_cnn.py` | experimentos de la CNN (calidad y especie), reanudable: `python 04_modelado/entrenar_cnn.py --tarea calidad` |
| `cnn_experimentos.ipynb` | análisis de los 10 experimentos de calidad y del clasificador de especie (curvas, comparación, regla de 2 EE) |
| `fuga_particion.ipynb` | cuánto infla las métricas partir por imagen en vez de por objeto (≈ 0.027 de F1, cota inferior) |
| `resultados_*.csv` | tablas de resultados para el informe |
| `figuras/` | figuras vectoriales (PDF) |
| `../src/models_classic.py`, `../src/cnn.py`, `../src/features.py` | código de los modelos |

Modelos entrenados: `../data/processed/` (no versionado); el modelo provisional de la app está en `../06_despliegue/modelo/`.

**Pendiente:** repetir con tres clases cuando exista *Regular* (`../03_preparacion_datos/construir_tres_clases.py`), evaluar en fotos propias y en frutas de *Mixed*, y usar el test una sola vez.
**Alimenta rúbrica:** 4, 5, 6.
Estado: 🟡 ML tradicional y CNN ejecutados en la tarea binaria; faltan *Regular* y la evaluación en fotos propias.
