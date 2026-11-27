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
| `analisis_modelos.py` | tabla train / val / test (`tabla_train_val_test.csv`), matrices de confusión y ejemplos de error de los modelos de tres clases; el texto está en `ANALISIS_ERRORES.md` |
| `experimento_transfer.py` | exploración posterior a la prueba: MobileNetV2 preentrenada, solo en validación (`resultados_transfer_validacion.csv`); no cambia la app |
| `../src/models_classic.py`, `../src/cnn.py`, `../src/features.py` | código de los modelos |

Modelos entrenados: `../data/processed/` (no versionado); el modelo provisional de la app está en `../06_despliegue/modelo/`.

**Tres clases (`entrenar_tres_clases.py`):** usa `data/processed/tres_clases.pkl` (de `../03_preparacion_datos/construir_tres_clases.py --propias-solo-prueba`). Se entrena y valida con Kaggle y *Mixed* (746 recortes etiquetados); las 69 fotos propias son solo la prueba final (`own_test`). Hay cuatro fases: `datos`, `clasicos`, `cnn` y `test`; la prueba se corre **una sola vez** y el script se niega a repetirla. Resultados: `resultados_tres_clases.csv` (validación) y `resultados_tres_clases_test.csv` (prueba). *Regular* tiene 137 imágenes de entrenamiento (todas de *Mixed*): se usan pesos de clase y, en la CNN, pesos suavizados (raíz), porque con el peso completo (~60) la red colapsó y predecía *Regular* para casi todo.
**Alimenta rúbrica:** 4, 5, 6.
Estado: ✅ ML tradicional y CNN ejecutados en dos y en tres clases, con la prueba usada una vez y el análisis de errores hecho. Pendiente: dimensión ambiental (tiempo y energía) y la app con tres clases.
