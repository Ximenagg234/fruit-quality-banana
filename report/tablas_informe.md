# Tablas del informe (generadas; no editar a mano)

Se generan con `python report/generar_tablas.py` a partir de los CSV de `04_modelado/`, `05_evaluacion/` y `data/mixed_labels/`. Cada tabla dice de qué archivo sale. *Regular* tiene pocos casos (ver los n): las diferencias entre modelos en esa clase no son concluyentes.

## Tabla 1. Validación (Kaggle y *Mixed*), tres clases

Fuente: `04_modelado/resultados_tres_clases.csv`. n = 2758 imágenes, de ellas 35 *Regular*. F1 macro y kappa ponderado (orden Buena < Regular < Mala); EE = error estándar binomial de la exactitud.

| Modelo | Exactitud ± EE | F1 macro | Kappa ponderado | Recall *Regular* | F1 *Regular* |
|---|---|---|---|---|---|
| Línea base (clase más frecuente) | 0.627 ± 0.009 | 0.257 | 0.000 | 0.00 | 0.000 |
| Regresión logística | 0.844 ± 0.007 | 0.625 | 0.758 | 0.40 | 0.146 |
| SVM (RBF) | 0.904 ± 0.006 | 0.693 | 0.833 | 0.34 | 0.258 |
| Random Forest | 0.919 ± 0.005 | 0.628 | 0.857 | 0.03 | 0.035 |
| Random Forest solo borde (control) | 0.907 ± 0.006 | 0.630 | 0.829 | 0.06 | 0.069 |
| CNN T6 (Flatten) | 0.909 ± 0.005 | 0.725 | 0.851 | 0.60 | 0.339 |
| CNN T7 (GAP) | 0.874 ± 0.006 | 0.623 | 0.763 | 0.11 | 0.116 |

## Tabla 2. Prueba (Kaggle y *Mixed*), usada una sola vez

Fuente: `04_modelado/resultados_tres_clases_test.csv`. n = 2746 imágenes, de ellas 27 *Regular*.

| Modelo | Exactitud ± EE | F1 macro | Kappa ponderado | Recall *Regular* | F1 *Regular* |
|---|---|---|---|---|---|
| Línea base (clase más frecuente) | 0.626 ± 0.009 | 0.257 | 0.000 | 0.00 | 0.000 |
| Regresión logística | 0.871 ± 0.006 | 0.659 | 0.801 | 0.59 | 0.196 |
| SVM (RBF) | 0.938 ± 0.005 | 0.774 | 0.886 | 0.48 | 0.441 |
| Random Forest | 0.941 ± 0.004 | 0.765 | 0.891 | 0.41 | 0.407 |
| Random Forest solo borde (control) | 0.911 ± 0.005 | 0.685 | 0.828 | 0.19 | 0.233 |
| CNN T6 (Flatten) | 0.913 ± 0.005 | 0.714 | 0.876 | 0.81 | 0.282 |
| CNN T7 (GAP) | 0.905 ± 0.006 | 0.736 | 0.824 | 0.63 | 0.391 |

## Tabla 3. Fotos propias (prueba final de uso único)

Fuente: `04_modelado/resultados_tres_clases_test.csv`, conjunto `own_test`. n = 69 fotos de 34 frutas: 49 *Buena*, 14 *Regular*, 6 *Mala*. La línea base de decir siempre *Buena* tiene exactitud 0.710.

| Modelo | Exactitud ± EE | F1 macro | Recall *Buena* | Recall *Regular* | Recall *Mala* |
|---|---|---|---|---|---|
| Línea base (clase más frecuente) | 0.710 ± 0.055 | 0.277 | 1.00 | 0.00 | 0.00 |
| Regresión logística | 0.565 ± 0.060 | 0.305 | 0.76 | 0.00 | 0.33 |
| SVM (RBF) | 0.638 ± 0.058 | 0.363 | 0.84 | 0.00 | 0.50 |
| Random Forest | 0.362 ± 0.058 | 0.243 | 0.41 | 0.00 | 0.83 |
| Random Forest solo borde (control) | 0.638 ± 0.058 | 0.346 | 0.86 | 0.00 | 0.33 |
| CNN T6 (Flatten) | 0.681 ± 0.056 | 0.439 | 0.84 | 0.00 | 1.00 |
| CNN T7 (GAP) | 0.652 ± 0.057 | 0.419 | 0.80 | 0.00 | 1.00 |

## Tabla 4. Fotos propias con y sin mano, y por especie (CNN T6)

Fuente: `04_modelado/resultados_tres_clases_test.csv`. Las 6 *Mala* están todas en mano; sin mano solo hay 14 fotos.

| Subconjunto | n | Exactitud ± EE | F1 macro |
|---|---|---|---|
| En mano | 55 | 0.764 ± 0.057 | 0.47 |
| Sin mano | 14 | 0.357 ± 0.128 | 0.28 |
| Manzana | 11 | 0.455 ± 0.150 | 0.33 |
| Banano | 13 | 0.615 ± 0.135 | 0.46 |
| Guayaba | 19 | 0.579 ± 0.113 | 0.40 |
| Lima | 9 | 1.000 ± 0.000 | 1.00 |
| Naranja | 17 | 0.824 ± 0.092 | 0.53 |

## Tabla 5. Entrenamiento, validación, prueba y fotos propias (F1 macro) y sobreajuste

Fuente: `04_modelado/tabla_train_val_test.csv`. El entrenamiento está submuestreado (<= 1 100 imágenes por especie y clase).

| Modelo | Entrenamiento | Validación | Prueba | Fotos propias | Brecha entren. − val. |
|---|---|---|---|---|---|
| Línea base (clase más frecuente) | 0.227 | 0.257 | 0.257 | 0.277 | -0.03 |
| Regresión logística | 0.748 | 0.625 | 0.659 | 0.305 | 0.12 |
| SVM (RBF) | 0.929 | 0.693 | 0.774 | 0.363 | 0.24 |
| Random Forest | 0.927 | 0.628 | 0.765 | 0.243 | 0.30 |
| Random Forest solo borde (control) | 1.000 | 0.630 | 0.685 | 0.346 | 0.37 |
| CNN T6 (Flatten) | 0.765 | 0.725 | 0.714 | 0.439 | 0.04 |
| CNN T7 (GAP) | 0.729 | 0.623 | 0.736 | 0.419 | 0.11 |

## Tabla 6. Acuerdo entre anotadores (recortes de *Mixed*)

Fuente: `data/mixed_labels/acuerdo_anotadores.csv`. Dos personas por ronda, etiquetando por separado; los desacuerdos se resolvieron en grupo.

| Ronda | Recortes | Sin descartar | Concordancia exacta | Kappa ponderado | Desacuerdos |
|---|---|---|---|---|---|
| 1 | 360 | 323 | 62.8% | 0.607 | 134 |
| 2 | 400 | 376 | 60.2% | 0.606 | 159 |

## Tabla 7. Costo de cómputo

Fuente: `05_evaluacion/`. Equipo: 12th Gen Intel(R) Core(TM) i7-1255U, 10 núcleos, 39.7 GB, sin GPU. La energía es **estimada** (potencia del fabricante, 15 a 55 W, por el tiempo), no medida.

| Concepto | Valor |
|---|---|
| Entrenamiento de los modelos finales (procesos) | 2131 s (0.59 h) |
| Entrenamiento de todo el proyecto (procesos) | 25946 s (7.2 h) |
| Energía estimada de todo el proyecto | 0.11 a 0.40 kWh |
| abrir y decodificar la foto (1 024 px) | 10 ms |
| logreg: características de color + predicción | 10 ms (p95 15 ms) |
| svm_rbf: características de color + predicción | 10 ms (p95 14 ms) |
| random_forest: características de color + predicción | 22 ms (p95 28 ms) |
| CNN T6: reducir a 96 px + predicción | 73 ms (p95 86 ms) |
| tamaño: segmentar y medir la longitud normalizada | 43 ms (p95 52 ms) |
| flujo completo de la app (CNN T6 + tamaño) | 116 ms (p95 128 ms) |

## Tabla 8. Exploración: transfer learning (solo validación)

Fuente: `04_modelado/resultados_transfer_validacion.csv`. Se hizo **después** de usar la prueba, así que solo se mide en validación (35 *Regular*; la diferencia no es concluyente). No cambia el modelo de la app.

| Modelo | F1 macro | Recall *Regular* | F1 *Regular* | F1 macro en *Mixed* |
|---|---|---|---|---|
| CNN propia (la de la app) | 0.725 | 0.60 | 0.339 | 0.350 |
| MobileNetV2 congelada | 0.760 | 0.49 | 0.391 | 0.491 |
| MobileNetV2 con ajuste fino | 0.717 | 0.26 | 0.257 | 0.422 |
