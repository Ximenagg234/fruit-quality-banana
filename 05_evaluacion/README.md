# Fase 5 — Evaluación

La evaluación de los modelos (tablas train / val / test, matrices de confusión, errores, fotos propias) está en `../04_modelado/`
(`resultados_tres_clases*.csv`, `tabla_train_val_test.csv`, `ANALISIS_ERRORES.md`). Esta carpeta guarda el **costo de cómputo**
(dimensión ambiental): `python 05_evaluacion/medir_costo.py`.

## Qué se midió y qué se estimó
| Dato | Cómo | Archivo |
|---|---|---|
| Hardware | leído del equipo: Intel Core i7-1255U (10 núcleos, 12 hilos), 39.7 GB de memoria, sin GPU (TensorFlow en Windows usa CPU) | `hardware.json` |
| Tiempo de entrenamiento | **medido** (reloj) por modelo; la ronda 1 (descartada al ampliar las etiquetas) viene de los registros de la ejecución | `costo_entrenamiento.csv` |
| Latencia de inferencia por imagen | **medida** (mediana y percentil 95) en 120 fotos propias, por modelo y para el flujo completo de la app | `costo_inferencia.csv` |
| Tamaño y parámetros de los modelos | **medido** | `tamano_modelos.csv` |
| Energía | **estimada**, no medida: potencia x tiempo con el rango del fabricante (base 15 W, turbo máximo 55 W, Intel ARK i7-1255U). No hubo vatímetro | columnas `kWh_estimado_*` y `Wh_por_1000_*` |
| Emisiones de CO₂ | **no calculadas**: el factor del sistema eléctrico colombiano (UPME) no se pudo verificar `[VERIFICAR]` | — |

## Resultados
- **Entrenamiento de los modelos finales (tres clases, ronda 2):** 2 131 s de procesos (0.59 h); el Random Forest, el SVM y la regresión logística tardaron 241, 261 y 94 s, y las CNN T6 y T7, 827 y 661 s. Los tres entrenamientos corrieron **a la vez** en el mismo equipo, así que cada tiempo está inflado frente a correr uno solo.
- **Todo el proyecto** (dos clases, dos rondas de tres clases y los intentos fallidos): 25 946 s de procesos = **7.2 h**, que a 15 – 55 W serían **0.11 a 0.40 kWh** (estimación; la suma cuenta por separado procesos que corrieron en paralelo, por lo que el tiempo de reloj real fue menor).
- **Inferencia (por foto, en CPU):** abrir la foto 10 ms; regresión logística 10 ms, SVM 11 ms y Random Forest 22 ms (con las características de color); **CNN T6 73 ms** (percentil 95: 86 ms); medir el tamaño 43 ms; **flujo completo de la app 116 ms** (percentil 95: 128 ms). A 15 – 55 W, mil predicciones de la app consumen entre 0.5 y 1.8 Wh (estimación).
- **Tamaño de los modelos:** regresión logística 0.01 MB, SVM 2.4 MB, Random Forest 13 MB y la CNN T6 29.5 MB (2 453 059 parámetros).

Límites: las latencias son de un portátil sin GPU con otras aplicaciones abiertas; la energía es una cota calculada, no una lectura de potencia.
Estado: ✅ hardware y tiempos medidos; energía estimada con su supuesto declarado.
