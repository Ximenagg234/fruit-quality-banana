# Matriz de trazabilidad: rúbrica → evidencia

Regla del equipo: **un criterio solo se marca ✅ cuando la evidencia existe, está enlazada y cumple TODO lo que exige el nivel "Outstanding" (4.5–5)**. Si falta una sola pieza, queda 🟡.

Estados: ⬜ sin empezar · 🟡 en progreso / incompleto · ✅ completo y verificado.

> La rúbrica no trae pesos por criterio (`Rubrica_proyecto.xlsx`). Verificar en INTU si el profesor los publicó; mientras tanto se trata a los 8 criterios como igual de importantes.

## Los 8 criterios

### 1. Dilemas éticos (PI1)
**Outstanding exige:** (a) todas las situaciones éticas relevantes (privacidad, consentimiento, sesgos, uso responsable) explicadas con profundidad; (b) cada decisión relacionada **explícitamente** con principios del código de ética; (c) tabla o sección completa con **decisiones éticas y medidas adoptadas**.

| Pieza | Evidencia | Estado |
|---|---|---|
| Tabla dilema → principio del código de ética → decisión/medida | `docs/etica_tabla.md` (v2: 8 filas con estado y evidencia; E1 a E7 ✅, E3 🟡 por la referencia IEEE y E8 pendiente de decisión de la autora) → informe, sección Ética | ✅ |
| Las medidas están realmente implementadas (no solo declaradas): p. ej. fotos sin rostros, sesgo del dataset medido en EDA | E1 a E7 con evidencia enlazada: 0 EXIF y 0 GPS en las 123 fotos, sin rostros y manos de integrantes con su aceptación (E1, E2); sesgo de fondo medido y controlado en cada familia de modelos (E4, `eda.ipynb` §7 y `resultados_tres_clases.csv`); prueba usada una sola vez con error estándar (E5); app con confianza y alcance declarado (E6); app que no guarda imágenes, verificada con una prueba (E7) | ✅ |
| Código de ética citado con referencia IEEE (IEEE/ACM) | `report/refs.bib` aún **no** trae los códigos de ética (se agregan cuando se verifiquen la versión 2018 de la ACM y la política 7.8 de IEEE) | ⬜ |

### 2. Impactos social, económico, ambiental y global (PI2)
**Outstanding exige:** (a) las **4 dimensiones** con argumentos técnicos y éticos; (b) matriz o esquema claro; (c) estrategias de mitigación **concretas y justificadas**.

| Pieza | Evidencia | Estado |
|---|---|---|
| Matriz 4 dimensiones × (impacto positivo, riesgo, mitigación) | `docs/impactos_matriz.md` (v3: contexto referenciado, riesgos, mitigaciones, datos medidos y análisis final) → informe | ✅ |
| Dimensión ambiental con **dato medido** (energía/tiempo de entrenamiento e inferencia, hardware) | `05_evaluacion/medir_costo.py` y `README.md`: hardware (i7-1255U, 10 núcleos, 39.7 GB, sin GPU), tiempos de entrenamiento por modelo medidos (modelos finales 0.59 h; todo el proyecto 7.2 h de procesos), latencia de inferencia medida (app completa 116 ms por foto) y tamaño de los modelos. **La energía es una estimación** (potencia del fabricante 15–55 W × tiempo: 0.11 a 0.40 kWh en todo el proyecto), no una lectura con vatímetro; el factor de emisión de la UPME no se verificó y no se usó | ✅ |
| Análisis inicial (en avance) **y** análisis final (lineamientos §6 piden ambos) | informe | ⬜ |

### 3. Formulación matemática (PI3)
**Outstanding exige:** formulación clara, completa y correcta, con variables y ecuaciones explícitas; demostrar **por qué es un problema complejo de ingeniería**; conectarla con los objetivos.

| Pieza | Evidencia | Estado |
|---|---|---|
| Variables, espacios, función objetivo, restricciones | `docs/formulacion_matematica.md` (v3, decisiones cerradas: especie, tamaño por especie, grupos; §0–§3 y §7) | ✅ |
| Argumento de complejidad | `docs/formulacion_matematica.md` §1 (cuatro fuentes de dificultad con datos medidos: atajo del fondo, fuga por dependencia, *Regular* escasa y cambio de dominio) | ✅ |
| Pseudocódigo/esquema formal del pipeline | informe | ⬜ |

### 4. Estrategia de solución y justificación técnica
**Outstanding exige:** selección del modelo justificada con precisión, **alternativas consideradas y por qué se descartaron**, coherencia con la formulación.

| Pieza | Evidencia | Estado |
|---|---|---|
| Tabla de alternativas (modelo, ventaja, límite, decisión) | `docs/estrategia_solucion.md` (v3: 10 decisiones con alternativas, límites, evidencia medida y trabajo futuro declarado; la decisión 10 cubre tres clases, el desbalance, la selección del modelo y la prueba); incluye transfer learning con MobileNetV2 probado como exploración posterior a la prueba, solo en validación (04_modelado/experimento_transfer.py) | ✅ |
| La elección se apoya en resultados (baseline vs. modelos) | `04_modelado/resultados_modelos_clasicos.csv` (dos clases) y `resultados_tres_clases.csv` / `resultados_tres_clases_test.csv` (tres clases): cada modelo frente a la línea base mayoritaria y al control solo borde; el modelo de la app (CNN T6) se eligió con la validación, antes de ver la prueba | ✅ |

### 5. Fundamentos matemáticos aplicados
**Outstanding exige:** todas las operaciones (normalización, métricas, función de pérdida, distancias, etc.) **justificadas teóricamente y conectadas con decisiones del proyecto**.

| Pieza | Evidencia | Estado |
|---|---|---|
| Normalización de píxeles y estandarización de características (solo con estadísticos de train) | `docs/formulacion_matematica.md` §4; la estandarización va dentro del `Pipeline` y se ajusta solo con entrenamiento (`src/models_classic.py`) | ✅ |
| Funciones de pérdida (entropía cruzada ponderada, hinge, Gini) | `docs/formulacion_matematica.md` §5; entropía cruzada ponderada en la CNN (`pesos_de_clase` en `04_modelado/entrenar_cnn.py`, con la variante suavizada), hinge en el SVM y Gini en el Random Forest | ✅ |
| Convolución, pooling, tamaño de salida y número de parámetros de la CNN, **calculados y verificados contra Keras** | `docs/formulacion_matematica.md` §6; `src/cnn.py` y `tests/test_cnn.py` (la fórmula coincide con `count_params()`: 4 288 067 con `Flatten` y 110 147 con GAP en el ejemplo de 128×128, y casos de 96 px con 6 clases) | ✅ |
| Métricas (F1 macro, kappa ponderado, error estándar binomial) | `docs/formulacion_matematica.md` §8; implementadas en `src/models_classic.py` (`evaluate`, con el kappa en el orden Buena < Regular < Mala) y probadas en `tests/test_models_classic.py` | ✅ |
| Medición de tamaño (PCA de la máscara) | `docs/formulacion_matematica.md` §3; `src/segmentation.py` y `tests/test_segmentation.py` (error de la longitud ≤ 3 % en formas sintéticas); medida real en cm de 69 fotos y validación de ℓ (`05_evaluacion/validar_tamano.py`): correlación de Spearman 0.26 en todas las fotos y de −0.66 a +0.54 por especie, así que ℓ es una medida relativa, no el tamaño real (se declara) | ✅ |

### 6. Análisis e interpretación de resultados
**Outstanding exige:** métricas, gráficos y comparaciones claras; interpretar precisión, error, generalización y **sobreajuste**; insights conectados con el problema.

| Pieza | Evidencia | Estado |
|---|---|---|
| Tabla comparativa de modelos en **train / val / test** | tres clases: validación en `04_modelado/resultados_tres_clases.csv` y prueba, usada **una sola vez**, en `resultados_tres_clases_test.csv` (Kaggle, *Mixed* y las 69 fotos propias, con EE); binaria en `resultados_modelos_clasicos.csv` y `resultados_cnn_calidad.csv`. `04_modelado/tabla_train_val_test.csv` reúne F1 y exactitud en entrenamiento, validación, prueba y fotos propias, con la brecha entrenamiento − validación | ✅ |
| Curvas de aprendizaje y de entrenamiento/validación de la CNN | `04_modelado/cnn_experimentos.ipynb` y `figuras/cnn_curvas_aprendizaje.pdf` (10 experimentos en Buena vs. Mala sobre Kaggle, con parada temprana) y `figuras/cnn_curvas_tres_clases.pdf` (validación por época de las CNN de tres clases, con los dos intentos que colapsaron como contraste) | ✅ |
| Matriz de confusión por modelo y análisis de errores con imágenes | `04_modelado/figuras/matrices_confusion_tres_clases.pdf` (CNN T6, SVM y Random Forest en validación, prueba y fotos propias), `figuras/errores_cnn_T6.pdf` (ejemplos de error con imágenes de Kaggle y *Mixed*) y `04_modelado/ANALISIS_ERRORES.md`; las fotos propias no se muestran por falta de permisos de las personas cuyas manos aparecen | ✅ |
| Generalización a **fotos propias** (dominio distinto al dataset) | `resultados_tres_clases_test.csv`, conjunto `own_test` (69 fotos, 34 frutas) con corte por mano y por especie: F1 macro de 0.24 a 0.44, ningún modelo detecta *Regular* (0 de 14) y ninguno supera la exactitud de la línea base mayoritaria (0.71); la mejor es la CNN T6 (0.44). La generalización es pobre y se declara | ✅ |
| Incertidumbre: error estándar con n pequeño | error estándar binomial en todas las tablas de validación y prueba (`models_classic.evaluate`), incluidas las fotos propias; en *Regular* (n = 35 en validación, 27 en prueba, 14 en fotos propias) las diferencias entre modelos no son concluyentes | ✅ |
| Comparación con la literatura | informe + refs | ⬜ |

### 7. Conclusiones técnicas y coherencia
**Outstanding exige:** conclusiones alineadas con formulación, metodología y resultados; mejoras con sustento técnico.

| Pieza | Evidencia | Estado |
|---|---|---|
| Cada conclusión cita la ecuación/figura/tabla que la respalda | informe | ⬜ |
| Trabajo futuro derivado de los errores observados | informe | ⬜ |

### 8. Calidad del informe
**Outstanding exige:** escritura clara y concisa, figuras de calidad, diagramas relevantes, secciones completas según la guía, **referencias IEEE**, código bien documentado.

| Pieza | Evidencia | Estado |
|---|---|---|
| ≤ 7 páginas, formato IEEE, plantilla del profesor | `report/main.tex` (PDF compilado) | ⬜ |
| Figuras vectoriales (PDF/SVG), ejes con etiquetas y unidades | `report/figures/` | ⬜ |
| Diagrama CRISP-DM **personalizado** | `docs/diagrama_crispdm.pdf`, `.svg` y `.png` (borrador vectorial con el flujo real del proyecto y su bucle de iteración; Juan Pablo lo afina; se edita con `docs/generar_diagrama_crispdm.py`) | 🟡 |
| Referencias IEEE; todo lo de terceros citado | `report/refs.bib`: 13 entradas verificadas contra Crossref, JMLR, arXiv y el original de la FAO. **Pendientes:** las dos cifras de UNEP, el año del dataset de Kaggle, los códigos de ética y la literatura sobre calidad de frutas | 🟡 |
| Código con docstrings y comentarios, repo ordenado por fases | todo el repo: cada módulo y cada función pública de `src/`, `03_preparacion_datos/`, `04_modelado/`, `05_evaluacion/`, `06_despliegue/`, `report/` y `docs/` tiene docstring (se comprobó con un recorrido del árbol sintáctico de Python); cada carpeta de fase tiene su README | ✅ |

## Requisitos duros del enunciado (no negociables)
Estado a la fecha de esta versión; solo se marca ✅ con evidencia verificable.

| Requisito (lineamientos) | Evidencia | Estado |
|---|---|---|
| Alcance multifruta confirmado por el profesor | `docs/REQUISITOS_LINEAMIENTOS.md` (alcance confirmado por el profesor), `docs/definicion_clases.md`; modelos de las seis especies entrenados y evaluados | ✅ |
| Dataset de Kaggle usado y citado; licencia verificada | `data/README.md` (licencia CC0 verificada en la ficha; falta la referencia IEEE con la fecha exacta en `report/refs.bib`) | 🟡 |
| `Mixed`: segmentar cada fruta individual | `src/mixed.py` y `03_preparacion_datos/segmentar_mixed.py`: 1 074 fotos → 8 194 recortes (3 978 pasan los filtros de forma, color y borde); validación **visual** y con una muestra etiquetada por dos personas: de 360 recortes que pasaron los filtros, 9 (2.5 %) se descartaron por consenso y 37 (10.3 %) por al menos una persona; el banano es el caso débil (4 de 60 por consenso y 21 de 60 por al menos una persona) seguido del limón/lima (3 por consenso) y la manzana (1). Limitaciones: la muestra sale de los recortes que ya pasaron los filtros, así que no mide las frutas que la segmentación no encontró ni las 4 216 descartadas por los filtros; no hay verdad de terreno de las cajas | ✅ |
| 30–50 imágenes propias anotadas (calidad y tamaño), por varias especies | `data/own_photos/annotations.csv`: 69 fotos del 23-nov de cinco especies (apple, banana, guava, lime, orange; 34 frutas) con `object_id`, calidad final por consenso del grupo y `max_dim_cm` medido con regla; reparto 49 Buena / 14 Regular / 6 Mala (lima solo Buena, sin granada; las 6 Mala están en mano). El tamaño anotado es la medida real en cm; `size_class` no se anota a mano: se deriva de ℓ por terciles por especie con solo entrenamiento (`docs/definicion_clases.md` §3). Limitaciones declaradas: pocas *Mala* y *Regular*, lima solo *Buena*, sin granada, las 6 *Mala* están en mano, y la calidad se decidió por consenso del grupo (sin kappa; las 360 de *Mixed* sí tienen dos anotadores independientes, kappa ponderado 0.607) | ✅ |
| Mecanismo para compartir anotaciones con la clase | el repositorio es **público** y trae las anotaciones: `data/own_photos/annotations.csv` (69 fotos propias) y `data/mixed_labels/etiquetas_mixed_final.csv` (760 recortes de *Mixed*) y , con el esquema en `data/README.md` §2–3 y las herramientas `etiquetador_mixed.py` y `calcular_acuerdo.py`. El enlace del repositorio se entregó en el documento del profesor (lo confirma Ximena; no es verificable desde el repo) | ✅ |
| ≥ 2 modelos de ML tradicional con hiperparámetros por validación cruzada | `04_modelado/modelos_clasicos.ipynb`: regresión logística, SVM RBF y Random Forest con `GridSearchCV` y `StratifiedGroupKFold` (tarea binaria en Kaggle) y, con tres clases, `04_modelado/entrenar_tres_clases.py` (`GridSearchCV` agrupado, F1 macro, pesos de clase; hiperparámetros en `resultados_tres_clases.csv`) | ✅ |
| ≥ 1 modelo de deep learning (CNN desde cero) | `src/cnn.py`, `04_modelado/entrenar_cnn.py` y `cnn_experimentos.ipynb`: 10 experimentos entrenados y analizados (tabla al estilo de `02-CNN_resuelto`), con control "solo borde" y regla de 2 EE; y con tres clases las CNN T6 (Flatten) y T7 (GAP) en `entrenar_tres_clases.py`; dos intentos previos con el peso de clase completo colapsaron y se documentan en el script; T6 gana en validación (F1 macro 0.725) y se evalúa en `resultados_tres_clases_test.csv` | ✅ |
| Comparación con línea base | línea base mayoritaria (F1 0.388 en dos clases; 0.257 en tres) y control "solo borde" en dos y tres clases (`resultados_tres_clases.csv`) | ✅ |
| CRISP-DM con evidencia: fase 2 (EDA ejecutado) | `02_comprension_datos/eda.ipynb` ejecutado con datos reales; falta medir la nitidez como parte de "calidad de imagen" | 🟡 |
| CRISP-DM: fases 1, 3, 4, 5 y 6 | Fase 1: `01_comprension_negocio/negocio.md`; fase 3: `03_preparacion_datos/`; fase 4: `04_modelado/`; fase 5: `04_modelado/ANALISIS_ERRORES.md` y `05_evaluacion/`; fase 6: `06_despliegue/` (cada carpeta tiene su README con lo que contiene) | ✅ |
| **App Streamlit funcional: cargar imagen o capturar con cámara y mostrar calidad y tamaño** | `06_despliegue/app.py` + `src/inference.py` + `tests/test_app.py`. Carga de imagen verificada en un navegador con fotos reales; el widget de cámara solo se monta al elegirla, pero **la captura real con cámara no se pudo verificar** (el navegador del entorno bloquea la cámara). La app usa ahora la CNN T6 de tres clases (`06_despliegue/modelo/calidad_cnn.keras`); sus predicciones coinciden en las 69 fotos propias con las de la evaluación; avisa que *Regular* es poco fiable. Tamaño como longitud normalizada, sin clases por especie | 🟡 |
| Diagrama de flujo CRISP-DM personalizado | `docs/diagrama_crispdm.pdf`, `.svg` y `.png`, regenerado con los datos finales (`docs/generar_diagrama_crispdm.py`); Juan Pablo lo afina | 🟡 |
| Informe ≤ 7 páginas IEEE y video ≤ 10 min | — | ⬜ |
| Análisis ético y análisis inicial y final de impactos | `docs/etica_tabla.md` (E1 a E7 con evidencia; E3 y E8 pendientes) y `docs/impactos_matriz.md` (v3 con análisis inicial y final); falta llevarlos al informe | 🟡 |

## Auditoría final (antes de entregar)
1. Recorrer los 8 criterios y marcar contra el texto literal del nivel 4.
2. Revisión cruzada: una persona del equipo califica el informe con la rúbrica **sin haberlo escrito**.
3. Revisar `docs/REQUISITOS_LINEAMIENTOS.md` completo.
