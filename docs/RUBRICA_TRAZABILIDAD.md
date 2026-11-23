# Matriz de trazabilidad: rúbrica → evidencia

Regla del equipo: **un criterio solo se marca ✅ cuando la evidencia existe, está enlazada y cumple TODO lo que exige el nivel "Outstanding" (4.5–5)**. Si falta una sola pieza, queda 🟡.

Estados: ⬜ sin empezar · 🟡 en progreso / incompleto · ✅ completo y verificado.

> La rúbrica no trae pesos por criterio (`Rubrica_proyecto.xlsx`). Verificar en INTU si el profesor los publicó; mientras tanto se trata a los 8 criterios como igual de importantes.

## Los 8 criterios

### 1. Dilemas éticos (PI1)
**Outstanding exige:** (a) todas las situaciones éticas relevantes (privacidad, consentimiento, sesgos, uso responsable) explicadas con profundidad; (b) cada decisión relacionada **explícitamente** con principios del código de ética; (c) tabla o sección completa con **decisiones éticas y medidas adoptadas**.

| Pieza | Evidencia | Estado |
|---|---|---|
| Tabla dilema → principio del código de ética → decisión/medida | `docs/etica_tabla.md` (borrador v1.1 multifruta, 8 filas con estado por medida; E8 pendiente de decisión de la autora) → informe, sección Ética | 🟡 |
| Las medidas están realmente implementadas (no solo declaradas): p. ej. fotos sin rostros, sesgo del dataset medido en EDA | Sesgo de fondo medido: `02_comprension_datos/eda.ipynb` §7 (F1 macro del borde solo: 0.76–0.96). Faltan `prepare_photos.py`, visto bueno de las personas que aparecen en las fotos y la app que no guarda imágenes | 🟡 |
| Código de ética citado con referencia IEEE (IEEE/ACM) | `report/refs.bib` | ⬜ |

### 2. Impactos social, económico, ambiental y global (PI2)
**Outstanding exige:** (a) las **4 dimensiones** con argumentos técnicos y éticos; (b) matriz o esquema claro; (c) estrategias de mitigación **concretas y justificadas**.

| Pieza | Evidencia | Estado |
|---|---|---|
| Matriz 4 dimensiones × (impacto positivo, riesgo, mitigación) | `docs/impactos_matriz.md` (borrador v2 multifruta, análisis inicial; faltan datos [MEDIR]) → informe | 🟡 |
| Dimensión ambiental con **dato medido** (energía/tiempo de entrenamiento e inferencia, hardware) | `05_evaluacion/` | ⬜ |
| Análisis inicial (en avance) **y** análisis final (lineamientos §6 piden ambos) | informe | ⬜ |

### 3. Formulación matemática (PI3)
**Outstanding exige:** formulación clara, completa y correcta, con variables y ecuaciones explícitas; demostrar **por qué es un problema complejo de ingeniería**; conectarla con los objetivos.

| Pieza | Evidencia | Estado |
|---|---|---|
| Variables, espacios, función objetivo, restricciones | `docs/formulacion_matematica.md` (borrador v2 multifruta: especie, tamaño por especie, grupos; ver §0–§3 y §7) | 🟡 |
| Argumento de complejidad | `docs/formulacion_matematica.md` §1 | 🟡 |
| Pseudocódigo/esquema formal del pipeline | informe | ⬜ |

### 4. Estrategia de solución y justificación técnica
**Outstanding exige:** selección del modelo justificada con precisión, **alternativas consideradas y por qué se descartaron**, coherencia con la formulación.

| Pieza | Evidencia | Estado |
|---|---|---|
| Tabla de alternativas (modelo, ventaja, límite, decisión) | `docs/estrategia_solucion.md` | ⬜ |
| La elección se apoya en resultados (baseline vs. modelos) | `04_modelado/modelos_clasicos.ipynb` y `04_modelado/resultados_modelos_clasicos.csv` (Buena vs. Mala en Kaggle, solo train y val; faltan *Regular*, la CNN y la evaluación en fotos propias) | 🟡 |

### 5. Fundamentos matemáticos aplicados
**Outstanding exige:** todas las operaciones (normalización, métricas, función de pérdida, distancias, etc.) **justificadas teóricamente y conectadas con decisiones del proyecto**.

| Pieza | Evidencia | Estado |
|---|---|---|
| Normalización de píxeles y estandarización de características (solo con estadísticos de train) | `docs/formulacion_matematica.md` §4 | 🟡 |
| Funciones de pérdida (entropía cruzada ponderada, hinge, Gini) | §5 | 🟡 |
| Convolución, pooling, tamaño de salida y número de parámetros de la CNN, **calculados y verificados contra Keras** | `docs/formulacion_matematica.md` §6; `src/cnn.py` y `tests/test_cnn.py` (la fórmula coincide con `count_params()`: 4 288 067 con `Flatten` y 110 147 con GAP en el ejemplo de 128×128, y casos de 96 px con 6 clases) | ✅ |
| Métricas (F1 macro, kappa ponderado, error estándar binomial) | §8 | 🟡 |
| Medición de tamaño (PCA de la máscara) | §3 | 🟡 |

### 6. Análisis e interpretación de resultados
**Outstanding exige:** métricas, gráficos y comparaciones claras; interpretar precisión, error, generalización y **sobreajuste**; insights conectados con el problema.

| Pieza | Evidencia | Estado |
|---|---|---|
| Tabla comparativa de modelos en **train / val / test** | train y val en `04_modelado/resultados_modelos_clasicos.csv` (F1 train, CV agrupada y val, brecha); el **test aún no se usa** (se usa una sola vez al final) | 🟡 |
| Curvas de aprendizaje y de entrenamiento/validación de la CNN | `05_evaluacion/` | ⬜ |
| Matriz de confusión por modelo y análisis de errores con imágenes | matriz del mejor modelo en val en `04_modelado/figuras/modelos_clasicos.pdf`; falta el análisis de errores con imágenes | 🟡 |
| Generalización a **fotos propias** (dominio distinto al dataset) | `05_evaluacion/` | ⬜ |
| Incertidumbre: error estándar con n pequeño | error estándar binomial en la tabla de val (`models_classic.evaluate`); falta aplicarlo a fotos propias | 🟡 |
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
| Diagrama CRISP-DM **personalizado** | `docs/diagrama_crispdm.svg` | ⬜ |
| Referencias IEEE; todo lo de terceros citado | `report/refs.bib` | ⬜ |
| Código con docstrings y comentarios, repo ordenado por fases | todo el repo | ⬜ |

## Requisitos duros del enunciado (no negociables)
Estado a la fecha de esta versión; solo se marca ✅ con evidencia verificable.

| Requisito (lineamientos) | Evidencia | Estado |
|---|---|---|
| Alcance multifruta confirmado por el profesor | `docs/REQUISITOS_LINEAMIENTOS.md`, `docs/definicion_clases.md` | 🟡 (diseño y datos listos; modelos pendientes) |
| Dataset de Kaggle usado y citado; licencia verificada | `data/README.md` (licencia CC0 verificada en la ficha; falta la referencia IEEE con la fecha exacta en `report/refs.bib`) | 🟡 |
| `Mixed`: segmentar cada fruta individual | `src/segmentation.py` (probada con formas sintéticas y revisada a ojo en fotos sobre superficie; **no mide bien fotos con mano**, que se detectan por contacto con el borde; `Mixed` trae varias frutas por foto y fondos de color: falta separar frutas individuales y validar) | 🟡 |
| 30–50 imágenes propias anotadas (calidad y tamaño), por varias especies | 91 fotos de banano recibidas, sin `object_id` ni etiquetas; fotos de las otras 5 especies pendientes | 🟡 |
| Mecanismo para compartir anotaciones con la clase | esquema en `data/README.md` §2–3 | 🟡 |
| ≥ 2 modelos de ML tradicional con hiperparámetros por validación cruzada | `04_modelado/modelos_clasicos.ipynb`: regresión logística, SVM RBF y Random Forest con `GridSearchCV` y `StratifiedGroupKFold` (tarea binaria en Kaggle; se repite con *Regular*) | 🟡 |
| ≥ 1 modelo de deep learning (CNN desde cero) | `src/cnn.py` (arquitectura y conteo de parámetros verificados; sin entrenar) | 🟡 |
| Comparación con línea base | línea base mayoritaria (F1 0.388) y línea base "solo borde" en el mismo notebook | 🟡 |
| CRISP-DM con evidencia: fase 2 (EDA ejecutado) | `02_comprension_datos/eda.ipynb` ejecutado con datos reales; falta medir la nitidez como parte de "calidad de imagen" | 🟡 |
| CRISP-DM: fases 1, 3, 4, 5 y 6 | — | ⬜ |
| **App Streamlit funcional: cargar imagen o capturar con cámara y mostrar calidad y tamaño** | `06_despliegue/app.py` + `src/inference.py` + `tests/test_app.py`. Carga de imagen verificada en un navegador con fotos reales; el widget de cámara solo se monta al elegirla, pero **la captura real con cámara no se pudo verificar** (el navegador del entorno bloquea la cámara). Modelo provisional Buena/Mala; tamaño como longitud normalizada, sin clases por especie | 🟡 |
| Diagrama de flujo CRISP-DM personalizado | — | ⬜ |
| Informe ≤ 7 páginas IEEE y video ≤ 10 min | — | ⬜ |
| Análisis ético y análisis inicial y final de impactos | `docs/etica_tabla.md`, `docs/impactos_matriz.md` (iniciales) | 🟡 |

## Auditoría final (antes de entregar)
1. Recorrer los 8 criterios y marcar contra el texto literal del nivel 4.
2. Revisión cruzada: una persona del equipo califica el informe con la rúbrica **sin haberlo escrito**.
3. Revisar `docs/REQUISITOS_LINEAMIENTOS.md` completo.
