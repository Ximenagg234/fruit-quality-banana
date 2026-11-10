# Matriz de trazabilidad: rúbrica → evidencia

Regla del equipo: **un criterio solo se marca ✅ cuando la evidencia existe, está enlazada y cumple TODO lo que exige el nivel "Outstanding" (4.5–5)**. Si falta una sola pieza, queda 🟡.

Estados: ⬜ sin empezar · 🟡 en progreso / incompleto · ✅ completo y verificado.

> La rúbrica no trae pesos por criterio (`Rubrica_proyecto.xlsx`). Verificar en INTU si el profesor los publicó; mientras tanto se trata a los 8 criterios como igual de importantes.

## Los 8 criterios

### 1. Dilemas éticos (PI1)
**Outstanding exige:** (a) todas las situaciones éticas relevantes (privacidad, consentimiento, sesgos, uso responsable) explicadas con profundidad; (b) cada decisión relacionada **explícitamente** con principios del código de ética; (c) tabla o sección completa con **decisiones éticas y medidas adoptadas**.

| Pieza | Evidencia | Estado |
|---|---|---|
| Tabla dilema → principio del código de ética → decisión/medida | `docs/etica_tabla.md` → informe, sección Ética | ⬜ |
| Las medidas están realmente implementadas (no solo declaradas): p. ej. fotos sin rostros, sesgo del dataset medido en EDA | enlaces a notebooks/código | ⬜ |
| Código de ética citado con referencia IEEE (IEEE/ACM) | `report/refs.bib` | ⬜ |

### 2. Impactos social, económico, ambiental y global (PI2)
**Outstanding exige:** (a) las **4 dimensiones** con argumentos técnicos y éticos; (b) matriz o esquema claro; (c) estrategias de mitigación **concretas y justificadas**.

| Pieza | Evidencia | Estado |
|---|---|---|
| Matriz 4 dimensiones × (impacto positivo, riesgo, mitigación) | `docs/impactos_matriz.md` → informe | ⬜ |
| Dimensión ambiental con **dato medido** (energía/tiempo de entrenamiento e inferencia, hardware) | `05_evaluacion/` | ⬜ |
| Análisis inicial (en avance) **y** análisis final (lineamientos §6 piden ambos) | informe | ⬜ |

### 3. Formulación matemática (PI3)
**Outstanding exige:** formulación clara, completa y correcta, con variables y ecuaciones explícitas; demostrar **por qué es un problema complejo de ingeniería**; conectarla con los objetivos.

| Pieza | Evidencia | Estado |
|---|---|---|
| Variables, espacios, función objetivo, restricciones | `docs/formulacion_matematica.md` (borrador listo, ver §1–§3) | 🟡 |
| Argumento de complejidad | `docs/formulacion_matematica.md` §1 | 🟡 |
| Pseudocódigo/esquema formal del pipeline | informe | ⬜ |

### 4. Estrategia de solución y justificación técnica
**Outstanding exige:** selección del modelo justificada con precisión, **alternativas consideradas y por qué se descartaron**, coherencia con la formulación.

| Pieza | Evidencia | Estado |
|---|---|---|
| Tabla de alternativas (modelo, ventaja, límite, decisión) | `docs/estrategia_solucion.md` | ⬜ |
| La elección se apoya en resultados (baseline vs. modelos) | `05_evaluacion/` | ⬜ |

### 5. Fundamentos matemáticos aplicados
**Outstanding exige:** todas las operaciones (normalización, métricas, función de pérdida, distancias, etc.) **justificadas teóricamente y conectadas con decisiones del proyecto**.

| Pieza | Evidencia | Estado |
|---|---|---|
| Normalización de píxeles y estandarización de características (solo con estadísticos de train) | `docs/formulacion_matematica.md` §4 | 🟡 |
| Funciones de pérdida (entropía cruzada ponderada, hinge, Gini) | §5 | 🟡 |
| Convolución, pooling, tamaño de salida y número de parámetros de la CNN, **calculados y verificados contra `model.summary()`** | §6 + notebook | 🟡 |
| Métricas (F1 macro, kappa ponderado, error estándar binomial) | §8 | 🟡 |
| Medición de tamaño (PCA de la máscara) | §3 | 🟡 |

### 6. Análisis e interpretación de resultados
**Outstanding exige:** métricas, gráficos y comparaciones claras; interpretar precisión, error, generalización y **sobreajuste**; insights conectados con el problema.

| Pieza | Evidencia | Estado |
|---|---|---|
| Tabla comparativa de modelos en **train / val / test** | `05_evaluacion/` | ⬜ |
| Curvas de aprendizaje y de entrenamiento/validación de la CNN | `05_evaluacion/` | ⬜ |
| Matriz de confusión por modelo y análisis de errores con imágenes | `05_evaluacion/` | ⬜ |
| Generalización a **fotos propias** (dominio distinto al dataset) | `05_evaluacion/` | ⬜ |
| Incertidumbre: error estándar con n pequeño | `05_evaluacion/` | ⬜ |
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

## Auditoría final (antes de entregar)
1. Recorrer los 8 criterios y marcar contra el texto literal del nivel 4.
2. Revisión cruzada: una persona del equipo califica el informe con la rúbrica **sin haberlo escrito**.
3. Revisar `docs/REQUISITOS_LINEAMIENTOS.md` completo.
