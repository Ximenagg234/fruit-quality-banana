# Estrategia de solución y alternativas consideradas (rúbrica 4)

> **Estado: v3, con los resultados de tres clases.** Cada decisión se apoya en una evidencia del repositorio. Los números de Kaggle de las decisiones 1 a 5 son de la tarea **Buena vs. Mala** y **no demuestran que el modelo aprenda calidad** (ver decisión 1); los de tres clases (decisión 10) incluyen *Mixed* y las fotos propias, y son los que se reportan.

## Restricciones que guían la estrategia

| Restricción | Consecuencia |
|---|---|
| El dataset base solo trae *Good* y *Bad*; **Regular no existe** | Se construye con 760 recortes de *Mixed* etiquetados a mano por dos personas (199 *Regular*) y las fotos propias como prueba (decisión 7) |
| El fondo ya predice la clase (EDA §7) | Toda métrica se reporta junto a la línea base "solo borde"; se evalúa en fotos propias con fondo controlado |
| Los grupos de imágenes del mismo objeto no son independientes | Partición por objeto, no por imagen (decisión 2) |
| Solo CPU (i7-1255U, 10 núcleos) y plazo de pocos días | Modelos pequeños, imágenes de 96 a 128 px y submuestreo por celda |
| Pocas fotos propias (69 fotos de 34 frutas; 14 *Regular* y 6 *Mala*) | Solo prueba final de uso único; error estándar binomial; las diferencias menores que 2 EE no son concluyentes |

## Decisiones, alternativas y evidencia

### 1. ¿Cómo leer el rendimiento en Kaggle?

| Alternativa | Ventaja | Límite | Decisión |
|---|---|---|---|
| Reportar solo F1 y exactitud en val | Simple | Engaña: un modelo que solo ve el borde de la imagen separa Good de Bad con F1 macro de 0.76 a 0.96 según la especie (`02_comprension_datos/eda.ipynb` §7) | Descartada |
| Reportar siempre la línea base "solo borde" junto a cada modelo | Cuantifica cuánto es fondo | Requiere entrenar cada familia dos veces | **Elegida** |

Resultado (`04_modelado/modelos_clasicos.ipynb`, F1 macro en val): mayoritaria 0.388; regresión logística 0.891 (solo borde 0.852); SVM 0.944 (0.903); Random Forest 0.947 (0.931). En banano, el Random Forest saca 0.997 con la imagen completa y con solo el borde.

### 2. Partición sin fuga de datos

| Alternativa | Ventaja | Límite | Decisión |
|---|---|---|---|
| Partición aleatoria por imagen | Trivial | Fotos del mismo objeto en train y test infla las métricas | Descartada. **Medido** (`04_modelado/fuga_particion.ipynb`, mismo Random Forest, 5 semillas): F1 macro 0.987 con partición aleatoria frente a 0.960 agrupada (inflación ≈ 0.027, cota inferior); el 49.9 % de las imágenes de prueba tiene un casi-duplicado en entrenamiento |
| Agrupar solo por marca de tiempo del nombre | Barato | Solo el 80.3 % de los nombres trae fecha; en *Good* banana solo el 10.2 % | Insuficiente |
| Tiempo (≤ 3 s) + hash perceptual (≤ 5) + contador con hash (≤ 30), componentes conexos | Cubre todas las convenciones de nombre | Fuga residual por rotaciones o recortes que el hash no detecta | **Elegida**: 3 360 grupos, grupo máximo 182, test por celda entre 14.2 % y 15.3 % |

Evidencia de calibración (`eda.ipynb` §5): con hash ≤ 40 aparece un grupo de 4 732 imágenes (la clausura transitiva funde objetos distintos) y con tiempo de 20 s la partición se descuadra (test por celda de 8.6 % a 53.8 %).

### 3. Características para ML tradicional

| Alternativa | Ventaja | Límite | Decisión |
|---|---|---|---|
| Píxeles crudos | Sin diseño manual | 49 152 variables para 128×128; ML tradicional sufre | Descartada |
| Histograma HSV 8×4×4 + 8 estadísticos de color (136) | Interpretable, barato, el tono es circular y el histograma lo respeta | No captura forma ni textura; hereda el sesgo de fondo | **Elegida** |
| HOG o textura (LBP, GLCM) | Captura forma y textura | No visto en clase; más tiempo | No probada: queda como trabajo futuro (la CNN ya aprende forma y textura) |

### 4. Modelos de ML tradicional (`GridSearchCV` agrupado, 5 pliegues, F1 macro)

| Alternativa | Resultado en val | Decisión |
|---|---|---|
| Regresión logística (softmax) | F1 0.891 | Línea base lineal |
| SVM RBF | F1 0.944 | **Elegida** |
| Random Forest | F1 0.947 | **Elegida** (entrega probabilidades; va en la app) |
| KNN, XGBoost | No probados | Recorte de alcance permitido: SVM y Random Forest ya cumplen "≥ 2 modelos" |

SVM y Random Forest **no se distinguen** (diferencia de 0.003, menor que un error estándar). Brecha train−val de ~0.05 en ambos: sobreajuste moderado. **La mejor CNN (E4, F1 0.938) no supera al Random Forest (0.947)**: diferencia de 0.008, menor que 2 EE (no concluyente).

### 5. CNN

| Alternativa | Ventaja | Límite | Decisión |
|---|---|---|---|
| CNN desde cero 32-64-128 | Recomendada por el enunciado; liviana en CPU; se vio en clase | Aprende fondo igual que los demás modelos | **Elegida.** Medido (10 experimentos, Buena vs. Mala en Kaggle): mejor F1 en val 0.938 (E4) y 0.929 con GAP (E5) |
| Transfer learning con capas congeladas (MobileNetV2, ImageNet) | Aprovecha lo aprendido con muchas imágenes | Más pesado; el enunciado recomienda priorizar desde cero; se probó **después de usar la prueba** | **Probado como exploración, solo en validación.** Congelada: F1 macro 0.760 (CNN propia: 0.725) y recall de *Regular* 0.49 (CNN propia: 0.60); con ajuste fino: 0.717 y 0.26. Con 35 casos de *Regular* la mejora **no es concluyente**. La app sigue con la CNN propia (`04_modelado/experimento_transfer.py`, `resultados_transfer_validacion.csv`) |
| `Flatten` frente a `GlobalAveragePooling2D` | GAP usa 110 147 parámetros frente a 4 288 067 con `Flatten` en 128 px y 3 clases (verificado contra Keras, `tests/test_cnn.py`) | Menos capacidad | **GAP.** Medido: sin diferencia concluyente (+0.007 sin aumento y −0.009 con aumento, menores que 2 EE) con 22 veces menos parámetros a 96 px |
| Aumento de datos, Dropout, L2, filtros, tasa de aprendizaje, resolución | Control del sobreajuste | — | Una cosa a la vez (`04_modelado/cnn_experimentos.ipynb`). **El aumento de datos ayuda** (+0.033 con `Flatten`, +0.016 con GAP, concluyentes); Dropout con `Flatten` y Dense 64 empeora (−0.016); L2 y 128 px no son concluyentes (128 px cuesta 59.4 s frente a 34.6 s por época); menos filtros (−0.022) y lr 3e-4 (−0.020) empeoran. Una sola semilla por experimento |
| Control "solo borde" con la CNN | Mide cuánto del rendimiento es fondo | — | **Medido (E10):** F1 0.871 con el centro enmascarado frente a 0.929 con la imagen completa; el fondo explica casi todo |

### 6. Segmentación de la fruta y tamaño

| Alternativa | Ventaja | Límite | Decisión |
|---|---|---|---|
| Distancia al color del fondo + Otsu | Sirve con fondos de color | Falla con luz desigual: toma manchas de luz como fruta | Solo para fondos de color (Mixed) |
| GrabCut | Modelo de color por objeto | En fotos reales se extendió a las sombras | Descartada |
| **Saturación HSV + Otsu** | Una sombra es gris y la fruta es saturada | Un fondo muy saturado la rompe | **Elegida** para fondos neutros (revisión visual en 8 fotos sobre superficie) |
| Longitud ℓ = 4√λ₁ / ancho | Estándar | Cambia si la foto es vertical u horizontal | Descartada |
| **ℓ = 4√λ₁ / min(H, W)** | Invariante a la orientación; verificada con elipses y discos de tamaño conocido (error ≤ 3 %) | Relativa: depende de la distancia a la cámara | **Elegida**; puede superar 1 en fotos verticales |

**Validación con la medida real** (`05_evaluacion/validar_tamano.py`, 69 fotos con `max_dim_cm`): la correlación de Spearman entre ℓ y los cm es **débil** (0.26 en todas; 0.71 en las 55 fotos en mano, 0.42 en las 14 sin mano) y, dentro de cada especie, va de −0.66 (naranja) a +0.54 (banano). ℓ no distingue el tamaño real cuando cambia la distancia de la cámara: es útil solo como medida **relativa** y se declara así en la app. Una mano o un brazo se detecta por contacto de la máscara con el borde (> 0.02; calibrado en las fotos propias: 0.000 en 7 fotos sobre superficie y 0.065 a 0.122 en 22 con mano, por revisión visual). **El tamaño solo es válido con la fruta sola sobre una superficie.**

### 7. Construcción de la clase *Regular*

| Alternativa | Ventaja | Límite | Decisión |
|---|---|---|---|
| Solo fotos propias | Control total | Pocas; todas de un dominio | Insuficiente |
| Frutas de *Mixed* segmentadas y etiquetadas a mano | Otro dominio, 6 especies | Costo de etiquetado; segmentación imperfecta (8 194 recortes, 3 978 pasan los filtros) | **Elegida**: 760 recortes (dos rondas de 360 y 400) etiquetados por dos personas por ronda; kappa ponderado 0.607 y 0.606 (`data/mixed_labels/acuerdo_anotadores.csv`); 291 desacuerdos resueltos en grupo; 746 recortes con etiqueta final (385 *Buena*, 199 *Regular*, 162 *Mala*) |
| Dataset adicional con nivel intermedio (AgriFreshNET, CC BY 4.0) | Muchas imágenes | No descargado: faltan especies y tamaño por confirmar; semántica distinta (vida útil) | No usada: la clase *Regular* se amplió con una segunda ronda de *Mixed* (decisión 7); queda como trabajo futuro |
| Sintetizar *Regular* a partir de *Good* y *Bad* | Gratis | Datos inventados | Descartada |

### 8. Clasificador de especie (necesario para el tamaño por especie)

| Alternativa | Resultado en val (6 clases) | Decisión |
|---|---|---|
| Random Forest sobre color, imagen completa | F1 macro 0.908 | Mejor de las probadas |
| CNN pequeña con GAP y aumento de datos | F1 macro 0.790 (detenida en la época 9, sin ajustar) | No supera al Random Forest |
| Control solo borde (Random Forest / CNN) | 0.800 / 0.658 | **El fondo identifica la especie**: el clasificador aprende en parte el entorno |

**Decisión:** no afirmar una exactitud de especie a partir de Kaggle. La app **no predice la especie**: el tamaño se reporta como longitud normalizada, sin clases por especie.

### 9. Despliegue

| Alternativa | Decisión |
|---|---|
| Tkinter o PyQt | Descartadas: Streamlit permite carga de imagen y cámara con poco código |
| **Streamlit** (no visto en clase: se declara) | **Elegida**; procesa en memoria y no guarda imágenes |
| Línea de empaque con Raspberry Pi | Extensión opcional; fuera del alcance |

### 10. Tres clases: desbalance, selección del modelo y prueba

| Alternativa | Ventaja | Límite | Decisión |
|---|---|---|---|
| Fotos propias repartidas entre entrenamiento, validación y prueba | La red ve el dominio propio (manos, fondos) | Con 34 frutas la prueba propia queda en 5 frutas y 12 fotos: no mide nada | Descartada |
| **Las 69 fotos propias solo como prueba final (`own_test`), de uso único** | Mide si el modelo funciona con fotos reales y no deja que nada propio entre al ajuste | El modelo nunca ve el dominio propio | **Elegida** (`construir_tres_clases.py --propias-solo-prueba`) |
| Pesos de clase completos ($w_k=N/(KN_k)$, ≈ 60 para *Regular*) | Compensa el desbalance (137 imágenes de *Regular* frente a más de 10 000) | En la CNN colapsó: predecía *Regular* para casi todo (F1 de validación 0.30 y 0.32, intentos T4 y T5) | **ML tradicional**: completos. **CNN**: descartados |
| **Pesos suavizados (raíz cuadrada) en la CNN** | Estabiliza el entrenamiento | Menos énfasis en *Regular* | **Elegidos** (CNN T6 y T7) |
| Sobremuestreo o SMOTE de *Regular* | Más ejemplos sintéticos | No visto en clase; no se probó por falta de tiempo | Trabajo futuro |

**Selección del modelo (con validación, antes de ver la prueba).** F1 macro de validación: CNN T6 0.725, SVM 0.693, Random Forest 0.628 (pierde *Regular*: recall 0.03), CNN T7 0.623, regresión logística 0.625, línea base 0.257. Se eligió la **CNN T6** (Flatten, 12 épocas) y es la que va en la app. Es también la que menos se sobreajusta (brecha de F1 entrenamiento − validación de 0.04 frente a 0.24 del SVM y 0.30 del Random Forest). Fuente: `04_modelado/resultados_tres_clases.csv` y `tabla_train_val_test.csv`.

**Prueba, usada una sola vez** (`resultados_tres_clases_test.csv`). En Kaggle y *Mixed* (2 746 imágenes, 27 *Regular*) gana el SVM (F1 macro 0.774) sobre el Random Forest (0.765), la CNN T7 (0.736) y la T6 (0.714). **En las 69 fotos propias, ningún modelo detecta *Regular* (0 de 14) y ninguno supera la exactitud de la línea base (0.710)**; la mejor es la CNN T6 (F1 macro 0.439, exactitud 0.681 ± 0.056), que acierta las 6 *Mala*. El Random Forest llama *Mala* a 29 de las 49 *Buena*. Las diferencias entre modelos en *Regular* no son concluyentes (35, 27 y 14 casos). El análisis completo está en `04_modelado/ANALISIS_ERRORES.md`.

**Conclusión de la estrategia.** El sistema distingue *Buena* y *Mala* en las condiciones del dataset, pero con fotos propias generaliza mal y casi no detecta *Regular*. La mejora más directa es conseguir más fotos propias de *Regular* y *Mala* (sobre superficie, sin mano) y entrenar con una parte de ellas.

## Qué falta para cerrar esta tabla

Nada de lo que depende de datos o modelos: todas las decisiones tienen evidencia. Quedan como **trabajo futuro** declarado: HOG o textura, evaluar el transfer learning con una prueba nueva, sobremuestreo de *Regular*, un dataset adicional con nivel intermedio y una CNN entrenada con parte de las fotos propias.
