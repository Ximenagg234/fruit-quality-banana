# Formulación matemática del problema

> **Estado: BORRADOR v2 (alcance multifruta).** Cubre rúbrica 3 (formulación), 4 (estrategia) y 5 (fundamentos). Los puntos marcados **[DECIDIR]** dependen de decisiones del equipo o de resultados del EDA y NO deben presentarse como definitivos en el informe hasta resolverse. Las cifras marcadas **[VERIFICAR]** deben contrastarse con el código antes de citarse; el conteo de parámetros de la CNN (§6) ya está verificado.

## 0. Notación

| Símbolo | Significado |
|---|---|
| $x \in [0,1]^{H\times W\times 3}$ | imagen RGB de una fruta, normalizada a $[0,1]$ |
| $c \in \{1,\dots,C\}$ | especie (manzana, banano, guayaba, lima, naranja, granada; $C=6$) |
| $H, W$ | alto y ancho tras redimensionar. **[DECIDIR]** propuesta inicial $H=W=128$ (costo en CPU); el dataset original es $256\times256$ |
| $y \in \{1,2,3\}$ | clase de calidad: 1 = Buena, 2 = Regular, 3 = Mala (**ordinal**: $1<2<3$) |
| $s \in \{S,M,L\}$ | clase de tamaño; $\ell \in \mathbb{R}_{\ge 0}$ longitud normalizada (§3) |
| $K=3$ | número de clases de calidad |
| $\mathcal{D}=\{(x_i,c_i,y_i,s_i,g_i)\}_{i=1}^{N}$ | conjunto de datos; $g_i$ = identificador del objeto físico (§7) |
| $f_\theta$ | clasificador con parámetros $\theta$ |
| $p_\theta(k\mid x)$ | probabilidad que el modelo asigna a la clase $k$ |
| $N_k$ | número de imágenes de entrenamiento de la clase $k$ |

## 1. Planteamiento y por qué es un problema complejo de ingeniería

**Objetivo.** Construir un sistema $g:x\mapsto(\hat c,\hat y,\hat s)$ que, dada una fotografía de una fruta sobre fondo uniforme, devuelva su especie $\hat c$, su calidad $\hat y\in\{1,2,3\}$ y su tamaño $\hat s\in\{S,M,L\}$, y que **generalice a fotos tomadas por nosotros con celular**, no solo al conjunto de referencia. La especie se predice con un clasificador aparte porque el tamaño (§3.1) y las reglas de calidad (`docs/definicion_clases.md`) dependen de ella.

**Riesgo esperado y riesgo empírico.** El modelo ideal minimiza el riesgo esperado

$$
R(f)=\mathbb{E}_{(x,y)\sim P}\big[\ell\big(f(x),y\big)\big],
$$

pero $P$ es desconocida; se minimiza el riesgo empírico regularizado sobre entrenamiento y se **estima $R$ en datos que el modelo no vio** (§7).

**Por qué es complejo** (cada punto se evidenciará con un resultado del proyecto, no solo se afirmará):

1. **Alta dimensionalidad:** con $H=W=128$, $d=H\cdot W\cdot 3=49\,152$ variables por imagen (196 608 a $256\times256$) frente a un número de imágenes mucho menor para las clases propias → riesgo de sobreajuste.
2. **Variabilidad intra-clase e inter-clase solapada:** la maduración es un continuo; la frontera Buena/Regular/Mala es una discretización de una variable continua (la fracción de mancha $\rho$, §3.2), por lo que existe **ruido de etiqueta inherente** cerca de los umbrales.
3. **Etiquetas ordinales:** confundir Buena con Mala es un error más grave que confundir Buena con Regular; la accuracy no lo refleja (se usa kappa ponderado, §8).
4. **Desbalance:** la clase Regular tendrá muchas menos imágenes que Buena/Mala (se construye con fotos propias y segmentaciones de `mixed_quality`).
5. **Cambio de dominio (*domain shift*):** $P_{\text{Kaggle}}(x,y)\neq P_{\text{propias}}(x,y)$ (iluminación, cámara, fondo). Se cuantifica con la brecha $\Delta=\text{F1}_{\text{test Kaggle}}-\text{F1}_{\text{fotos propias}}$.
6. **Fuga de datos por dependencia:** las imágenes se toman en ráfagas desde varios ángulos y parte del dataset parece contener versiones recortadas o en gris de una misma foto; imágenes del mismo objeto no son independientes (§7). *Medido en el EDA:* solo el 80.3 % de los nombres de archivo trae fecha, y las reglas de agrupación tienen fuga residual conocida.
7. **Atajos y cambio de especie:** *medido en el EDA*, un modelo que solo ve el borde de la imagen separa *Good* de *Bad* con F1 macro de 0.76 a 0.96 según la especie, así que el fondo o la sesión de captura delatan la clase; además, la definición de "defecto" cambia con la especie (`docs/definicion_clases.md`), por lo que una frontera de decisión aprendida en una especie no se transfiere automáticamente a otra.
8. **Salida múltiple y restricción de cómputo:** tres salidas (especie, calidad y tamaño) y despliegue en CPU **[DECIDIR]** con latencia objetivo < 1 s por imagen.

## 2. Formulación de la clasificación de calidad

Un clasificador probabilístico produce $z=f_\theta(x)\in\mathbb{R}^K$ (logits) y

$$
p_\theta(k\mid x)=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}},\qquad \hat y=\arg\max_k\,p_\theta(k\mid x).
$$

**Problema de optimización (formulación general):**

$$
\theta^\star=\arg\min_{\theta}\;\underbrace{-\frac1N\sum_{i=1}^{N}w_{y_i}\log p_\theta(y_i\mid x_i)}_{\text{entropía cruzada ponderada}}\;+\;\lambda\,\|\theta\|_2^2,
\qquad w_k=\frac{N}{K\,N_k}.
$$

- $w_k$ compensa el desbalance: una clase con pocos ejemplos pesa más (con clases balanceadas, $w_k=1$).
- $\lambda\ge0$ controla la regularización L2.
- Para la entropía cruzada con softmax, el gradiente respecto a los logits es $\partial L/\partial z_k=p_k-\mathbb{1}[k=y]$, lo que justifica su uso con descenso de gradiente (señal de error proporcional a lo que el modelo se equivoca).

## 3. Estimación del tamaño y validación de etiquetas

### 3.1 Longitud normalizada (salida de tamaño)

Se segmenta la fruta (máscara binaria $M\in\{0,1\}^{H\times W}$, p. ej. umbral de Otsu o en HSV sobre fondo uniforme). Sea $\mathcal{P}=\{p_j=(u_j,v_j)\}$ el conjunto de píxeles con $M=1$, $\bar p$ su centroide y

$$
\Sigma=\frac{1}{|\mathcal{P}|}\sum_{j}(p_j-\bar p)(p_j-\bar p)^\top,\qquad \lambda_1\ge\lambda_2\ \text{valores propios de }\Sigma .
$$

Para una región elíptica uniforme con semieje mayor $a$, la varianza a lo largo de ese eje es $a^2/4$, luego la longitud del eje mayor es

$$
L=4\sqrt{\lambda_1},\qquad \ell=\frac{L}{\min(H,W)}\quad(\text{longitud normalizada}).
$$

Se normaliza por el **lado menor** de la imagen, $\min(H,W)$, y no por el ancho, para que $\ell$ no cambie si la foto es vertical u horizontal (con el mismo celular y la misma distancia, el lado menor abarca el mismo campo de visión en ambas orientaciones). Por eso $\ell$ **puede superar 1** en una foto vertical con una fruta larga sin que haya ningún error. Además de $\ell$ se calcula la fracción $b$ del borde de la imagen que cae dentro de la máscara: un $b$ alto ($>0.02$) delata una mano o un brazo que entra por el borde, o una fruta cortada, y la medida no se considera fiable (umbral calibrado en las fotos propias: $b=0$ en 7 fotos sobre superficie y $0.065$–$0.122$ en 22 fotos con mano, por revisión visual).

La fórmula sirve para **cualquier forma**: en una elipse da el eje mayor y en un disco de radio $r$ la varianza a lo largo de cualquier eje es $r^2/4$, luego $L=4\sqrt{r^2/4}=2r$, el diámetro. Por eso una sola definición cubre al banano (alargado) y a la naranja o la granada (redondas).

Las clases de tamaño se definen **por especie** con terciles **calculados solo con el conjunto de entrenamiento**: $\hat s=S$ si $\ell<\tau_1^{(c)}$, $M$ si $\tau_1^{(c)}\le\ell<\tau_2^{(c)}$, $L$ si $\ell\ge\tau_2^{(c)}$, con $\tau_1^{(c)},\tau_2^{(c)}$ los cuantiles $1/3$ y $2/3$ de $\ell$ de la especie $c$ en entrenamiento (evita filtrar información del test y no compara el tamaño de una manzana con el de un banano).

**Limitaciones que se declararán en el informe:** (i) $\ell$ depende de la distancia cámara–fruta: las imágenes de Kaggle **no tienen escala**, así que allí el tamaño es solo relativo; en las fotos propias la regla o el objeto de referencia permiten medirlo en cm y **validar** $\ell$ **[DECIDIR]**; (ii) la banana es curva y el eje mayor de PCA subestima ligeramente la longitud de arco; (iii) una mano que sostiene la fruta o varias frutas en la foto alteran la máscara.

### 3.2 Fracción de mancha (criterio operacional de calidad)

Sea $B_c\subset\text{HSV}$ la región de color de defecto de la especie $c$ (p. ej. marrón/negro en el banano; la definición de defecto cambia con la especie, ver `docs/definicion_clases.md` §2; umbrales a calibrar). Se define

$$
\rho(x)=\frac{\sum_{p\in\mathcal{P}}\mathbb{1}[x_p\in B_c]}{|\mathcal{P}|}\in[0,1].
$$

Regla de etiquetado propuesta (ver `docs/definicion_clases.md`): Buena si $\rho<0.10$; Regular si $0.10\le\rho\le0.40$; Mala si $\rho>0.40$ o hay moho/zona blanda. **Los umbrales 0.10 y 0.40 son una propuesta, no una norma [DECIDIR]**; $\rho$ permite validar objetivamente las etiquetas manuales y medir el acuerdo entre anotadores.

## 4. Preprocesamiento y normalización

1. **Escalado de píxeles:** $x=\text{imagen}_{uint8}/255\in[0,1]$. Justificación: estabiliza el gradiente y homogeneiza la escala de entrada.
2. **Características manuales para ML tradicional** (histogramas de color en HSV, estadísticos de $\rho$, textura): se **estandarizan** por característica

$$
\tilde z_j=\frac{z_j-\mu_j}{\sigma_j},
$$

con $\mu_j,\sigma_j$ **calculados solo con entrenamiento** (dentro de un `Pipeline`, para que la validación cruzada no filtre estadísticos). Es imprescindible para SVM y KNN, que dependen de distancias euclídeas $\|z-z'\|_2$.
3. **Aumento de datos** (rotación, volteo, zoom, brillo/contraste) solo sobre entrenamiento, para simular la variabilidad de dominio.
4. **Desbalance:** pesos $w_k$ (§2) y/o aumento focalizado en la clase minoritaria. No se hace sobremuestreo antes de dividir los datos.

## 5. Funciones de pérdida y criterios de ajuste por familia de modelo

*(Se conservarán en el informe solo las de los modelos finalmente elegidos.)*

| Modelo **[DECIDIR]** | Criterio / pérdida | Hiperparámetros a ajustar |
|---|---|---|
| Regresión logística / softmax (línea base) | entropía cruzada + L2, §2 | $C=1/\lambda$ |
| SVM con núcleo RBF | $\min_{w,b,\xi}\tfrac12\|w\|^2+C\sum_i\xi_i$ s.a. $y_i(w^\top\phi(x_i)+b)\ge1-\xi_i$; $K(x,x')=e^{-\gamma\|x-x'\|^2}$ | $C,\gamma$ |
| Random Forest / XGBoost | impureza de Gini $G=1-\sum_k p_k^2$ por nodo / objetivo regularizado de segundo orden | profundidad, nº de árboles, tasa de aprendizaje |
| CNN | entropía cruzada ponderada (§2) + L2/Dropout; optimizador Adam | filtros, kernel, dropout, tasa de aprendizaje |

## 6. CNN: operaciones y conteo de parámetros

**Correlación cruzada** (lo que implementan Keras y PyTorch como "convolución"), para entrada $I$ y kernel $K$ de $k\times k$:

$$
(I\star K)(i,j)=\sum_{m=0}^{k-1}\sum_{n=0}^{k-1}I(i+m,\,j+n)\,K(m,n).
$$

**Tamaño de salida** con entrada $n$, kernel $k$, relleno $p$ y paso $s$ (padding `valid`): $\;n_{\text{out}}=\lfloor (n-k+2p)/s\rfloor+1$; con padding `same`: $\lceil n/s\rceil$.

**Parámetros de una capa Conv2D** con $c_{in}$ canales de entrada y $c_{out}$ filtros: $\;(k^2c_{in}+1)\,c_{out}$. De una capa densa: $(n_{in}+1)\,n_{out}$.

**Max-pooling** $2\times2$ con paso 2 reduce alto y ancho a la mitad y no tiene parámetros. **ReLU:** $\max(0,z)$.

**Ejemplo ilustrativo (conteos verificados contra `model.count_params()` de Keras en `tests/test_cnn.py`; implementación en `src/cnn.py`)** — entrada $128\times128\times3$, tres bloques Conv($3\times3$, `same`)+MaxPool(2) con 32, 64, 128 filtros:

| Capa | Salida | Parámetros |
|---|---|---|
| Conv 32 | $128\times128\times32$ | $(9\cdot3+1)\cdot32=896$ |
| Conv 64 (tras pool) | $64\times64\times64$ | $(9\cdot32+1)\cdot64=18\,496$ |
| Conv 128 (tras pool) | $32\times32\times128$ | $(9\cdot64+1)\cdot128=73\,856$ |
| Tras 3 pools | $16\times16\times128$ | 0 |
| Flatten → Dense 128 | $32\,768\to128$ | $32\,768\cdot128+128=4\,194\,432$ |
| Dense 3 (softmax) | 3 | $128\cdot3+3=387$ |
| **Total** | | **4 288 067** |

Con `GlobalAveragePooling2D` en lugar de `Flatten`, la capa Dense(128) pasa a $128\cdot128+128=16\,512$ parámetros y el total a **110 147**: casi todo el modelo con `Flatten` está en una sola capa densa, lo que aumenta el riesgo de sobreajuste con pocos datos (mismo análisis que en la guía del curso `02-CNN_resuelto`). Esa decisión se justificará con la curva de entrenamiento/validación.

## 7. Estrategia de validación (anti-fuga de datos)

- **Partición por objeto, no por imagen.** Sea $g_i$ el objeto físico de la imagen $i$. Train/val/test se construyen de modo que **ningún $g$ aparezca en dos particiones** y estratificando por **especie × calidad** (`StratifiedGroupKFold`, no visto en clase: se declara y cita). Motivo: imágenes del mismo objeto (ráfagas, ángulos) están fuertemente correlacionadas; separarlas al azar infla las métricas.
- **Cómo se forman los grupos** (el dataset de Kaggle no trae identificador de objeto): dentro de cada celda especie × fuente, se unen las imágenes con marcas de tiempo separadas por $\le 3$ s, las de **hash perceptual** (aHash de 256 bits) a distancia de Hamming $\le 5$ y, para los nombres `IMG_NNNN` sin fecha, las de contador consecutivo con hash a distancia $\le 30$; los grupos son los componentes conexos. Los umbrales se **calibraron con datos** (`02_comprension_datos/eda.ipynb`): con umbrales de hash mayores la clausura transitiva forma grupos gigantes (hasta 4 732 imágenes) y la partición por celda se descuadra. Limitación declarada: rotaciones o recortes que el hash no detecta dejan **fuga residual**, mayor en *Good* apple, guava, banana y pomegranate.
- **Selección de hiperparámetros:** $k$-fold agrupado ($k=5$) sobre entrenamiento con `GridSearchCV`; el conjunto de test se usa **una sola vez** al final.
- **Conjuntos de evaluación por dominio:** (a) test del dataset de referencia, (b) fotos propias (dominio real), (c) frutas segmentadas de `Mixed`.
- **Línea base de atajo:** junto a cada modelo se reporta el rendimiento de un clasificador que **solo ve el borde de la imagen**; si el modelo no la supera con claridad, no se puede atribuir su rendimiento a la calidad de la fruta.

## 8. Métricas e incertidumbre

Con la matriz de confusión $C_{ij}$ (fila = real, columna = predicho):

$$
P_k=\frac{C_{kk}}{\sum_i C_{ik}},\quad R_k=\frac{C_{kk}}{\sum_j C_{kj}},\quad F1_k=\frac{2P_kR_k}{P_k+R_k},\quad F1_{\text{macro}}=\frac1K\sum_k F1_k .
$$

- **Métricas globales y por especie:** cada métrica se calcula sobre todo el conjunto y por separado para cada especie $c$ (un promedio global puede esconder una especie que falla; la granada, por ejemplo, domina el conteo de imágenes *Good*).
- **F1 macro** (y no solo accuracy) porque hay desbalance: predecir siempre la clase mayoritaria da alta accuracy y F1 macro bajo (paradoja de la exactitud vista en clase).
- **Kappa de Cohen ponderado cuadráticamente**, por la naturaleza ordinal:

$$
\kappa_w=1-\frac{\sum_{i,j}w_{ij}O_{ij}}{\sum_{i,j}w_{ij}E_{ij}},\qquad w_{ij}=\frac{(i-j)^2}{(K-1)^2},
$$

donde $O$ son las frecuencias observadas y $E$ las esperadas por azar.
- **Error estándar binomial** de una exactitud $\hat p$ con $n$ imágenes de prueba: $\mathrm{SE}=\sqrt{\hat p(1-\hat p)/n}$. Con $n\approx30$–$50$ fotos propias, **una sola imagen mueve la métrica 2–3 puntos**; las diferencias menores que $2\,\mathrm{SE}$ no se declararán concluyentes (criterio usado por el profesor en la guía de CNN).
- **Sobreajuste:** brecha $\text{métrica}_{train}-\text{métrica}_{val}$ y curvas de aprendizaje (métrica vs. tamaño de entrenamiento).

## 9. Preguntas abiertas que bloquean el cierre de este documento

1. ~~¿El dataset de Kaggle trae identificador de fruta?~~ **Resuelto en el EDA:** no lo trae; se infiere con tiempo + hash + contador (§7), con fuga residual declarada.
2. ¿Qué modelos de ML tradicional se eligen finalmente (SVM y RF/XGBoost son los candidatos)?
3. $H,W$ definitivos (medir una época de la CNN en CPU antes de decidir) y si la CNN usa `Flatten` o `GlobalAveragePooling2D`.
4. Umbrales de $\rho$ por especie y tratamiento de la fruta verde (propuesta: Buena si no tiene defectos).
5. Referencia de escala para el tamaño en fotos propias (regla en cuadro).
6. Fuentes definitivas de la clase *Regular* (fotos propias, frutas de *Mixed* etiquetadas y, si se aprueba, un dataset adicional con nivel intermedio).
