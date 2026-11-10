# Formulación matemática del problema

> **Estado: BORRADOR v1.** Cubre rúbrica 3 (formulación), 4 (estrategia) y 5 (fundamentos). Los puntos marcados **[DECIDIR]** dependen de decisiones del equipo o de resultados del EDA y NO deben presentarse como definitivos en el informe hasta resolverse. Las cifras de ejemplo marcadas **[VERIFICAR]** deben contrastarse con `model.summary()` o con el código antes de citarse.

## 0. Notación

| Símbolo | Significado |
|---|---|
| $x \in [0,1]^{H\times W\times 3}$ | imagen RGB de una banana, normalizada a $[0,1]$ |
| $H, W$ | alto y ancho tras redimensionar. **[DECIDIR]** propuesta inicial $H=W=128$ (costo en CPU); el dataset original es $256\times256$ |
| $y \in \{1,2,3\}$ | clase de calidad: 1 = Buena, 2 = Regular, 3 = Mala (**ordinal**: $1<2<3$) |
| $s \in \{S,M,L\}$ | clase de tamaño; $\ell \in \mathbb{R}_{\ge 0}$ longitud normalizada (§3) |
| $K=3$ | número de clases de calidad |
| $\mathcal{D}=\{(x_i,y_i,s_i,g_i)\}_{i=1}^{N}$ | conjunto de datos; $g_i$ = identificador de la banana física |
| $f_\theta$ | clasificador con parámetros $\theta$ |
| $p_\theta(k\mid x)$ | probabilidad que el modelo asigna a la clase $k$ |
| $N_k$ | número de imágenes de entrenamiento de la clase $k$ |

## 1. Planteamiento y por qué es un problema complejo de ingeniería

**Objetivo.** Construir un sistema $g:x\mapsto(\hat y,\hat s)$ que, dada una fotografía de una banana sobre fondo uniforme, devuelva su calidad $\hat y\in\{1,2,3\}$ y su tamaño $\hat s\in\{S,M,L\}$, y que **generalice a fotos tomadas por nosotros con celular**, no solo al conjunto de referencia.

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
6. **Fuga de datos por dependencia:** el dataset muestra cada fruta desde 5 ángulos; imágenes de la misma banana no son independientes (§7).
7. **Salida múltiple y restricción de cómputo:** dos salidas (calidad y tamaño) y despliegue en CPU **[DECIDIR]** con latencia objetivo < 1 s por imagen.

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

Se segmenta la banana (máscara binaria $M\in\{0,1\}^{H\times W}$, p. ej. umbral de Otsu o en HSV sobre fondo uniforme). Sea $\mathcal{P}=\{p_j=(u_j,v_j)\}$ el conjunto de píxeles con $M=1$, $c$ su centroide y

$$
\Sigma=\frac{1}{|\mathcal{P}|}\sum_{j}(p_j-c)(p_j-c)^\top,\qquad \lambda_1\ge\lambda_2\ \text{valores propios de }\Sigma .
$$

Para una región elíptica uniforme con semieje mayor $a$, la varianza a lo largo de ese eje es $a^2/4$, luego la longitud del eje mayor es

$$
L=4\sqrt{\lambda_1},\qquad \ell=\frac{L}{W}\quad(\text{longitud normalizada}).
$$

Las clases de tamaño se definen por **terciles calculados solo con el conjunto de entrenamiento**: $\hat s=S$ si $\ell<\tau_1$, $M$ si $\tau_1\le\ell<\tau_2$, $L$ si $\ell\ge\tau_2$, con $\tau_1,\tau_2$ los cuantiles $1/3$ y $2/3$ de $\ell$ en entrenamiento (evita filtrar información del test; mismo criterio que en el informe del Ingenio).

**Limitaciones que se declararán en el informe:** (i) $\ell$ depende de la distancia cámara–fruta; las fotos propias se toman a distancia fija o con objeto de referencia **[DECIDIR]**; (ii) la banana es curva y el eje mayor de PCA subestima ligeramente la longitud de arco.

### 3.2 Fracción de mancha (criterio operacional de calidad)

Sea $B\subset\text{HSV}$ la región de color marrón/negro (umbrales a calibrar en el EDA). Se define

$$
\rho(x)=\frac{\sum_{p\in\mathcal{P}}\mathbb{1}[x_p\in B]}{|\mathcal{P}|}\in[0,1].
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

**Ejemplo ilustrativo [VERIFICAR con `model.summary()`]** — entrada $128\times128\times3$, tres bloques Conv($3\times3$, `same`)+MaxPool(2) con 32, 64, 128 filtros:

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

- **Partición por banana, no por imagen.** Sea $g_i$ la banana física de la imagen $i$. Train/val/test se construyen de modo que **ningún $g$ aparezca en dos particiones** (`GroupKFold`/`GroupShuffleSplit`), y estratificando por clase cuando es posible. Motivo: imágenes de una misma fruta (5 ángulos, 3 días) están fuertemente correlacionadas; separarlas al azar infla las métricas.
  **[DECIDIR / VERIFICAR en EDA]** Si el dataset de Kaggle no trae identificador de fruta, se agrupará por similitud (p. ej. *hash* perceptual) y se documentará la limitación.
- **Selección de hiperparámetros:** $k$-fold agrupado ($k=5$) sobre entrenamiento con `GridSearchCV`; el conjunto de test se usa **una sola vez** al final.
- **Tres conjuntos de evaluación:** (a) test del dataset de referencia, (b) fotos propias (dominio real), (c) frutas segmentadas de `mixed_quality`.

## 8. Métricas e incertidumbre

Con la matriz de confusión $C_{ij}$ (fila = real, columna = predicho):

$$
P_k=\frac{C_{kk}}{\sum_i C_{ik}},\quad R_k=\frac{C_{kk}}{\sum_j C_{kj}},\quad F1_k=\frac{2P_kR_k}{P_k+R_k},\quad F1_{\text{macro}}=\frac1K\sum_k F1_k .
$$

- **F1 macro** (y no solo accuracy) porque hay desbalance: predecir siempre la clase mayoritaria da alta accuracy y F1 macro bajo (paradoja de la exactitud vista en clase).
- **Kappa de Cohen ponderado cuadráticamente**, por la naturaleza ordinal:

$$
\kappa_w=1-\frac{\sum_{i,j}w_{ij}O_{ij}}{\sum_{i,j}w_{ij}E_{ij}},\qquad w_{ij}=\frac{(i-j)^2}{(K-1)^2},
$$

donde $O$ son las frecuencias observadas y $E$ las esperadas por azar.
- **Error estándar binomial** de una exactitud $\hat p$ con $n$ imágenes de prueba: $\mathrm{SE}=\sqrt{\hat p(1-\hat p)/n}$. Con $n\approx30$–$50$ fotos propias, **una sola imagen mueve la métrica 2–3 puntos**; las diferencias menores que $2\,\mathrm{SE}$ no se declararán concluyentes (criterio usado por el profesor en la guía de CNN).
- **Sobreajuste:** brecha $\text{métrica}_{train}-\text{métrica}_{val}$ y curvas de aprendizaje (métrica vs. tamaño de entrenamiento).

## 9. Preguntas abiertas que bloquean el cierre de este documento

1. ¿El dataset de Kaggle trae identificador de fruta para agrupar? (EDA)
2. ¿Qué modelos de ML tradicional se eligen finalmente (SVM y RF/XGBoost son los candidatos)?
3. $H,W$ definitivos y si la CNN usa `Flatten` o `GlobalAveragePooling2D`.
4. Umbrales de $\rho$ y tratamiento de la banana verde (propuesta: Buena).
5. Referencia de escala para el tamaño en fotos propias.
