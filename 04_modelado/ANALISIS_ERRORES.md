# Análisis de errores (tres clases)

Fuentes: `resultados_tres_clases.csv` (validación), `resultados_tres_clases_test.csv` (prueba, usada una sola vez),
`tabla_train_val_test.csv`, `errores_resumen.csv`, `errores_fotos_propias.csv` y las figuras
`figuras/matrices_confusion_tres_clases.pdf` y `figuras/errores_cnn_T6.pdf`. Se generan con `analisis_modelos.py`.
Los números son los medidos; las explicaciones marcadas como **hipótesis** no se han probado.

## 1. Sobreajuste (F1 macro de entrenamiento menos validación)
| Modelo | F1 entrenamiento | F1 validación | Brecha |
|---|---|---|---|
| Regresión logística | 0.748 | 0.625 | 0.12 |
| SVM | 0.929 | 0.693 | 0.24 |
| Random Forest | 0.927 | 0.628 | 0.30 |
| Random Forest solo borde | 1.000 | 0.630 | 0.37 |
| CNN T7 (GAP) | 0.729 | 0.623 | 0.11 |
| **CNN T6 (Flatten)** | 0.765 | 0.725 | **0.04** |

El SVM y el Random Forest memorizan el entrenamiento (en el control de solo borde el F1 de entrenamiento es 1.0); la CNN T6
es la que menos se sobreajusta. El entrenamiento está submuestreado (<= 1 100 imágenes por especie y clase).

## 2. Qué se confunde
- **Regular es la clase débil.** En validación la CNN T6 acierta 21 de 35 *Regular* (13 las llama *Mala* y 1 *Buena*), pero
  también llama *Regular* a 68 imágenes que no lo son (precisión 0.24, F1 0.34). En la prueba de Kaggle y *Mixed* acierta 22 de 27, con
  129 predichas como *Regular* (precisión 0.17). Detecta bastante, pero con muchas falsas alarmas.
- **Los errores graves (Buena contra Mala)** existen pero son menos frecuentes que los que involucran *Regular*: en la prueba, la T6
  confunde 73 *Buena* con *Mala* y 54 *Mala* con *Buena* de unas 2 700 imágenes.
- **En las imágenes de Mixed de la prueba** (102 recortes) la T6 acierta solo el 9 % de las *Buena* (las llama *Regular*), aunque
  recupera el 81 % de las *Regular*: ahí cambia precisión por cobertura (`resultados_tres_clases_test.csv`, `test_mixed`).
- Los ejemplos de `figuras/errores_cnn_T6.pdf` muestran errores en frutas con manchas pequeñas o con otras frutas en el recorte.

## 3. Fotos propias (69 fotos, 34 frutas)
- **Ningún modelo detecta *Regular*** (0 de 14). La CNN T6 manda 8 a *Buena* y 6 a *Mala*.
- La T6 acierta las 6 *Mala* y 41 de 49 *Buena*; su exactitud (0.68) queda por debajo de la de la línea base de decir siempre
  *Buena* (0.71), con un error estándar de ±0.06.
- **Con y sin mano.** Las 6 *Mala* están todas en mano. Sobre superficie sin mano solo hay 7 *Buena* y 7 *Regular* (14 fotos): la T6 acierta 5 de
  14 (0 de 7 *Regular*). Con 14 fotos no hay base para afirmar un efecto de la mano, pero es compatible con la confusión
  entre escena y calidad que se vio al preparar las fotos (**hipótesis**).
- **El Random Forest** llama *Mala* a 29 de las 49 *Buena* de las fotos propias (exactitud 0.36). Que sea el modelo que más falla
  justo en fotos con fondos y manos distintos a Kaggle es compatible con que use el fondo como pista (**hipótesis**; el control de solo borde
  ya mostró que el fondo informa sobre *Buena* y *Mala*).
- Por especie (`resultados_tres_clases_test.csv`, conjuntos `own_<especie>`): la lima (solo *Buena*) la acierta la T6 en las 9 fotos; la
  manzana es donde peor le va (exactitud 0.45).

## 4. Lo que no se puede concluir
*Regular* tiene 35 casos en validación, 27 en prueba y 14 en fotos propias: las diferencias entre modelos en esa clase están dentro del
error. Las fotos propias no se muestran en las figuras porque faltan los permisos de las personas cuyas manos aparecen.
