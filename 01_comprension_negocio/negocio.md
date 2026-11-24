# Fase 1 — Comprensión del negocio

## El problema
La calidad de la fruta (qué se puede vender, qué se descarta y cuánto se aprovecha) se decide hoy a ojo: es lento, subjetivo y cambia de una persona a otra. Clasificar mal tiene costo en los dos sentidos: descartar fruta aprovechable es desperdicio y dejar pasar fruta en mal estado llega al consumidor. El contexto del desperdicio de alimentos y su fuente están en `../docs/impactos_matriz.md`; las cifras de UNEP que allí aparecen siguen **[VERIFICAR]** y no se repiten aquí.

## Qué se construye
Un sistema que recibe **una fotografía de una fruta** y devuelve:
1. su **calidad**: *Buena*, *Regular* o *Mala* (definiciones operacionales en `../docs/definicion_clases.md`);
2. su **tamaño**: longitud normalizada (medida relativa; la medida real se usa para validarla).

Se entrega como una app en Streamlit (cargar una imagen o usar la cámara), como **apoyo a la decisión** y no como decisión final ni certificación de inocuidad. Especies: manzana, banano, guayaba, lima, naranja y granada.

## Quién lo usa y para qué
| Persona | Para qué |
|---|---|
| Productor o comercializador pequeño | Tener un segundo criterio, más uniforme, al clasificar fruta |
| Persona encargada de la selección | Revisar primero lo que la app marca como dudoso (confianza baja) |
| Consumidor | Consulta informal; la app no certifica nada |

## Preguntas que guían el proyecto
1. ¿Se puede distinguir la calidad de la fruta de varias especies con ML tradicional y con una CNN pequeña, en CPU?
2. ¿Cuánto del rendimiento viene de la fruta y cuánto del fondo? (atajo del fondo, `../02_comprension_datos/eda.ipynb` §7)
3. ¿Funciona con fotos propias distintas del dataset (manos, otros fondos)?
4. ¿La clase intermedia *Regular*, que el dataset no trae, se puede construir y aprender?

## Objetivos medibles y lo que se midió
Los objetivos se miden frente a la línea base de predecir siempre la clase más frecuente; no se fijaron umbrales de éxito después de ver los resultados.

| Objetivo | Cómo se mide | Resultado | Fuente |
|---|---|---|---|
| Clasificar en tres clases mejor que la línea base | F1 macro en validación | CNN T6 0.725 frente a 0.257 de la línea base | `../04_modelado/resultados_tres_clases.csv` |
| Reportar la generalización a fotos propias con su incertidumbre | F1 macro y exactitud ± error estándar en 69 fotos de 34 frutas | F1 macro 0.439; **no se detecta *Regular*** (0 de 14); la exactitud (0.681 ± 0.056) no supera la de la línea base (0.710) | `../04_modelado/resultados_tres_clases_test.csv` |
| Construir *Regular* con etiquetas confiables | Kappa ponderado entre dos anotadores | 0.607 y 0.606 en dos rondas (760 recortes); 199 *Regular* | `../data/mixed_labels/acuerdo_anotadores.csv` |
| App funcional con carga y cámara, y respuesta en menos de 1 s en CPU | Latencia por foto | 116 ms por foto (carga de imagen verificada; **la cámara falta probarla en un computador real**) | `../05_evaluacion/costo_inferencia.csv` |
| Estimar el tamaño y validarlo con la medida real | Correlación de Spearman entre ℓ y los cm medidos con regla | Débil: 0.26 en las 69 fotos; dentro de cada especie va de −0.66 a +0.54. ℓ **no sirve como tamaño real** si cambia la distancia de la cámara; solo como medida relativa | `../05_evaluacion/validacion_tamano_resumen.csv` |
| Informar el costo de cómputo | Tiempo y energía | 7.2 h de procesos de entrenamiento en todo el proyecto; energía **estimada** de 0.11 a 0.40 kWh | `../05_evaluacion/` |

## Criterios de éxito y límites
- Es un éxito que el sistema **supere la línea base** y que se **reporten con honestidad** los casos en que falla; no que acierte siempre.
- Límites declarados: una sola fruta por foto, fondo liso y claro; el tamaño es relativo; con fotos propias el acierto es bajo y *Regular* casi no se detecta; el sistema no certifica inocuidad.

## Ética e impactos
`../docs/etica_tabla.md` y `../docs/impactos_matriz.md`.
