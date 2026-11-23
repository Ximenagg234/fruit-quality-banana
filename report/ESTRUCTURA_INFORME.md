# Estructura del informe IEEE (máximo 7 páginas)

Guía para armar el informe en Overleaf o Word con la plantilla del profesor (`conference-template-letter.docx`, o la plantilla clonable de la carpeta `PROYECTO/`). **No hay LaTeX en el entorno de desarrollo, así que no se compiló nada.** Las secciones y su orden salen de los lineamientos (§7). Responsable: Natalia; revisor: Juan Pablo (revisión cruzada con la rúbrica, sin haberlo escrito).

## Reglas

- Máximo **7 páginas**, formato IEEE de dos columnas; se quita todo el texto guía de la plantilla.
- Figuras **vectoriales** (PDF o SVG), numeradas como "Fig. 1", con ejes rotulados con palabras y unidades.
- Referencias **numeradas y en formato IEEE**, solo las **leídas y usadas**: salen de `report/refs.bib`, donde cada una dice contra qué se verificó. Las marcadas `[VERIFICAR]` no se citan hasta comprobarlas.
- **Ningún número sin fuente.** Toda cifra sale de un notebook ejecutado o de un CSV del repo (tabla de abajo). Lo que no existe aún se escribe `[MEDIR]`, no se estima.
- Técnicas **no vistas en clase**, que hay que declarar: `StratifiedGroupKFold`, hash perceptual, componentes conexos de un grafo, segmentación por umbral de Otsu, transformada de distancia y `watershed`, Streamlit.
- La declaración del uso de herramientas de IA (fila E8) **la decide Ximena**; no se escribe sin hablar con ella.

## Presupuesto de páginas (sugerido) y contenido por sección

| Sección | Págs. | Qué debe decir | Evidencia en el repo | Criterio |
|---|---|---|---|---|
| Título, resumen | 0.2 | Problema, método, resultado principal y la advertencia del atajo del fondo | todo lo demás | 8 |
| Introducción | 0.6 | Desperdicio de alimentos y clasificación manual; objetivo; contribuciones | `docs/impactos_matriz.md` (contexto con fuente), `fao2019sofa` | 2, 8 |
| Fundamentos teóricos | 0.9 | Solo lo necesario: pérdida y softmax, SVM, Random Forest, CNN (conteo de parámetros), F1, kappa ponderado, PCA para el tamaño. Sin generalidades | `docs/formulacion_matematica.md` | 3, 5 |
| Metodología | 1.4 | Diagrama CRISP-DM **propio**, datos, partición por objeto, clase Regular, modelos, validación | `docs/diagrama_crispdm.pdf`, `docs/estrategia_solucion.md`, `docs/definicion_clases.md` | 4, 5, 8 |
| Resultados | 1.6 | Tablas y figuras de las métricas por modelo, por especie y por dominio | tabla de abajo | 6 |
| Análisis de resultados | 1.3 | ¿Generalizan? ¿Sobreajuste? ¿Qué falla? El atajo del fondo, fotos propias, comparación con la literatura | `04_modelado/`, EDA §7 | 6, 7 |
| Ética e impactos | (dentro de metodología o análisis) | Tabla de dilemas con principios del código de ética y matriz de impactos | `docs/etica_tabla.md`, `docs/impactos_matriz.md` | 1, 2 |
| Conclusiones y trabajo futuro | 0.5 | Cada conclusión cita la tabla o figura que la respalda | — | 7 |
| Referencias | 0.5 | IEEE, solo las usadas | `report/refs.bib` | 8 |

## Cifras ya medidas y de dónde salen

| Cifra | Fuente |
|---|---|
| 19 526 imágenes; 6 especies; *Good* 11 664, *Bad* 6 788, *Mixed* 1 074; *Good* granada 5 940 (51 % de *Good*) | `02_comprension_datos/eda.ipynb` §1 |
| 80.3 % de los nombres con fecha; 10 convenciones de nombre | `eda.ipynb` §2 |
| 3 360 grupos; test por celda entre 14.2 % y 15.3 %; calibración (hash ≤ 40: grupo de 4 732) | `eda.ipynb` §5–6 |
| 1 188 imágenes en escala de grises, 20.0 % de *Good* granada; 564 imágenes de más de 1 000 px, todas de *Good* granada | `eda.ipynb` §3–4 |
| **Atajo del fondo:** F1 macro solo con el borde = 0.76 (manzana) a 0.96 (granada); 0.84 con todas | `eda.ipynb` §7, `02_comprension_datos/figuras/prueba_atajo_borde.pdf` |
| Modelos clásicos (F1 macro en val): mayoritaria 0.388; logística 0.891; SVM 0.944; Random Forest 0.947. Solo borde: 0.852; 0.903; 0.931 | `04_modelado/resultados_modelos_clasicos.csv`, `modelos_clasicos.ipynb` |
| CNN: 4 288 067 parámetros con `Flatten` y 110 147 con GAP (128 px, 3 clases) | `src/cnn.py`, `tests/test_cnn.py` |
| Segmentación: error de la longitud ≤ 3 % en formas sintéticas; contacto con el borde 0.000 en 7 fotos sobre superficie y 0.065–0.122 en 22 con mano (revisión visual) | `tests/test_segmentation.py`, `docs/estrategia_solucion.md` §6 |
| Mixed: 8 194 recortes, 3 978 pasan los filtros; muestra de 360 para etiquetar | `03_preparacion_datos/segmentar_mixed.py` |
| **CNN (Buena vs. Mala, val):** mejor E4 0.938 (`Flatten`) y E5 0.929 (GAP, 22 veces menos parámetros, diferencia no concluyente); aumento de datos ayuda (+0.033 y +0.016); no supera al Random Forest (0.947); control solo borde E10 = 0.871. **Especie:** Random Forest sobre color 0.908, CNN 0.790; solo borde 0.800 y 0.658 | `04_modelado/resultados_cnn_calidad.csv`, `resultados_cnn_especie.csv`, `cnn_experimentos.ipynb` |
| Partir por imagen infla el F1 en ≈ 0.027 (0.987 frente a 0.960; cota inferior); 49.9 % de las imágenes de prueba con casi-duplicado en train | `04_modelado/fuga_particion.ipynb` |
| Fotos propias, *Regular*, evaluación en fotos propias, kappa entre anotadores, test | **[MEDIR]**: aún no existen |

## Figuras disponibles

| Figura | Archivo |
|---|---|
| Diagrama CRISP-DM personalizado | `docs/diagrama_crispdm.pdf` (y `.svg`, `.png`) |
| Inventario *Good* vs. *Bad* | `02_comprension_datos/figuras/inventario_good_bad.pdf` |
| Variabilidad de color | `02_comprension_datos/figuras/variabilidad_color.pdf` |
| Calibración del hash perceptual | `02_comprension_datos/figuras/calibracion_hash.pdf` |
| Prueba del atajo del fondo | `02_comprension_datos/figuras/prueba_atajo_borde.pdf` |
| Modelos clásicos y matriz de confusión | `04_modelado/figuras/modelos_clasicos.pdf` |
| Curvas de aprendizaje de la CNN | `04_modelado/figuras/cnn_curvas_aprendizaje.pdf` |
| CNN frente a ML tradicional y control solo borde | `04_modelado/figuras/cnn_comparacion.pdf` |
| Inflación por partición aleatoria | `04_modelado/figuras/fuga_particion.pdf` |

## Antes de entregar

1. Cada número del informe tiene su fila en la tabla de arriba (o en un notebook enlazado).
2. Cada referencia citada está en `refs.bib` y se leyó.
3. Las técnicas no vistas en clase están declaradas.
4. `docs/RUBRICA_TRAZABILIDAD.md` está actualizada y se recorrió criterio por criterio contra el texto Outstanding.
5. El PDF cumple el límite de 7 páginas.
