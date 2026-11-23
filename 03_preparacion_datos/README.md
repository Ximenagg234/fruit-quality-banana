# Fase 3 — Preparación de los datos

**Ya hecho:** la partición por objeto, sin fuga de datos, está en `../02_comprension_datos/eda.ipynb` (grupos por tiempo, hash perceptual y contador); la segmentación de la fruta y el tamaño están en `../src/segmentation.py`; la preparación de las fotos propias, en `../src/prepare_photos.py`.

## Frutas individuales de `Mixed` y etiquetado (construcción de la clase *Regular*)

`Mixed` trae varias frutas por foto y ninguna etiqueta. El flujo para usarlo es:

| Paso | Herramienta | Resultado |
|---|---|---|
| 1. Segmentar cada fruta | `segmentar_mixed.py` (usa `../src/mixed.py`) | recortes, manifiesto con métricas y banderas de calidad, y una **muestra reproducible** de 60 recortes por especie |
| 2. Etiquetar a mano, cada persona por su cuenta | `etiquetador_mixed.py` (Streamlit) | `etiquetas_mixed_<INICIALES>.csv` por persona |
| 3. Medir el acuerdo y resolver desacuerdos | `calcular_acuerdo.py` | kappa ponderado cuadráticamente y `desacuerdos.csv` |

Guía para quien etiqueta: ver `docs/REPARTO_TRABAJO.md` §5 y las reglas de clase en `docs/definicion_clases.md`.

La segmentación **no tiene verdad de terreno** y se valida a ojo: las frutas sobre pasto o los bananos sobre vegetación son los casos difíciles, y por eso la herramienta incluye el botón "Descartar".

**Pendiente:** unir las etiquetas finales con las fotos propias para construir *Regular*; normalización y aumento de datos; manejo del desbalance (submuestreo por celda especie × clase y pesos de clase, ya usados en los modelos).
**Alimenta rúbrica:** 3, 5 (normalización), 6.
Estado: 🟡 (segmentación y herramientas listas; faltan las etiquetas)
