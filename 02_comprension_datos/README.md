# Fase 2 — Comprensión de los datos

**Notebook:** `eda.ipynb` (se ejecuta desde esta carpeta: `jupyter lab` y abrirlo; usa `../src/data_utils.py`).
**Datos esperados en:** `../data/raw/Dataset Fruit/` (ver `../data/README.md`). Se puede cambiar con la variable de entorno `FRUIT_DATA_ROOT`.
**Genera:** figuras en `figures/` (PDF vectorial o PNG) y el índice `../data/processed/banana_index.csv` (ignorado por Git).

**Qué hace:** inventario de imágenes de banana, agrupación por banana física a partir de la marca de tiempo del nombre de archivo (**hipótesis que se valida visualmente**), estadísticos de calidad de imagen, línea base trivial con color global (con partición por grupo), duplicados y revisión de `Mixed`.

**Importante:** el notebook se sube **sin salidas**; tras ejecutarlo con los datos reales, completar la tabla de la sección 8 y subir las figuras de `figures/`.

**Alimenta rúbrica:** 1 (sesgos), 6, 8.
Estado: 🟡 código listo, falta ejecutarlo con los datos reales
