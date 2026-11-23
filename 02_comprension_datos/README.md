# Fase 2 — Comprensión de los datos

**Notebook:** `eda.ipynb` (se ejecuta desde esta carpeta con Jupyter; usa los módulos de `../src/`).
**Datos esperados en:** `../data/raw/Dataset Fruit/` (ver `../data/README.md`).
**Genera:** figuras vectoriales en `figuras/` (PDF) y, en `../data/processed/` (ignorado por Git), el índice, los hashes, las miniaturas y la partición fija (semilla 42).

**Qué hace:**
1. Inventario por especie y fuente (*Good*, *Bad*, *Mixed*) y desbalances.
2. Convenciones de nombre de archivo y cobertura de marcas de tiempo.
3. Tamaños de imagen y variabilidad de color.
4. Agrupación por **objeto físico** (tiempo + hash perceptual + contador), con calibración de los umbrales sobre los datos reales y su sensibilidad.
5. Partición train/val/test por grupo, estratificada por especie × calidad, sin solapes verificados.
6. **Prueba del atajo:** un Random Forest que solo ve el borde de la imagen (fondo) frente a uno que ve el centro o la imagen completa; usa solo train y val.

**Resultados y hallazgos:** al final del notebook (sección 8). El notebook se sube **con salidas**, ejecutado con los datos reales.

**Alimenta rúbrica:** 1 (sesgos), 6, 8.
Estado: 🟡 ejecutado; falta medir la nitidez como parte de "calidad de imagen".
