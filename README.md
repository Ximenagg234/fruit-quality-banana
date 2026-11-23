# fruit-quality-banana

Sistema de clasificación de **calidad** (Buena / Regular / Mala) y estimación de **tamaño** de frutas de varias especies (manzana, banano, guayaba, lima, naranja y granada) a partir de imágenes estáticas, con una app en Streamlit (carga de imagen o cámara), para el proyecto final de **Algoritmos y Programación III (2026-2)**, Universidad Icesi. Metodología: **CRISP-DM**.

## Equipo
| Integrante | GitHub | Rol principal |
|---|---|---|
| Ximena Gómez Gómez | @Ximenagg234 | _por definir_ |
| Natalia Delgado | _por definir_ | _por definir_ |
| Juan Pablo Bello | _por definir_ | _por definir_ |

## Estructura (por fases de CRISP-DM)
| Carpeta | Contenido |
|---|---|
| `01_comprension_negocio/` | contexto, objetivos SMART, impacto |
| `02_comprension_datos/` | EDA |
| `03_preparacion_datos/` | segmentación, partición, normalización |
| `04_modelado/` | modelos de ML, CNN, ajuste de hiperparámetros |
| `05_evaluacion/` | métricas, análisis de errores, generalización |
| `06_despliegue/` | interfaz Streamlit |
| `src/` | código compartido y documentado |
| `data/` | instrucciones de descarga, fotos propias, anotaciones |
| `report/` | informe IEEE (LaTeX) |
| `docs/` | formulación matemática, ética, impactos, trazabilidad de la rúbrica |
| `video/` | enlace y guion del video |

## Documentos clave
- [Formulación matemática](docs/formulacion_matematica.md)
- [Definición de clases y protocolo de fotos](docs/definicion_clases.md)
- [EDA multifruta (ejecutado)](02_comprension_datos/eda.ipynb)
- [Trazabilidad de la rúbrica](docs/RUBRICA_TRAZABILIDAD.md)
- [Checklist de los lineamientos](docs/REQUISITOS_LINEAMIENTOS.md)

## Cómo ejecutar
```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
# datos: ver data/README.md
streamlit run 06_despliegue/app.py   # app (ver 06_despliegue/README.md)
```
_(Las instrucciones por fase se completarán a medida que existan los notebooks.)_

## Datos y referencias
- Dataset base: *Fruit Quality Classification*, `ryandpark`, Kaggle (licencia por verificar en su ficha). Ver [`data/README.md`](data/README.md).
- Todo código o dato de terceros se referencia explícitamente en el informe y en el archivo donde se usa.

## Estado
Ver la matriz de trazabilidad para el avance por criterio de la rúbrica. **Entrega: jueves 26 de noviembre de 2026, en la mañana.**
