# fruit-quality-banana

Sistema de clasificación de **calidad** (Buena / Regular / Mala) y estimación de **tamaño** de frutas de varias especies (manzana, banano, guayaba, lima, naranja y granada) a partir de imágenes estáticas, con una app en Streamlit (carga de imagen o cámara), para el proyecto final de **Algoritmos y Programación III (2026-2)**, Universidad Icesi. Metodología: **CRISP-DM**.

## Entregables
- **Informe IEEE (7 páginas):** [`report/informe.pdf`](report/informe.pdf)
- **Video (7:17):** https://drive.google.com/file/d/1tPYpLZFJcIF8_H9Ii96Xt_JxhAxQ07Hp/view?usp=sharing
- **App:** [`06_despliegue/`](06_despliegue/README.md)

## Equipo
| Integrante | GitHub | Rol principal |
|---|---|---|
| Ximena Gómez Gómez | @Ximenagg234 | Integra el repositorio; datos, modelos y app |
| Natalia Delgado | @ndelgadoar | Informe y referencias; etiquetado |
| Juan Pablo Bello | @Bellito29 | Diagrama, video y revisión; etiquetado |

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
| `video/` | enlace del video |

## Documentos clave
- [Formulación matemática](docs/formulacion_matematica.md)
- [Reparto del trabajo y flujo de colaboración](docs/REPARTO_TRABAJO.md)
- [Estrategia de solución y alternativas](docs/estrategia_solucion.md)
- [Diagrama CRISP-DM](docs/diagrama_crispdm.svg)
- [Estructura del informe y referencias verificadas](report/ESTRUCTURA_INFORME.md)
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
Por fase (desde la raíz, con el entorno activado; los datos de Kaggle se descargan según `data/README.md`):
```bash
python 03_preparacion_datos/construir_tres_clases.py --help    # tabla unificada de tres clases y su partición
python 04_modelado/entrenar_tres_clases.py --fase clasicos     # ML tradicional (fases: datos, clasicos, cnn, test)
python 04_modelado/analisis_modelos.py                         # tabla train/val/test, matrices y errores
python 05_evaluacion/medir_costo.py                            # hardware, tiempos y energía estimada
python report/generar_tablas.py                                # tablas del informe con los números finales
python -m pytest tests -q                                      # pruebas
```
Los modelos entrenados y los datos procesados no se versionan (salvo el modelo de la app); las etiquetas finales sí (`data/mixed_labels/`, `data/own_photos/annotations.csv`).

## Datos y referencias
- Dataset base: *Fruit Quality Classification*, `ryandpark`, Kaggle (licencia CC0, dominio público, verificada en su ficha). Ver [`data/README.md`](data/README.md).
- Todo código o dato de terceros se referencia explícitamente en el informe y en el archivo donde se usa.

## Resultados principales (tres clases)
- **Validación** (Kaggle y *Mixed*): la CNN T6 llega a F1 macro 0.725 frente a 0.257 de la línea base; es la que usa la app.
- **Prueba** (Kaggle y *Mixed*, usada una sola vez): el SVM llega a 0.774 y la CNN T6 a 0.714.
- **Fotos propias** (69 fotos, 34 frutas): F1 macro 0.439 con la CNN T6; **ningún modelo detecta *Regular*** (0 de 14) y ninguno supera la exactitud de la línea base. La generalización a fotos reales es pobre y se declara.
- **Tamaño:** la longitud normalizada es una medida relativa (correlación débil con la medida real en cm).
- Detalle: `04_modelado/ANALISIS_ERRORES.md`, `report/tablas_informe.md` y `docs/estrategia_solucion.md`.

## Estado
Ver la matriz de trazabilidad para el avance por criterio de la rúbrica. **Entrega: jueves 26 de noviembre de 2026, en la mañana.**
