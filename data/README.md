# Datos

Las imágenes **no se suben al repositorio** (excepto las fotos propias, ya reducidas y sin metadatos). Ver `.gitignore`.

## 1. Dataset de referencia (obligatorio, a citar)
**Fruit Quality Classification**, autor `ryandpark`, Kaggle.
https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
*Licencia: **CC0 (dominio público)**, verificada en el campo "License" de la ficha de Kaggle el 22 de noviembre de 2026. Autor: Ryan Park. La fecha exacta de publicación o actualización **[VERIFICAR]** en la ficha para la referencia IEEE.*

- Según su ficha: 19 526 imágenes; clases de calidad Bad / Good / Mixed; 6 frutas (apple, banana, guava, lime, orange, pomegranate).
- Cada fruta fotografiada desde varios ángulos.
- `Mixed` contiene **varias frutas por imagen**: hay que segmentarlas individualmente. En `Mixed` la lima aparece como `Lemon`.

Descarga (requiere cuenta de Kaggle y su CLI):

```bash
pip install kaggle
kaggle datasets download -d ryandpark/fruit-quality-classification -p data/raw --unzip
```

Estructura esperada (el nombre `Qualit` es el original):

```
data/raw/Dataset Fruit/
├── Good Quality_Fruits/<Fruta>_Good/
├── Bad Quality_Fruits/<Fruta>_Bad/
└── Mixed Qualit_Fruits/<Fruta>/
```

Lo medido sobre este dataset (inventario, convenciones de nombre, tamaños, agrupación por objeto y prueba de atajos) está en `02_comprension_datos/eda.ipynb`. El índice y los hashes se generan en `data/processed/` (no versionado).

## 2. Fotos propias (30–50 exigidas por los lineamientos)
Se guardan en `data/own_photos/` junto con `annotations.csv`. Protocolo de captura en `docs/definicion_clases.md` §4.

Esquema de `annotations.csv` (una fila por foto):

| Columna | Descripción |
|---|---|
| `image_id` | nombre del archivo (`<codigo>_d<dia>_<angulo>.jpg`) |
| `species` | especie (apple, banana, guava, lime, orange, pomegranate) |
| `object_id` | identificador del objeto físico (para separar train/test por objeto) |
| `day` | día de la sesión (0, 2, 4, 6…) |
| `angle` | ángulo (frente, arriba, atrás…) |
| `background` | fondo (pared, cartulina…) |
| `in_hand` | 1 si la fruta está sostenida en la mano, 0 si está sobre una superficie |
| `quality_a`, `quality_b` | etiqueta de cada anotador, de forma independiente (Buena/Regular/Mala) |
| `quality_final` | etiqueta tras resolver desacuerdos |
| `size_class` | Pequeño/Mediano/Grande |
| `max_dim_cm` | lado más largo medido con regla, en cm (si se midió) |
| `stain_pct` | fracción de superficie con defecto ($\rho$), opcional |
| `photographer` | quién la tomó |
| `notes` | observaciones (iluminación, golpe, moho…) |

## 3. Anotaciones compartidas con la clase
**[DECIDIR con el profesor/otros grupos]** mecanismo para compartir (carpeta en la nube o repositorio común). El esquema de arriba es el que se compartiría.

## 4. Otros datasets (solo si se aprueban; todo se cita)
Candidatos revisados en sus páginas oficiales, **no descargados**:
- *AgriFreshNET Freshness and Shelf-Life Image Dataset*, Mendeley Data, DOI 10.17632/42m5tb7yv9.1, CC BY 4.0: tres niveles (Fresh / Semi-Fresh / Rotten); falta confirmar especies y tamaño del archivo.
- *Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality*, Mendeley Data, DOI 10.17632/bdd69gyhv8.1, CC BY 4.0: solo fresca/podrida (usar únicamente las imágenes originales, no las aumentadas).
