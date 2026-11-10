# Datos

Las imágenes **no se suben al repositorio** (excepto las fotos propias, que son pocas). Ver `.gitignore`.

## 1. Dataset de referencia (obligatorio, a citar)
**Fruit Quality Classification**, autor `ryandpark`, Kaggle, licencia CC0.
https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification

- Según su ficha: 19 526 imágenes; clases de calidad Bad / Good / Mixed; 6 frutas (apple, banana, guava, lime, orange, pomegranate).
- Tamaños: Bad y Good 256×256; Mixed 256×192. Cada fruta fotografiada desde varios ángulos.
- `Mixed` contiene **varias frutas por imagen**: hay que segmentarlas individualmente.

Descarga (requiere cuenta de Kaggle y su CLI):

```bash
pip install kaggle
kaggle datasets download -d ryandpark/fruit-quality-classification -p data/raw --unzip
```

> Pendiente en el EDA (`02_comprension_datos/`): confirmar los nombres reales de las carpetas, cuántas imágenes de banana hay por clase, y si existe algún identificador que permita saber qué imágenes son de la misma fruta.

## 2. Fotos propias (30–50, mínimo exigido)
Se guardan en `data/own_photos/` junto con `annotations.csv`.

Esquema de `annotations.csv` (una fila por foto):

| Columna | Descripción |
|---|---|
| `image_id` | nombre del archivo |
| `banana_id` | identificador de la banana física (para separar train/test por banana) |
| `day` | día de la sesión (0, 2, 4, 6…) |
| `angle` | ángulo (frente, arriba, atrás…) |
| `quality_a`, `quality_b` | etiqueta de cada anotador (Buena/Regular/Mala) |
| `quality_final` | etiqueta tras resolver desacuerdos |
| `size_class` | Pequeño/Mediano/Grande |
| `stain_pct` | fracción de mancha calculada ($\rho$), opcional |
| `photographer` | quién la tomó |
| `notes` | observaciones (iluminación, golpe, moho…) |

## 3. Anotaciones compartidas con la clase
**[DECIDIR con el profesor/otros grupos]** mecanismo para compartir (carpeta en la nube o repositorio común).
