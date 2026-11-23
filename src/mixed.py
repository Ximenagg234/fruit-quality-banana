"""Segmentación de frutas individuales en las imágenes de ``Mixed`` (varias frutas por foto).

Los lineamientos (§2.1) piden "segmentar individualmente cada fruta presente en las
imágenes" de ``Mixed``. El método es clásico (sin aprendizaje):

1. máscara de primer plano por saturación (fondos neutros) o por distancia al color del
   fondo (fondos lisos de color, como cian o morado); se elige según la saturación del marco,
2. separación de frutas que se tocan con la **transformada de distancia** y ``watershed``,
3. filtro por área, y recorte cuadrado alrededor de cada fruta.

``Mixed`` no trae etiquetas por fruta ni fondo uniforme (hay pasto, periódico, mesas): el
método **no es perfecto** y no hay verdad de terreno con la que medir su exactitud. Cada
recorte lleva banderas (área, solidez, contacto con el borde) para filtrarlo, y la validación
es visual (ver ``03_preparacion_datos/segmentar_mixed.py``). La transformada de distancia y el
``watershed`` **no se vieron en clase** (se declaran en el informe).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2hsv
from skimage.feature import peak_local_max
from skimage.measure import label as sk_label
from skimage.measure import regionprops
from skimage.segmentation import watershed

import segmentation as sg

#: Si la saturación media del marco supera este valor, el fondo se considera de color y se usa
#: el método ``distance`` (con fondos neutros se usa ``saturation``).
BORDER_SAT_LIMIT = 0.25


def border_saturation(img: np.ndarray, frame: int = 4) -> float:
    """Saturación HSV media del marco de ``frame`` píxeles de la imagen."""
    h, w = img.shape[:2]
    frame = max(1, min(frame, h // 4, w // 4))
    inner = np.zeros((h, w), dtype=bool)
    inner[frame:h - frame, frame:w - frame] = True
    return float(rgb2hsv(img)[..., 1][~inner].mean())


def choose_method(img: np.ndarray) -> str:
    """``'distance'`` si el fondo es de color (marco muy saturado) y ``'saturation'`` si es neutro."""
    return "distance" if border_saturation(img) > BORDER_SAT_LIMIT else "saturation"


def extract_instances(img: np.ndarray, method: str = "auto", min_area_frac: float = 0.004,
                      max_area_frac: float = 0.5, min_distance: int = 6) -> tuple[list[dict], str]:
    """Frutas individuales de una imagen.

    Args:
        img: RGB ``uint8`` ``(h, w, 3)``.
        method: ``'auto'``, ``'saturation'`` o ``'distance'`` (ver :func:`choose_method`).
        min_area_frac: área mínima de una fruta, como fracción de la imagen (descarta ruido).
        max_area_frac: área máxima (descarta fondos mal segmentados que cubren casi toda la imagen).
        min_distance: distancia mínima entre centros de frutas al separarlas con ``watershed``.

    Returns:
        ``(instancias, método_usado)``. Cada instancia es un diccionario con ``bbox``
        ``(x0, y0, x1, y1)``, ``area_frac``, ``solidity`` (área / área de la envolvente convexa;
        valores bajos delatan frutas fundidas o máscaras irregulares) y ``touches_border``.
    """
    used = choose_method(img) if method == "auto" else method
    mask = sg.foreground_mask(img, method=used)
    if mask is None:
        return [], used
    mask = ndimage.binary_fill_holes(mask)
    dist = ndimage.gaussian_filter(ndimage.distance_transform_edt(mask), 1.0)
    comps = sk_label(mask)
    peaks = peak_local_max(dist, min_distance=min_distance, threshold_abs=2.0, labels=comps, exclude_border=False)
    if len(peaks) == 0:
        labels = comps
    else:
        markers = np.zeros(mask.shape, dtype=np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        labels = watershed(-dist, markers, mask=mask)
    h, w = mask.shape
    bg = sg.estimate_background(img)
    val = img.max(axis=2).astype(np.float64) / 255.0                 # componente V de HSV
    out = []
    for r in regionprops(labels):
        frac = r.area / mask.size
        if not (min_area_frac <= frac <= max_area_frac):
            continue
        y0, x0, y1, x1 = r.bbox
        ys, xs = r.coords[:, 0], r.coords[:, 1]
        color = img[ys, xs].astype(np.float64).mean(axis=0)
        bw, bh = x1 - x0, y1 - y0
        out.append({"bbox": (int(x0), int(y0), int(x1), int(y1)), "area_frac": float(frac), "solidity": float(r.solidity),
                    "touches_border": bool(x0 == 0 or y0 == 0 or x1 == w or y1 == h),
                    "mean_v": float(val[ys, xs].mean()), "bg_dist": float(np.linalg.norm(color - bg)),
                    "aspect": float(max(bw, bh) / max(1, min(bw, bh))), "min_side": int(min(bw, bh))})
    out.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))          # orden estable: de arriba abajo e izquierda a derecha
    return out, used


def crop_square(img: np.ndarray, bbox: tuple[int, int, int, int], pad: float = 0.2, out_size: int = 128) -> Image.Image:
    """Recorte cuadrado centrado en la fruta, con margen ``pad`` y reescalado a ``out_size`` px.

    Se usan los píxeles originales (no la máscara) para que quien etiqueta vea el contexto. Si el
    recorte se sale de la imagen, la ventana se desplaza para que quede dentro (sin rellenar).
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    lado = int(round(max(x1 - x0, y1 - y0) * (1 + 2 * pad)))
    lado = max(16, min(lado, h, w))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    ax = int(np.clip(cx - lado // 2, 0, w - lado))
    ay = int(np.clip(cy - lado // 2, 0, h - lado))
    return Image.fromarray(img[ay:ay + lado, ax:ax + lado]).resize((out_size, out_size), Image.LANCZOS)


def quality_flags(inst: dict, solidity_min: float = 0.85, min_area_frac: float = 0.006, min_v: float = 0.25,
                  min_bg_dist: float = 45.0, max_aspect: float = 5.0, min_side: int = 12) -> dict:
    """Banderas para decidir qué recortes se etiquetan.

    ``compacta``: solidez alta (no es una mancha irregular ni frutas fundidas); ``no_borde``: no toca el
    borde de la imagen; ``grande``: área suficiente; ``color_ok``: no es casi negro y su color difiere del
    del fondo (descarta franjas oscuras y parches de fondo); ``forma_ok``: proporción y tamaño razonables.
    """
    return {"compacta": inst["solidity"] >= solidity_min, "no_borde": not inst["touches_border"],
            "grande": inst["area_frac"] >= min_area_frac,
            "color_ok": inst["mean_v"] >= min_v and inst["bg_dist"] >= min_bg_dist,
            "forma_ok": inst["aspect"] <= max_aspect and inst["min_side"] >= min_side}


def pick_label_sample(manifest, per_species: int = 60, max_per_image: int = 2, max_inst_per_image: int = 12,
                      seed: int = 42):
    """Muestra reproducible de recortes para etiquetar.

    Toma ``per_species`` recortes por especie, a lo sumo ``max_per_image`` de una misma foto (para
    diversificar), solo recortes que pasan todas las banderas de :func:`quality_flags` y solo de fotos con
    a lo sumo ``max_inst_per_image`` recuadros (más suele indicar un fondo con mucho ruido). Devuelve el
    manifiesto filtrado con la columna ``orden`` (orden de presentación, igual para todos los anotadores).

    Raises:
        KeyError: si faltan columnas del manifiesto.
    """
    import pandas as pd
    flags = ["compacta", "no_borde", "grande", "color_ok", "forma_ok"]
    need = {"crop_id", "species", "source_image", "n_inst_imagen", *flags}
    faltan = need - set(manifest.columns)
    if faltan:
        raise KeyError(f"faltan columnas en el manifiesto: {sorted(faltan)}")
    buenos = manifest[manifest[flags].all(axis=1) & (manifest["n_inst_imagen"] <= max_inst_per_image)]
    partes = []
    for _, g in buenos.groupby("species", sort=True):
        g = g.sample(frac=1.0, random_state=seed)                     # baraja
        g = g.groupby("source_image", sort=False).head(max_per_image)  # a lo sumo N por foto
        partes.append(g.head(per_species))
    muestra = pd.concat(partes) if partes else buenos.iloc[0:0]
    muestra = muestra.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    muestra["orden"] = np.arange(1, len(muestra) + 1)
    return muestra
