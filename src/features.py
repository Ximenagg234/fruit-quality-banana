"""Miniaturas y características de color por región de la imagen.

Se usan en dos lugares:

* el EDA, para medir si el **fondo** delata la clase (prueba del atajo: un modelo que
  solo ve el borde de la imagen no debería distinguir Buena de Mala),
* la preparación de datos, como base de las características para los modelos clásicos.

Las imágenes se reducen a una miniatura cuadrada sin conservar la proporción
(suficiente para estadísticos de color; no para medir tamaño). Todos los estadísticos
se calculan por imagen, sin información de otras imágenes, de modo que no hay fuga
entre particiones.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image

#: Nombres de las 8 características de :func:`color_stats`.
COLOR_FEATURES = ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std", "sat_mean", "val_mean"]


def load_thumb(path: Path | str, size: int = 64) -> np.ndarray:
    """Miniatura RGB ``(size, size, 3)`` en ``uint8``, sin conservar la proporción."""
    with Image.open(path) as im:
        im.draft("RGB", (size * 4, size * 4))       # acelera JPEG grandes
        return np.asarray(im.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.uint8)


def thumbs_matrix(paths: Iterable[Path | str], size: int = 64, workers: int = 8) -> np.ndarray:
    """Miniaturas de todas las imágenes: arreglo ``(n, size, size, 3)`` en ``uint8``."""
    paths = list(paths)
    if not paths:
        return np.empty((0, size, size, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return np.stack(list(ex.map(lambda p: load_thumb(p, size), paths)))


def region_mask(size: int, region: str, frame: int = 8) -> np.ndarray:
    """Máscara booleana ``(size, size)`` de una región.

    ``'border'``: marco de ``frame`` píxeles de ancho; ``'center'``: el resto
    (interior); ``'full'``: toda la imagen.

    Raises:
        ValueError: si la región no existe o el marco no deja interior.
    """
    if region == "full":
        return np.ones((size, size), dtype=bool)
    if region not in ("border", "center"):
        raise ValueError(f"región desconocida: {region!r}")
    if not 0 < frame < size // 2:
        raise ValueError("el marco debe ser positivo y dejar un interior")
    inner = np.zeros((size, size), dtype=bool)
    inner[frame:size - frame, frame:size - frame] = True
    return ~inner if region == "border" else inner


def color_stats(thumbs: np.ndarray, region: str = "full", frame: int = 8, chunk: int = 2000) -> np.ndarray:
    """Ocho estadísticos de color por imagen en la región indicada.

    Orden (``COLOR_FEATURES``): media y desviación de R, G, B (en [0, 1]) y media de
    saturación y de valor (HSV). No se usa el tono medio porque es una variable circular.

    Args:
        thumbs: arreglo ``(n, s, s, 3)`` ``uint8`` de :func:`thumbs_matrix`.
        region: ``'full'``, ``'border'`` o ``'center'``.
        frame: ancho del marco en píxeles (para ``border`` y ``center``).
        chunk: imágenes por bloque, para acotar la memoria.

    Returns:
        Arreglo ``(n, 8)`` de ``float64``.
    """
    n, size = thumbs.shape[0], thumbs.shape[1]
    mask = region_mask(size, region, frame)
    out = np.empty((n, len(COLOR_FEATURES)), dtype=np.float64)
    for start in range(0, n, chunk):
        px = thumbs[start:start + chunk][:, mask, :].astype(np.float64) / 255.0   # (b, m, 3)
        hsv = rgb_to_hsv(px)
        out[start:start + chunk] = np.concatenate(
            [px.mean(axis=1), px.std(axis=1), hsv[..., 1].mean(axis=1, keepdims=True),
             hsv[..., 2].mean(axis=1, keepdims=True)], axis=1)
    return out
