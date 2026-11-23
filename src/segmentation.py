"""Segmentación de la fruta sobre fondo simple y medición de tamaño.

Método clásico (sin aprendizaje), pensado para fondo liso:

1. el color del fondo se estima con la mediana de los píxeles del marco de la imagen,
2. se mide la distancia de cada píxel a ese color,
3. se binariza con el umbral de Otsu,
4. se limpia con operaciones morfológicas y se conserva el objeto conectado más grande.

El tamaño se mide con la longitud normalizada ``ell = 4 * sqrt(lambda_1) / W`` (ver
``docs/formulacion_matematica.md`` §3.1): ``lambda_1`` es el mayor valor propio de la
covarianza de las coordenadas de la máscara, así que ``4 * sqrt(lambda_1)`` es el eje
mayor de una elipse (y el diámetro de un disco). Es una medida **relativa**: depende de la
distancia cámara-fruta.

Limitaciones que se declaran: una mano o una sombra con color distinto al fondo se
incluyen en la máscara y alteran la medida; varias frutas se tratan como un solo
objeto si se tocan. Segmentación por umbral de Otsu, operaciones morfológicas y PCA de
la máscara **no se vieron en clase** (PCA sí, ver ``02_actividad_PCA``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu


def estimate_background(img: np.ndarray, frame: int = 6) -> np.ndarray:
    """Color de fondo RGB: mediana de los píxeles del marco de ``frame`` píxeles."""
    h, w = img.shape[:2]
    frame = max(1, min(frame, h // 4, w // 4))
    inner = np.zeros((h, w), dtype=bool)
    inner[frame:h - frame, frame:w - frame] = True
    return np.median(img[~inner].reshape(-1, 3), axis=0)


def segment_fruit(img: np.ndarray, frame: int = 6, min_area_frac: float = 0.01) -> Optional[np.ndarray]:
    """Máscara booleana ``(h, w)`` de la fruta, o ``None`` si no se encuentra un objeto.

    Args:
        img: imagen RGB ``uint8`` ``(h, w, 3)``.
        frame: ancho del marco que se usa para estimar el fondo.
        min_area_frac: fracción mínima de la imagen que debe ocupar el objeto.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img debe ser RGB (h, w, 3)")
    bg = estimate_background(img, frame)
    dist = np.linalg.norm(img.astype(np.float64) - bg, axis=2)
    if dist.max() - dist.min() < 1e-6:                      # imagen uniforme: no hay objeto
        return None
    mask = dist > threshold_otsu(dist)
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    mask = ndimage.binary_fill_holes(labels == (int(np.argmax(sizes)) + 1))
    return mask if mask.mean() >= min_area_frac else None


def mask_geometry(mask: np.ndarray) -> dict:
    """Geometría de una máscara por PCA de las coordenadas de sus píxeles.

    Returns:
        ``area_frac`` (fracción de la imagen), ``length_px`` (``4*sqrt(lambda_1)``, eje
        mayor), ``width_px`` (``4*sqrt(lambda_2)``, eje menor) y ``ell`` (longitud
        normalizada por el ancho de la imagen).

    Raises:
        ValueError: si la máscara tiene menos de 2 píxeles.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        raise ValueError("la máscara debe tener al menos 2 píxeles")
    cov = np.cov(np.vstack([xs, ys]), bias=True)           # (1/|P|) sum (p-c)(p-c)^T
    l2, l1 = np.linalg.eigvalsh(cov)                       # ascendentes: l1 es el mayor
    length, width = 4 * np.sqrt(max(l1, 0.0)), 4 * np.sqrt(max(l2, 0.0))
    return {"area_frac": float(mask.mean()), "length_px": float(length), "width_px": float(width),
            "ell": float(length / mask.shape[1])}


def size_class(ell: float, tau1: float, tau2: float) -> str:
    """Clase de tamaño por umbrales: ``Pequeño`` si ``ell < tau1``, ``Mediano`` si ``< tau2``, si no ``Grande``.

    Los umbrales son los terciles de ``ell`` **de la especie, calculados solo con
    entrenamiento** (``docs/formulacion_matematica.md`` §3.1).
    """
    if tau1 > tau2:
        raise ValueError("tau1 no puede ser mayor que tau2")
    return "Pequeño" if ell < tau1 else ("Mediano" if ell < tau2 else "Grande")


def terciles(values) -> tuple[float, float]:
    """Cuantiles 1/3 y 2/3 de ``ell`` (umbrales de tamaño); usar solo valores de entrenamiento."""
    v = np.asarray(values, dtype=float)
    return float(np.quantile(v, 1 / 3)), float(np.quantile(v, 2 / 3))
