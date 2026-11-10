"""Utilidades de datos para el proyecto *fruit-quality-banana*.

Cubre la fase 2 de CRISP-DM (comprensión de los datos):

* inventario de imágenes y lectura de la marca de tiempo del nombre de archivo,
* agrupación de imágenes por banana física (para evitar fuga de datos),
* estadísticos de calidad de imagen,
* detección de casi-duplicados con un hash perceptual,
* hojas de contacto para validar visualmente los grupos.

Dataset de referencia: *Fruit Quality Classification* (R. D. Park, Kaggle, CC0),
https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification

Los nombres de archivo del dataset siguen el patrón de la cámara del celular
``IMG<AAAAMMDD><HHMMSS>[_NN].jpg``. La idea de agrupar por cercanía temporal es una
**hipótesis** de este proyecto (no viene documentada en el dataset) y debe validarse
visualmente (ver ``contact_sheet``).
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png"}
_TS_RE = re.compile(r"^IMG(\d{8})(\d{6})(?:_(\d+))?$", re.IGNORECASE)


def parse_timestamp(filename: str) -> tuple[Optional[datetime], int]:
    """Extrae la fecha-hora y el sufijo de duplicado de un nombre de archivo.

    Args:
        filename: nombre con o sin extensión, p. ej. ``IMG20200729190220_01.jpg``.

    Returns:
        ``(timestamp, sufijo)``. ``timestamp`` es ``None`` si el nombre no sigue el
        patrón; ``sufijo`` es 0 si no hay ``_NN`` (p. ej. ``_01`` -> 1).
    """
    match = _TS_RE.match(Path(filename).stem)
    if not match:
        return None, 0
    ts = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    return ts, int(match.group(3)) if match.group(3) else 0


def list_images(folder: Path | str) -> pd.DataFrame:
    """Lista las imágenes de una carpeta (sin recursión) con su marca de tiempo.

    Returns:
        DataFrame con columnas ``path``, ``filename``, ``timestamp`` (``NaT`` si el
        nombre no sigue el patrón) y ``dup_suffix``.

    Raises:
        FileNotFoundError: si la carpeta no existe.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {folder}")
    rows = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            ts, suffix = parse_timestamp(p.name)
            rows.append({"path": str(p), "filename": p.name,
                         "timestamp": ts, "dup_suffix": suffix})
    df = pd.DataFrame(rows, columns=["path", "filename", "timestamp", "dup_suffix"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def consecutive_gaps(df: pd.DataFrame) -> pd.Series:
    """Segundos entre fotos consecutivas (ordenadas por tiempo), útil para elegir el umbral."""
    ts = df["timestamp"].dropna().sort_values()
    return ts.diff().dt.total_seconds().dropna()


def assign_groups(df: pd.DataFrame, gap_seconds: float = 3.0, prefix: str = "") -> pd.DataFrame:
    """Asigna un identificador de grupo (banana física aproximada) por cercanía temporal.

    Se ordenan las imágenes por ``(timestamp, dup_suffix)`` y se abre un grupo nuevo
    cuando el salto con la imagen anterior supera ``gap_seconds``. Las imágenes sin
    marca de tiempo forman cada una su propio grupo (no se puede inferir nada).

    Args:
        df: salida de :func:`list_images` (de **una sola** clase/carpeta: una misma
            banana no puede estar en dos clases).
        gap_seconds: umbral de salto temporal. **Hipótesis a validar** con
            :func:`contact_sheet` y la sensibilidad del número de grupos.
        prefix: prefijo del identificador (p. ej. ``"good_"``).

    Returns:
        Copia de ``df`` ordenada, con columnas ``group`` y ``group_size``.
    """
    out = df.copy()
    out["_has_ts"] = out["timestamp"].notna()
    out = out.sort_values(["_has_ts", "timestamp", "dup_suffix"], ascending=[False, True, True])
    gaps = out["timestamp"].diff().dt.total_seconds()
    new_group = (gaps > gap_seconds) | gaps.isna()
    labels = new_group.cumsum().astype(int)
    out["group"] = [f"{prefix}{g:04d}" for g in labels]
    # sin marca de tiempo: un grupo por imagen
    no_ts = ~out["_has_ts"]
    if no_ts.any():
        out.loc[no_ts, "group"] = [f"{prefix}nots_{i:04d}" for i in range(int(no_ts.sum()))]
    out["group_size"] = out.groupby("group")["path"].transform("size")
    return out.drop(columns="_has_ts").reset_index(drop=True)


def group_sensitivity(df: pd.DataFrame, gaps: Iterable[float] = (1, 2, 3, 5, 10, 30)) -> pd.DataFrame:
    """Número de grupos y tamaño medio de grupo para distintos umbrales de salto."""
    rows = []
    for g in gaps:
        grouped = assign_groups(df, gap_seconds=g)
        sizes = grouped.groupby("group").size()
        rows.append({"gap_seconds": g, "n_groups": int(sizes.size),
                     "mean_size": float(sizes.mean()), "max_size": int(sizes.max())})
    return pd.DataFrame(rows)


def image_stats(path: Path | str) -> dict:
    """Estadísticos simples de una imagen: tamaño, color medio (RGB y HSV) y nitidez.

    La nitidez es la varianza de las diferencias horizontales y verticales de la
    imagen en gris (valores bajos sugieren desenfoque). Se calcula sobre una versión
    reducida a 128 px de lado máximo para ser rápida.
    """
    with Image.open(path) as im:
        width, height = im.size
        im = im.convert("RGB")
        im.thumbnail((128, 128))
        rgb = np.asarray(im, dtype=np.float64)
        hsv = np.asarray(im.convert("HSV"), dtype=np.float64)
    gray = rgb.mean(axis=2)
    sharp = float(np.diff(gray, axis=0).var() + np.diff(gray, axis=1).var())
    return {"width": width, "height": height,
            "r": rgb[..., 0].mean(), "g": rgb[..., 1].mean(), "b": rgb[..., 2].mean(),
            "hue": hsv[..., 0].mean() * 360 / 255, "sat": hsv[..., 1].mean() / 255,
            "val": hsv[..., 2].mean() / 255, "sharpness": sharp}


def ahash(path: Path | str, size: int = 8) -> int:
    """Hash perceptual promedio (aHash) de ``size*size`` bits como entero."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float64)
    bits = (arr > arr.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    """Distancia de Hamming entre dos hashes enteros."""
    return bin(a ^ b).count("1")


def near_duplicates(hashes: dict[str, int], max_dist: int = 2) -> list[tuple[str, str, int]]:
    """Pares de imágenes con hash casi idéntico (distancia de Hamming <= ``max_dist``).

    Complejidad O(n^2): aceptable para ~1 000 imágenes por carpeta.
    """
    keys = list(hashes)
    pairs = []
    for i, ka in enumerate(keys):
        ha = hashes[ka]
        for kb in keys[i + 1:]:
            d = hamming(ha, hashes[kb])
            if d <= max_dist:
                pairs.append((ka, kb, d))
    return pairs


def contact_sheet(paths: list[str], ncols: int = 5, thumb: int = 160) -> Image.Image:
    """Hoja de contacto (cuadrícula de miniaturas) para inspección visual."""
    nrows = int(np.ceil(len(paths) / ncols)) or 1
    sheet = Image.new("RGB", (ncols * thumb, nrows * thumb), (255, 255, 255))
    for k, p in enumerate(paths):
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb, thumb))
            r, c = divmod(k, ncols)
            sheet.paste(im, (c * thumb, r * thumb))
    return sheet
