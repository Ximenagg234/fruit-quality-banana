"""Agrupación de imágenes por **objeto físico** para evitar fuga de datos.

El dataset fotografía cada fruta desde varios ángulos y en ráfagas, así que repartir
las imágenes al azar entre train y test infla las métricas (ver
``docs/formulacion_matematica.md`` §7). Como solo parte de los nombres de archivo trae
fecha, se combinan dos señales dentro de cada celda ``(especie, fuente)`` (un mismo
objeto no puede estar en dos especies ni en dos clases):

1. **Tiempo:** fotos consecutivas separadas por no más de ``gap_seconds``.
2. **Hash perceptual:** imágenes casi idénticas (distancia de Hamming <= ``max_hash_dist``).

Los grupos son los componentes conexos del grafo que forman ambas señales.

Ambos umbrales son **hipótesis**: se calibran con la sensibilidad
(:func:`grouping_sensitivity`) y con inspección visual, no se fijan a ojo. Limitación
conocida: versiones rotadas, recortadas o aumentadas de una misma foto pueden tener
un hash distinto y quedar en grupos separados (fuga residual).

Técnicas no vistas en clase (declarar en el informe): hash perceptual y componentes
conexos de un grafo (``scipy.sparse.csgraph``).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

#: Umbrales elegidos tras la calibración sobre el dataset real
#: (``02_comprension_datos/eda.ipynb``). Con hash de 256 bits, las fotos
#: consecutivas (<= 3 s) tienen una distancia mediana de 8 y los pares al azar de la
#: misma celda de 114; con ``max_hash_dist >= 10`` la clausura transitiva forma
#: grupos gigantes (hasta 2 957 imágenes con 40) que rompen la estratificación.
DEFAULT_GAP_SECONDS = 3.0
DEFAULT_HASH_DIST = 5


def packed_ahash(path: Path | str, size: int = 16) -> np.ndarray:
    """Hash perceptual promedio (aHash) de ``size*size`` bits, empaquetado en bytes.

    Se reduce la imagen a ``size x size`` en gris y cada píxel aporta un bit
    (mayor que la media o no). Con ``size=16`` son 256 bits (32 bytes).
    """
    with Image.open(path) as im:
        im.draft("L", (size * 4, size * 4))      # acelera la lectura de JPEG grandes
        arr = np.asarray(im.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float64)
    return np.packbits((arr > arr.mean()).ravel())


def hash_matrix(paths: Iterable[Path | str], size: int = 16, workers: int = 8) -> np.ndarray:
    """Matriz ``(n, size*size/8)`` de hashes empaquetados, uno por imagen y en orden."""
    if (size * size) % 8:
        raise ValueError("size*size debe ser múltiplo de 8")
    paths = list(paths)
    if not paths:
        return np.empty((0, size * size // 8), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(lambda p: packed_ahash(p, size), paths))
    return np.vstack(rows)


def hash_edges(packed: np.ndarray, max_dist: int, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Pares ``(i, j)`` con ``i < j`` y distancia de Hamming <= ``max_dist``.

    Calcula todas las distancias por bloques de ``chunk`` filas (O(n^2) en tiempo pero
    memoria acotada: ``chunk * n * bytes_por_hash``).
    """
    n = len(packed)
    rows, cols = [], []
    for start in range(0, n, chunk):
        block = packed[start:start + chunk]
        dist = np.bitwise_count(block[:, None, :] ^ packed[None, :, :]).sum(axis=2, dtype=np.int32)
        bi, bj = np.nonzero(dist <= max_dist)
        bi = bi + start
        keep = bi < bj
        rows.append(bi[keep])
        cols.append(bj[keep])
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(rows), np.concatenate(cols)


def time_edges(timestamps: pd.Series, gap_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Pares de imágenes consecutivas en el tiempo separadas por <= ``gap_seconds``.

    Las posiciones devueltas son índices posicionales en ``timestamps``. Las
    imágenes sin fecha (``NaT``) no generan aristas.
    """
    ts = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    order = ts[ts.notna()].sort_values(kind="stable").index.to_numpy()
    if len(order) < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    gaps = ts.loc[order].diff().dt.total_seconds().to_numpy()[1:]
    close = gaps <= gap_seconds
    return order[:-1][close], order[1:][close]


def build_groups(df: pd.DataFrame, packed: Optional[np.ndarray] = None,
                 gap_seconds: Optional[float] = 3.0, max_hash_dist: Optional[int] = None,
                 cell_cols: tuple[str, ...] = ("species", "source"),
                 prefix: str = "g") -> pd.DataFrame:
    """Asigna ``group`` (objeto físico aproximado) combinando tiempo y hash.

    Args:
        df: índice con ``timestamp`` y las columnas de ``cell_cols`` (p. ej. la salida
            de ``data_utils.index_dataset``).
        packed: hashes empaquetados alineados con las filas de ``df`` (de
            :func:`hash_matrix`); si es ``None`` no se usa el hash.
        gap_seconds: salto temporal máximo; ``None`` desactiva la señal de tiempo.
        max_hash_dist: distancia de Hamming máxima; ``None`` desactiva el hash.
        cell_cols: solo se unen imágenes de una misma celda.
        prefix: prefijo del identificador de grupo.

    Returns:
        Copia de ``df`` (índice reiniciado) con ``group`` y ``group_size``.
    """
    df = df.reset_index(drop=True)
    if packed is not None and len(packed) != len(df):
        raise ValueError("packed debe tener una fila por cada fila de df")
    rows, cols = [np.empty(0, dtype=np.int64)], [np.empty(0, dtype=np.int64)]
    for idx in df.groupby(list(cell_cols), sort=False).indices.values():
        idx = np.asarray(idx)
        if gap_seconds is not None:
            i, j = time_edges(df.loc[idx, "timestamp"], gap_seconds)
            rows.append(idx[i]); cols.append(idx[j])
        if packed is not None and max_hash_dist is not None and len(idx) > 1:
            i, j = hash_edges(packed[idx], max_hash_dist)
            rows.append(idx[i]); cols.append(idx[j])
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(len(df), len(df)))
    _, labels = connected_components(graph, directed=False)
    out = df.copy()
    out["group"] = [f"{prefix}{k:05d}" for k in labels]
    out["group_size"] = out.groupby("group")["group"].transform("size")
    return out


def grouping_sensitivity(df: pd.DataFrame, packed: Optional[np.ndarray],
                         settings: Iterable[tuple[Optional[float], Optional[int]]],
                         cell_cols: tuple[str, ...] = ("species", "source")) -> pd.DataFrame:
    """Efecto de cada ``(gap_seconds, max_hash_dist)`` en el número y tamaño de los grupos.

    Un umbral demasiado laxo funde objetos distintos en grupos gigantes (``max_size``
    alto, poca estratificación posible); uno demasiado estricto deja fuga de datos.
    """
    rows = []
    for gap, dist in settings:
        g = build_groups(df, packed, gap_seconds=gap, max_hash_dist=dist, cell_cols=cell_cols)
        sizes = g.groupby("group").size()
        rows.append({"gap_seconds": gap, "max_hash_dist": dist, "n_groups": int(sizes.size),
                     "mean_size": float(sizes.mean()), "max_size": int(sizes.max()),
                     "pct_imgs_in_multi": float((g["group_size"] > 1).mean() * 100)})
    return pd.DataFrame(rows)
