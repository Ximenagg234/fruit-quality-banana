"""Lógica del etiquetado manual independiente y del acuerdo entre anotadores.

Se usa para etiquetar los recortes de ``Mixed`` (``03_preparacion_datos/etiquetador_mixed.py``) y
sirve igual para las fotos propias. Cada persona etiqueta **sin ver** las etiquetas de la otra y
guarda su propio archivo (``etiquetas_mixed_<INICIALES>.csv``); después :func:`acuerdo` mide el
acuerdo con el **kappa de Cohen ponderado cuadráticamente** (las clases son ordinales:
Buena < Regular < Mala) y lista los desacuerdos para resolverlos.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import cohen_kappa_score

CLASES = ["Buena", "Regular", "Mala"]          # orden ordinal
DESCARTAR = "Descartar"                          # no es una fruta, o el recorte no sirve
VALIDAS = CLASES + [DESCARTAR]


def normalizar_iniciales(texto: str) -> str:
    """Iniciales en mayúsculas y solo letras o números (para el nombre del archivo).

    Raises:
        ValueError: si no queda ningún carácter válido.
    """
    limpio = re.sub(r"[^A-Za-z0-9]", "", texto or "").upper()
    if not limpio:
        raise ValueError("escribe tus iniciales (letras o números)")
    return limpio


def ruta_etiquetas(carpeta: Path | str, iniciales: str, prefijo: str = "etiquetas_mixed") -> Path:
    """Ruta del archivo de etiquetas de una persona."""
    return Path(carpeta) / f"{prefijo}_{normalizar_iniciales(iniciales)}.csv"


def cargar_muestra(carpeta: Path | str) -> pd.DataFrame:
    """Muestra de recortes a etiquetar, en el orden de presentación (igual para todos).

    Raises:
        FileNotFoundError: si falta ``muestra_etiquetado.csv`` en la carpeta.
    """
    ruta = Path(carpeta) / "muestra_etiquetado.csv"
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró {ruta}")
    return pd.read_csv(ruta).sort_values("orden").reset_index(drop=True)


def cargar_etiquetas(ruta: Path | str, id_col: str = "crop_id") -> dict[str, dict]:
    """Etiquetas ya guardadas: ``{id: {'calidad': ..., 'notas': ...}}`` (vacío si el archivo no existe)."""
    ruta = Path(ruta)
    if not ruta.is_file():
        return {}
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    return {r[id_col]: {"calidad": r["calidad"], "notas": r.get("notas", "")} for _, r in df.iterrows()}


def _escribir(ruta: Path, etiquetas: dict, id_col: str) -> None:
    filas = [{id_col: k, "calidad": v["calidad"], "notas": v.get("notas", "")} for k, v in etiquetas.items()]
    tmp = ruta.with_suffix(".tmp")
    pd.DataFrame(filas, columns=[id_col, "calidad", "notas"]).to_csv(tmp, index=False)
    tmp.replace(ruta)                                            # escritura atómica: no deja archivos a medias


def guardar_etiqueta(ruta: Path | str, item_id: str, calidad: str, notas: str = "", id_col: str = "crop_id") -> None:
    """Guarda o reemplaza la etiqueta de un recorte (el archivo se reescribe de forma atómica).

    Raises:
        ValueError: si ``calidad`` no es Buena, Regular, Mala ni Descartar.
    """
    if calidad not in VALIDAS:
        raise ValueError(f"calidad debe ser una de {VALIDAS}")
    ruta = Path(ruta)
    etiquetas = cargar_etiquetas(ruta, id_col)
    etiquetas[item_id] = {"calidad": calidad, "notas": (notas or "").strip()}
    _escribir(ruta, etiquetas, id_col)


def borrar_etiqueta(ruta: Path | str, item_id: str, id_col: str = "crop_id") -> None:
    """Quita la etiqueta de un recorte (para rehacerla). No falla si no existía."""
    ruta = Path(ruta)
    etiquetas = cargar_etiquetas(ruta, id_col)
    if etiquetas.pop(item_id, None) is not None:
        _escribir(ruta, etiquetas, id_col)


def primer_pendiente(muestra: pd.DataFrame, etiquetas: dict, id_col: str = "crop_id") -> Optional[int]:
    """Posición (0-based) del primer recorte sin etiquetar, o ``None`` si ya están todos."""
    for pos, item in enumerate(muestra[id_col]):
        if item not in etiquetas:
            return pos
    return None


def progreso(muestra: pd.DataFrame, etiquetas: dict, id_col: str = "crop_id") -> tuple[int, int]:
    """``(etiquetados, total)``."""
    return sum(1 for i in muestra[id_col] if i in etiquetas), len(muestra)


def acuerdo(a: pd.DataFrame, b: pd.DataFrame, id_col: str = "crop_id") -> dict:
    """Acuerdo entre dos anotadores.

    Args:
        a, b: tablas con las columnas ``id_col`` y ``calidad`` de cada persona.

    Returns:
        ``n_comunes`` (ítems etiquetados por ambos), ``n_validos`` (sin ``Descartar``), ``concordancia``
        (proporción de coincidencias exactas), ``kappa_w`` (kappa ponderado cuadráticamente, ``nan`` si
        no se puede calcular), ``matriz`` (confusión A x B) y ``desacuerdos`` (ítems en los que difieren,
        incluidos los descartados por solo uno de los dos).
    """
    m = a[[id_col, "calidad"]].merge(b[[id_col, "calidad"]], on=id_col, suffixes=("_a", "_b"))
    validos = m[(m["calidad_a"].isin(CLASES)) & (m["calidad_b"].isin(CLASES))]
    kappa = float("nan")
    if len(validos) and validos[["calidad_a", "calidad_b"]].stack().nunique() > 1:
        kappa = float(cohen_kappa_score(validos["calidad_a"], validos["calidad_b"], labels=CLASES, weights="quadratic"))
    matriz = pd.crosstab(m["calidad_a"], m["calidad_b"]).reindex(index=VALIDAS, columns=VALIDAS, fill_value=0)
    return {"n_comunes": len(m), "n_validos": len(validos),
            "concordancia": float((m["calidad_a"] == m["calidad_b"]).mean()) if len(m) else float("nan"),
            "kappa_w": kappa, "matriz": matriz,
            "desacuerdos": m[m["calidad_a"] != m["calidad_b"]].reset_index(drop=True)}
