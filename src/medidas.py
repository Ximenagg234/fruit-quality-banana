"""Registro de la medida real de cada fruta (lado más largo, en cm) de las fotos propias.

La medida se toma con regla sobre la fruta física y sirve para **validar** la longitud normalizada ℓ
(``docs/definicion_clases.md`` §3) y como columna ``max_dim_cm`` de ``data/own_photos/annotations.csv``.
Cada archivo ``medidas_fotos_<INICIALES>.csv`` tiene una fila por foto: ``image_id``, ``max_dim_cm`` (vacío si no
se pudo medir) y ``notas``. La escritura es atómica para no dejar archivos a medias.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

import labeling as lb

#: Rango plausible (cm) para el lado más largo de una fruta: protege contra errores de digitación.
MIN_CM, MAX_CM = 1.0, 40.0
COLUMNAS = ["image_id", "max_dim_cm", "notas"]


def parsear_cm(texto: str) -> float:
    """Convierte ``'12,5'`` o ``'12.5'`` a ``12.5``.

    Raises:
        ValueError: si no es un número o queda fuera de ``[MIN_CM, MAX_CM]``.
    """
    limpio = (texto or "").strip().lower().replace("cm", "").strip().replace(",", ".")
    try:
        valor = float(limpio)
    except ValueError:
        raise ValueError(f"'{texto}' no es un número (escribe, por ejemplo, 12.5)") from None
    if not MIN_CM <= valor <= MAX_CM:
        raise ValueError(f"{valor:g} cm está fuera del rango esperado ({MIN_CM:g} a {MAX_CM:g} cm)")
    return round(valor, 1)


def ruta_medidas(carpeta: Path | str, iniciales: str, prefijo: str = "medidas_fotos") -> Path:
    """Ruta del archivo de medidas de una persona."""
    return Path(carpeta) / f"{prefijo}_{lb.normalizar_iniciales(iniciales)}.csv"


def cargar_medidas(ruta: Path | str) -> dict[str, dict]:
    """Medidas ya guardadas: ``{image_id: {'max_dim_cm': str, 'notas': str}}`` (vacío si no existe el archivo)."""
    ruta = Path(ruta)
    if not ruta.is_file():
        return {}
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    return {r["image_id"]: {"max_dim_cm": r["max_dim_cm"], "notas": r.get("notas", "")} for _, r in df.iterrows()}


def _escribir(ruta: Path, medidas: dict) -> None:
    filas = [{"image_id": k, "max_dim_cm": v["max_dim_cm"], "notas": v.get("notas", "")} for k, v in medidas.items()]
    tmp = ruta.with_suffix(".tmp")
    pd.DataFrame(filas, columns=COLUMNAS).to_csv(tmp, index=False)
    tmp.replace(ruta)


def guardar_medida(ruta: Path | str, image_id: str, cm: Optional[float], notas: str = "") -> None:
    """Guarda o reemplaza la medida de una foto. ``cm=None`` registra que no se pudo medir.

    Raises:
        ValueError: si ``cm`` está fuera del rango esperado.
    """
    if cm is not None and not MIN_CM <= cm <= MAX_CM:
        raise ValueError(f"{cm:g} cm está fuera del rango esperado ({MIN_CM:g} a {MAX_CM:g} cm)")
    ruta = Path(ruta)
    medidas = cargar_medidas(ruta)
    medidas[image_id] = {"max_dim_cm": "" if cm is None else f"{cm:g}", "notas": (notas or "").strip()}
    _escribir(ruta, medidas)


def primer_pendiente(ids: list[str], medidas: dict) -> Optional[int]:
    """Posición (0-based) de la primera foto sin registro (medida o 'no medida'), o ``None``."""
    for pos, i in enumerate(ids):
        if i not in medidas:
            return pos
    return None
