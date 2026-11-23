"""Tabla unificada de tres clases (Buena, Regular, Mala) a partir de todas las fuentes de datos.

Fuentes:
* **kaggle**: carpetas *Good* y *Bad* (etiquetas **débiles**: Good -> Buena, Bad -> Mala). No traen *Regular*.
* **mixed**: recortes de frutas de ``Mixed`` etiquetados a mano por dos personas (``labeling.py``).
* **own**: fotos propias del grupo con ``quality_final`` en ``data/own_photos/annotations.csv``.

Regla de diseño (``docs/definicion_clases.md``): **cada fuente debe aportar las tres clases**. Si *Regular*
viniera solo de una fuente, el modelo podría aprender la fuente (fondo, cámara, resolución) en lugar de la
calidad; :func:`source_class_report` lo detecta. Todas las tablas comparten las columnas de ``COLUMNS``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

import labeling as lb

COLUMNS = ["path", "species", "label", "source", "group", "label_origin"]
CLASES = lb.CLASES                                     # Buena, Regular, Mala


def from_kaggle(partition: pd.DataFrame) -> pd.DataFrame:
    """Filas de Kaggle (*Good* y *Bad*) desde la partición fija (``partition.pkl``), con etiqueta débil.

    Raises:
        KeyError: si faltan columnas de la partición.
    """
    need = {"path", "species", "label_weak", "group"}
    if need - set(partition.columns):
        raise KeyError(f"faltan columnas en la partición: {sorted(need - set(partition.columns))}")
    d = partition[partition["label_weak"].isin(CLASES)]
    return pd.DataFrame({"path": d["path"], "species": d["species"], "label": d["label_weak"], "source": "kaggle",
                         "group": d["group"], "label_origin": "debil_kaggle"}).reset_index(drop=True)


def resolve_labels(a: pd.DataFrame, b: pd.DataFrame, resolved: Optional[pd.DataFrame] = None,
                   id_col: str = "crop_id") -> pd.DataFrame:
    """Etiqueta final de cada ítem a partir de dos anotadores y de los desacuerdos ya resueltos.

    * Si los dos coinciden en Buena, Regular o Mala: esa es la etiqueta (``origen = 'acuerdo'``).
    * Si difieren: se usa ``calidad_final`` de ``resolved`` (``origen = 'resuelto'``); si no está resuelto, el
      ítem **se excluye** (no se adivina).
    * Si alguno lo descartó, o la etiqueta final es ``Descartar``: se excluye.

    Returns:
        DataFrame con ``id_col``, ``calidad`` y ``origen``.
    """
    m = a[[id_col, "calidad"]].merge(b[[id_col, "calidad"]], on=id_col, suffixes=("_a", "_b"))
    filas = []
    res = {} if resolved is None else dict(zip(resolved[id_col], resolved.get("calidad_final", pd.Series(dtype=str))))
    for r in m.itertuples(index=False):
        ia, ib, ic = getattr(r, id_col), r.calidad_a, r.calidad_b
        if lb.DESCARTAR in (ic, ib):
            continue
        if ib == ic:
            filas.append({id_col: ia, "calidad": ib, "origen": "acuerdo"})
        elif res.get(ia) in CLASES:
            filas.append({id_col: ia, "calidad": res[ia], "origen": "resuelto"})
    return pd.DataFrame(filas, columns=[id_col, "calidad", "origen"])


def from_mixed(final: pd.DataFrame, manifest: pd.DataFrame, crops_dir: Path | str) -> pd.DataFrame:
    """Recortes de ``Mixed`` con etiqueta final, con la ruta del recorte y el grupo de su foto de origen.

    Todas las frutas de una misma foto comparten ``group`` (así no se separan entre train y test).

    Raises:
        KeyError: si falta ``group``, ``species`` o ``crop_id`` en el manifiesto.
    """
    crops_dir = Path(crops_dir)
    m = final.merge(manifest[["crop_id", "species", "group"]], on="crop_id", how="inner")
    return pd.DataFrame({"path": [str(crops_dir / "recortes" / s / f"{c}.jpg") for c, s in zip(m["crop_id"], m["species"])],
                         "species": m["species"], "label": m["calidad"], "source": "mixed", "group": m["group"],
                         "label_origin": "manual_" + m["origen"]}).reset_index(drop=True)


def from_own(annotations: pd.DataFrame, photos_dir: Path | str) -> pd.DataFrame:
    """Fotos propias con ``quality_final`` (se ignoran las filas sin etiqueta final).

    El ``group`` es el ``object_id``: todas las fotos de un mismo objeto van juntas a una partición.

    Raises:
        KeyError: si faltan ``image_id``, ``species``, ``object_id`` o ``quality_final``.
    """
    need = {"image_id", "species", "object_id", "quality_final"}
    if need - set(annotations.columns):
        raise KeyError(f"faltan columnas en annotations.csv: {sorted(need - set(annotations.columns))}")
    d = annotations[annotations["quality_final"].isin(CLASES)]
    photos_dir = Path(photos_dir)
    return pd.DataFrame({"path": [str(photos_dir / i) for i in d["image_id"]], "species": d["species"].str.lower(),
                         "label": d["quality_final"], "source": "own", "group": "own_" + d["object_id"].astype(str),
                         "label_origin": "manual_grupo"}).reset_index(drop=True)


def combine(*frames: pd.DataFrame, lemon_as_lime: bool = False) -> pd.DataFrame:
    """Une las fuentes en una sola tabla (columnas de ``COLUMNS``).

    Args:
        lemon_as_lime: si es ``True``, la especie ``lemon`` (así se llama la lima en *Mixed*) se une con ``lime``
            (así se llama en *Good*/*Bad*). Es una decisión del grupo **[DECIDIR]**; por defecto no se unen.

    Raises:
        ValueError: si hay etiquetas fuera de Buena, Regular y Mala, o una ruta repetida.
    """
    df = pd.concat([f[COLUMNS] for f in frames], ignore_index=True)
    if not df["label"].isin(CLASES).all():
        raise ValueError("hay etiquetas que no son Buena, Regular ni Mala")
    if df["path"].duplicated().any():
        raise ValueError(f"rutas repetidas, p. ej. {df.loc[df['path'].duplicated(), 'path'].iloc[0]}")
    if lemon_as_lime:
        df = df.assign(species=df["species"].replace({"lemon": "lime"}))
    return df


def source_class_report(df: pd.DataFrame, min_per_class: int = 5) -> pd.DataFrame:
    """Imágenes por fuente y clase, con la bandera ``completa`` (cada clase con al menos ``min_per_class``).

    Una fuente incompleta (p. ej. *Regular* solo en una fuente) expone al modelo a aprender la fuente en
    lugar de la calidad; hay que completarla o reportar el riesgo.
    """
    t = pd.crosstab(df["source"], df["label"]).reindex(columns=CLASES, fill_value=0)
    t["completa"] = (t[CLASES] >= min_per_class).all(axis=1)
    return t


def species_class_report(df: pd.DataFrame, min_per_class: int = 5) -> pd.DataFrame:
    """Imágenes por especie y clase (cada especie debería tener las tres clases para evaluar por especie)."""
    t = pd.crosstab(df["species"], df["label"]).reindex(columns=CLASES, fill_value=0)
    t["completa"] = (t[CLASES] >= min_per_class).all(axis=1)
    return t
