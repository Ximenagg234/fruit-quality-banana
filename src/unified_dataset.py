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
    * Si difieren (también cuando uno de los dos lo descartó): se usa ``calidad_final`` de ``resolved``
      (``origen = 'resuelto'``); la decisión conjunta prevalece sobre el descarte de una sola persona. Si no está
      resuelto, el ítem **se excluye** (no se adivina).
    * Si los dos lo descartaron, o la etiqueta final es ``Descartar`` o no es válida: se excluye.

    Returns:
        DataFrame con ``id_col``, ``calidad`` y ``origen``.
    """
    m = a[[id_col, "calidad"]].merge(b[[id_col, "calidad"]], on=id_col, suffixes=("_a", "_b"))
    filas = []
    res = {} if resolved is None else dict(zip(resolved[id_col], resolved.get("calidad_final", pd.Series(dtype=str))))
    for r in m.itertuples(index=False):
        ia, ib, ic = getattr(r, id_col), r.calidad_a, r.calidad_b
        if ib == ic:
            if ib in CLASES:
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


def assign_splits(df: pd.DataFrame, kaggle_partition: pd.DataFrame, seed: int = 42,
                  val_size: float = 0.15, test_size: float = 0.15, own_as_test: bool = False) -> pd.DataFrame:
    """Agrega la columna ``split`` (train, val o test) a la tabla unificada.

    * Las filas de **Kaggle conservan exactamente** la partición fija de ``partition.pkl`` (así el test
      de Kaggle sigue intacto y los resultados anteriores siguen siendo comparables).
    * Las filas de **Mixed** y de las **fotos propias** reciben una partición nueva **por grupo**
      (``StratifiedGroupKFold`` estratificado por clase): todas las frutas de una misma foto, o todas las
      fotos de un mismo objeto, caen en una sola partición.
    * Con ``own_as_test=True`` las **fotos propias** no se reparten: todas reciben ``split = 'own_test'``
      (prueba propia, que se usa una sola vez al final) y solo *Mixed* se parte por grupo. El nombre distinto
      evita que se mezclen por descuido con la prueba de Kaggle o de *Mixed* (``'test'``).

    Raises:
        KeyError: si la partición de Kaggle no trae ``path`` y ``split``.
        AssertionError: si algún grupo queda en dos particiones.
    """
    import splits as sp

    if {"path", "split"} - set(kaggle_partition.columns):
        raise KeyError("la partición de Kaggle debe traer 'path' y 'split'")
    out = df.copy()
    out["split"] = out["path"].map(dict(zip(kaggle_partition["path"], kaggle_partition["split"])))
    if own_as_test:
        out.loc[out["source"] == "own", "split"] = "own_test"
    nuevo = out["split"].isna()
    if nuevo.any():
        sub = out[nuevo].reset_index()
        tr, va, te = sp.grouped_train_val_test(sub, label_col="label", group_col="group", val_size=val_size,
                                               test_size=test_size, seed=seed)
        for nombre, parte in (("train", tr), ("val", va), ("test", te)):
            out.loc[parte["index"].to_numpy(), "split"] = nombre
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        sp.assert_no_group_overlap(out[out["split"] == a], out[out["split"] == b], "group")
    return out
