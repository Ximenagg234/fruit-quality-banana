"""Preparación de las fotos propias: reducir, quitar metadatos, renombrar y registrar.

Medida de privacidad E1 de ``docs/etica_tabla.md``: ninguna foto sube al repositorio
con metadatos (EXIF, GPS, miniaturas incrustadas). Pasos por foto:

1. aplicar la orientación EXIF (para que la imagen no quede girada al borrar la etiqueta),
2. reducir el lado mayor a ``max_side`` píxeles (nunca se amplía),
3. guardar como JPEG **sin** EXIF ni perfil ICC,
4. comprobar con :func:`has_metadata` que el archivo resultante está limpio.

El mapa de fotos (CSV) lo llenan las personas del grupo: ``original`` (nombre del
archivo de origen) e ``image_id`` (nombre final, p. ej. ``A03_d0_top.jpg``) son
obligatorias; el resto de columnas de ``ANNOTATION_COLUMNS`` son opcionales. Los
originales no se tocan ni se suben.

Uso::

    python src/prepare_photos.py <carpeta_origen> <mapa.csv> data/own_photos
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

#: Columnas de ``data/own_photos/annotations.csv`` (ver ``data/README.md``).
ANNOTATION_COLUMNS = ["image_id", "species", "object_id", "day", "angle", "background", "in_hand",
                      "quality_a", "quality_b", "quality_final", "size_class", "max_dim_cm",
                      "stain_pct", "photographer", "notes"]
REQUIRED_MAP_COLUMNS = ["original", "image_id"]


def has_metadata(path: Path | str) -> bool:
    """``True`` si la imagen conserva EXIF, perfil ICC o información adicional.

    Un JPEG guardado por Pillow sin ``exif=`` ni ``icc_profile=`` no debe tener ninguna.
    """
    with Image.open(path) as im:
        extra = {k for k in im.info if k not in {"jfif", "jfif_version", "jfif_unit", "jfif_density",
                                                 "dpi", "progressive", "progression"}}
        return bool(im.getexif()) or bool(extra)


def prepare_photo(src: Path | str, dst: Path | str, max_side: int = 1024, quality: int = 90) -> tuple[int, int]:
    """Reduce una foto, quita sus metadatos y la guarda como JPEG.

    Returns:
        ``(ancho, alto)`` de la imagen resultante.

    Raises:
        RuntimeError: si el archivo guardado aún conserva metadatos.
    """
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")      # orientación primero, luego se descarta el EXIF
        if max(im.size) > max_side:
            scale = max_side / max(im.size)
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        size = im.size
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, format="JPEG", quality=quality, optimize=True)   # sin exif= ni icc_profile=
    if has_metadata(dst):
        raise RuntimeError(f"La imagen guardada aún tiene metadatos: {dst}")
    return size


def fill_image_ids(mapa: pd.DataFrame) -> pd.DataFrame:
    """Genera ``image_id`` donde está vacío, como ``<object_id>_d<day>_<angle>.jpg``.

    Si dos fotos darían el mismo nombre, se agrega ``_2``, ``_3``… Los ``image_id`` ya escritos se
    respetan. Cada fila sin ``image_id`` necesita ``object_id``; ``day`` vale 0 y ``angle`` vale ``x``
    si están vacíos.

    Raises:
        ValueError: si una fila sin ``image_id`` no tiene ``object_id``.
    """
    mapa = mapa.copy()
    if "image_id" not in mapa.columns:
        mapa["image_id"] = ""
    usados = set(mapa.loc[mapa["image_id"] != "", "image_id"])
    for i in mapa.index[mapa["image_id"] == ""]:
        objeto = mapa.at[i, "object_id"].strip() if "object_id" in mapa.columns else ""
        if not objeto:
            raise ValueError(f"La foto {mapa.at[i, 'original']!r} no tiene image_id ni object_id")
        dia = (mapa.at[i, "day"].strip() if "day" in mapa.columns else "") or "0"
        angulo = (mapa.at[i, "angle"].strip() if "angle" in mapa.columns else "") or "x"
        base = f"{objeto}_d{dia}_{angulo}"
        nombre, k = f"{base}.jpg", 1
        while nombre in usados:
            k += 1
            nombre = f"{base}_{k}.jpg"
        usados.add(nombre)
        mapa.at[i, "image_id"] = nombre
    return mapa


def load_photo_map(map_csv: Path | str) -> pd.DataFrame:
    """Lee el mapa de fotos, completa los ``image_id`` vacíos y valida columnas, duplicados y extensiones.

    Raises:
        ValueError: si faltan columnas, hay ``image_id`` repetidos u originales repetidos.
    """
    mapa = pd.read_csv(map_csv, dtype=str, keep_default_na=False)
    if "original" not in mapa.columns:
        raise ValueError("Faltan columnas obligatorias en el mapa: ['original']")
    mapa = fill_image_ids(mapa)
    faltan = [c for c in REQUIRED_MAP_COLUMNS if c not in mapa.columns]
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias en el mapa: {faltan}")
    for col in REQUIRED_MAP_COLUMNS:
        if mapa[col].duplicated().any():
            raise ValueError(f"Valores repetidos en '{col}': {sorted(mapa.loc[mapa[col].duplicated(), col])[:3]}")
    if not mapa["image_id"].str.lower().str.endswith((".jpg", ".jpeg")).all():
        raise ValueError("Todos los image_id deben terminar en .jpg o .jpeg")
    return mapa


def prepare_folder(src_dir: Path | str, map_csv: Path | str, dst_dir: Path | str,
                   max_side: int = 1024, quality: int = 90) -> pd.DataFrame:
    """Procesa todas las fotos del mapa y escribe ``annotations.csv`` en ``dst_dir``.

    Si el archivo ya existe, se agregan las filas nuevas y se rechaza cualquier
    ``image_id`` que ya estuviera registrado. Las columnas que el mapa no trae quedan vacías.

    Returns:
        DataFrame con las filas agregadas a ``annotations.csv``.

    Raises:
        FileNotFoundError: si alguna foto del mapa no está en ``src_dir``.
        ValueError: si el mapa es inválido o repite un ``image_id`` ya registrado.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    mapa = load_photo_map(map_csv)
    ausentes = [o for o in mapa["original"] if not (src_dir / o).is_file()]
    if ausentes:
        raise FileNotFoundError(f"No se encontraron {len(ausentes)} fotos del mapa, p. ej. {ausentes[:3]}")
    ann_path = dst_dir / "annotations.csv"
    previas = pd.read_csv(ann_path, dtype=str, keep_default_na=False) if ann_path.exists() else \
        pd.DataFrame(columns=ANNOTATION_COLUMNS)
    repetidas = set(mapa["image_id"]) & set(previas["image_id"])
    if repetidas:
        raise ValueError(f"image_id ya registrados en annotations.csv: {sorted(repetidas)[:3]}")
    for fila in mapa.itertuples(index=False):
        prepare_photo(src_dir / fila.original, dst_dir / fila.image_id, max_side, quality)
    nuevas = mapa.reindex(columns=ANNOTATION_COLUMNS, fill_value="")
    pd.concat([previas.reindex(columns=ANNOTATION_COLUMNS), nuevas]).to_csv(ann_path, index=False)
    return nuevas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src_dir", help="carpeta con las fotos originales")
    ap.add_argument("map_csv", help="CSV con las columnas original, image_id (y las de annotations.csv)")
    ap.add_argument("dst_dir", help="destino, normalmente data/own_photos")
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args()
    nuevas = prepare_folder(args.src_dir, args.map_csv, args.dst_dir, args.max_side)
    print(f"{len(nuevas)} fotos preparadas en {args.dst_dir} (sin metadatos verificado)")


if __name__ == "__main__":
    main()
