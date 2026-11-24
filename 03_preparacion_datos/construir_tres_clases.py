"""Construye la tabla unificada de tres clases (Buena, Regular, Mala) con su partición.

Une Kaggle (*Good*/*Bad*, etiquetas débiles), los recortes de ``Mixed`` ya etiquetados por dos personas y las
fotos propias con ``quality_final``. Las filas de Kaggle conservan la partición fija; las demás se parten **por
grupo**. Al final imprime cuántas imágenes aporta cada fuente a cada clase y avisa si alguna fuente no aporta las
tres (riesgo de que el modelo aprenda la fuente en lugar de la calidad).

Uso (desde la raíz del repositorio)::

    python 03_preparacion_datos/construir_tres_clases.py \\
        --etiquetas-a data/processed/mixed_crops/etiquetas_mixed_NP.csv \\
        --etiquetas-b data/processed/mixed_crops/etiquetas_mixed_JPB.csv \\
        --desacuerdos desacuerdos.csv \\
        --anotaciones data/own_photos/annotations.csv

Todos los argumentos de Mixed y de fotos propias son opcionales: si faltan, esa fuente se omite.
``desacuerdos.csv`` es el que produce ``calcular_acuerdo.py`` con la columna ``calidad_final`` ya completada.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import unified_dataset as ud  # noqa: E402

PROC = REPO / "data" / "processed"


def construir(args: argparse.Namespace) -> pd.DataFrame:
    part = pd.read_pickle(Path(args.particion))
    frames = [ud.from_kaggle(part[part["split"].isin(["train", "val", "test"])])]
    if args.etiquetas_a and args.etiquetas_b:
        a, b = (pd.read_csv(f, dtype=str, keep_default_na=False) for f in (args.etiquetas_a, args.etiquetas_b))
        resueltos = pd.read_csv(args.desacuerdos, dtype=str, keep_default_na=False) if args.desacuerdos else None
        final = ud.resolve_labels(a, b, resueltos)
        man = pd.read_csv(Path(args.recortes) / "manifiesto.csv")
        frames.append(ud.from_mixed(final, man, args.recortes))
        print(f"Mixed: {len(final)} recortes con etiqueta final de {len(a.merge(b, on='crop_id'))} etiquetados por ambos "
              f"({(final['origen'] == 'resuelto').sum()} resueltos tras un desacuerdo)")
    if args.anotaciones:
        ann = pd.read_csv(args.anotaciones, dtype=str, keep_default_na=False)
        own = ud.from_own(ann, Path(args.anotaciones).parent)
        frames.append(own)
        print(f"Fotos propias: {len(own)} con quality_final")
    df = ud.combine(*frames, lemon_as_lime=args.lemon_como_lima)
    return ud.assign_splits(df, part, seed=args.semilla, own_as_test=args.propias_solo_prueba)


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--particion", default=str(PROC / "partition.pkl"), help="partición fija de Kaggle")
    ap.add_argument("--etiquetas-a")
    ap.add_argument("--etiquetas-b")
    ap.add_argument("--desacuerdos", help="CSV con calidad_final completada")
    ap.add_argument("--recortes", default=str(PROC / "mixed_crops"))
    ap.add_argument("--anotaciones", help="data/own_photos/annotations.csv")
    ap.add_argument("--lemon-como-lima", action="store_true", help="une las especies lemon y lime [DECIDIR]")
    ap.add_argument("--propias-solo-prueba", action="store_true",
                    help="las fotos propias no se reparten: todas son la prueba propia ('own_test'), de uso único")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--salida", default=str(PROC / "tres_clases.pkl"))
    args = ap.parse_args(argv)
    df = construir(args)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.salida)

    print(f"\nTabla unificada: {len(df):,} imágenes -> {args.salida}")
    rep = ud.source_class_report(df)
    print("\nImágenes por fuente y clase:")
    print(rep.to_string())
    incompletas = list(rep.index[~rep["completa"]])
    if incompletas:
        print(f"\nAVISO: fuentes que NO aportan las tres clases: {incompletas}. "
              "Kaggle nunca trae Regular; si Mixed o las fotos propias tampoco aportan las tres, el modelo podría aprender la fuente.")
    print("\nImágenes por especie y clase:")
    print(ud.species_class_report(df).to_string())
    print("\nImágenes por partición y clase:")
    print(pd.crosstab(df["split"], df["label"]).reindex(["train", "val", "test", "own_test"]).dropna(how="all").to_string())
    return df


if __name__ == "__main__":
    main()
