"""Segmenta las frutas individuales de ``Mixed`` y prepara la muestra para etiquetar a mano.

Para cada imagen de ``Mixed`` (varias frutas por foto, sin etiquetas) detecta las frutas con
``src/mixed.py``, guarda un recorte cuadrado de cada una y escribe un **manifiesto** con su
posición, métricas y banderas de calidad. Luego elige una **muestra reproducible** (por defecto
60 recortes por especie) para que las personas del grupo la etiqueten de forma independiente
con ``03_preparacion_datos/etiquetador_mixed.py``.

Cada recorte hereda el ``group`` de su foto de origen (calculado como en el EDA), de modo que todas
las frutas de una misma foto queden en la misma partición al entrenar (sin fuga de datos).

Uso (desde la raíz del repositorio)::

    python 03_preparacion_datos/segmentar_mixed.py                 # salida en data/processed/mixed_crops
    python 03_preparacion_datos/segmentar_mixed.py --por-especie 80

La validación de la segmentación es **visual**: ``Mixed`` no tiene verdad de terreno. Los recortes
con ruido se descartan al etiquetar (botón "Descartar"); las frutas verdes sobre pasto y los
bananos sobre vegetación son los casos más difíciles.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import data_utils as du  # noqa: E402
import groups as gr  # noqa: E402
import mixed as mx  # noqa: E402

RAW = REPO / "data" / "raw" / "Dataset Fruit"
PROC = REPO / "data" / "processed"


def main() -> None:
    """Línea de comandos: segmenta las frutas de *Mixed* y prepara la muestra para etiquetar."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--salida", default=str(PROC / "mixed_crops"))
    ap.add_argument("--por-especie", type=int, default=60, help="recortes a etiquetar por especie")
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()
    salida = Path(args.salida)

    df = du.index_dataset(RAW).reset_index(drop=True)
    packed = np.load(PROC / "hashes16.npy")
    if len(packed) != len(df):
        sys.exit("hashes16.npy no coincide con el índice; ejecuta primero 02_comprension_datos/eda.ipynb")
    g = gr.build_groups(df, packed, gap_seconds=gr.DEFAULT_GAP_SECONDS, max_hash_dist=gr.DEFAULT_HASH_DIST,
                        counter_hash_dist=gr.DEFAULT_COUNTER_HASH_DIST)
    mix = g[g["source"] == "mixed"]
    print(f"{len(mix):,} imágenes de Mixed en {mix['group'].nunique()} grupos")

    (salida / "recortes").mkdir(parents=True, exist_ok=True)
    filas = []
    for fila in mix.itertuples(index=False):
        with Image.open(fila.path) as im:
            arr = np.asarray(im.convert("RGB"))
        inst, metodo = mx.extract_instances(arr)
        stem = Path(fila.filename).stem
        for k, ins in enumerate(inst):
            crop_id = f"{fila.species}_{stem}_{k:02d}"
            ruta = salida / "recortes" / fila.species / f"{crop_id}.jpg"
            ruta.parent.mkdir(parents=True, exist_ok=True)
            mx.crop_square(arr, ins["bbox"]).save(ruta, quality=92)
            x0, y0, x1, y1 = ins["bbox"]
            filas.append({"crop_id": crop_id, "species": fila.species, "source_image": fila.filename, "k": k,
                          "x0": x0, "y0": y0, "x1": x1, "y1": y1, "area_frac": round(ins["area_frac"], 5),
                          "solidity": round(ins["solidity"], 4), "touches_border": ins["touches_border"],
                          "mean_v": round(ins["mean_v"], 4), "bg_dist": round(ins["bg_dist"], 2),
                          "aspect": round(ins["aspect"], 3), "min_side": ins["min_side"], "method": metodo,
                          "n_inst_imagen": len(inst), "group": fila.group, **mx.quality_flags(ins)})
    man = pd.DataFrame(filas)
    man.to_csv(salida / "manifiesto.csv", index=False)

    muestra = mx.pick_label_sample(man, per_species=args.por_especie, seed=args.semilla)
    muestra.to_csv(salida / "muestra_etiquetado.csv", index=False)
    (salida / "fuentes").mkdir(exist_ok=True)
    for _, r in muestra.drop_duplicates("source_image")[["species", "source_image"]].iterrows():
        origen = next(RAW.glob(f"Mixed*/{r.species.capitalize()}/{r.source_image}"), None)
        if origen is not None:
            shutil.copy2(origen, salida / "fuentes" / f"{r.species}__{r.source_image}")

    resumen = man.groupby("species").agg(imagenes=("source_image", "nunique"), recortes=("crop_id", "size"),
                                         buenos=("crop_id", lambda s: int(man.loc[s.index, ["compacta", "no_borde", "grande", "color_ok", "forma_ok"]].all(axis=1).sum())))
    resumen["en_muestra"] = muestra.groupby("species").size()
    print(resumen.fillna(0).astype(int).to_string())
    print(f"\ntotal: {len(man):,} recortes | muestra para etiquetar: {len(muestra)} | salida: {salida}")


if __name__ == "__main__":
    main()
