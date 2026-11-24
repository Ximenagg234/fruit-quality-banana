"""Valida la longitud normalizada ℓ contra la medida real en cm de las fotos propias.

ℓ = 4 sqrt(λ1) / min(H, W) (``src/segmentation.py``) es una medida **relativa**: depende de la distancia de la cámara a la fruta.
Aquí se compara con ``max_dim_cm`` (lado más largo medido con regla, ``data/own_photos/annotations.csv``) en las 69 fotos propias.
Si ℓ midiera tamaño real, ℓ y los cm irían juntos; la distancia de la cámara y la mano rompen esa relación, y eso es lo que se mide.

Salida (desde la raíz del repositorio: ``python 05_evaluacion/validar_tamano.py``): ``05_evaluacion/validacion_tamano.csv`` (una fila por
foto) y ``05_evaluacion/validacion_tamano_resumen.csv`` (correlación de Spearman por subconjunto, con n).
Las fotos propias no están en el repositorio: se leen de la carpeta indicada (por defecto la de trabajo local).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import inference as inf  # noqa: E402

FOTOS = REPO.parent / "etiquetado_23nov" / "fotos_para_etiquetar" / "recortes" / "foto"


def main(fotos: Path = FOTOS) -> None:
    """Calcula ℓ de cada foto propia y su correlación con la medida en cm."""
    ann = pd.read_csv(REPO / "data" / "own_photos" / "annotations.csv", dtype=str)
    filas = []
    for r in ann.itertuples():
        im = ImageOps.exif_transpose(Image.open(fotos / r.image_id)).convert("RGB")
        m = inf.measure_size(im)
        filas.append({"image_id": r.image_id, "especie": r.species, "object_id": r.object_id, "en_mano": int(r.in_hand),
                      "cm": float(r.max_dim_cm), "ell": np.nan if m is None else m["ell"],
                      "toca_borde": np.nan if m is None else m["border_frac"] > inf.HAND_BORDER_FRAC})
    d = pd.DataFrame(filas)
    d.to_csv(REPO / "05_evaluacion" / "validacion_tamano.csv", index=False)
    res = []
    subconjuntos = {"todas": d, "en mano": d[d["en_mano"] == 1], "sin mano": d[d["en_mano"] == 0]}
    for esp in sorted(d["especie"].unique()):
        subconjuntos[f"especie {esp}"] = d[d["especie"] == esp]
    for nombre, s in subconjuntos.items():
        s = s.dropna(subset=["ell"])
        rho, p = spearmanr(s["cm"], s["ell"]) if len(s) > 3 and s["cm"].nunique() > 1 else (np.nan, np.nan)
        res.append({"subconjunto": nombre, "n": len(s), "spearman": round(float(rho), 3), "p": round(float(p), 4),
                    "sin_fruta_detectada": int(d.shape[0] - d.dropna(subset=["ell"]).shape[0]) if nombre == "todas" else ""})
    pd.DataFrame(res).to_csv(REPO / "05_evaluacion" / "validacion_tamano_resumen.csv", index=False)
    print(pd.DataFrame(res).to_string(index=False))


if __name__ == "__main__":
    main()
