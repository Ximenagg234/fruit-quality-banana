"""Acuerdo entre dos anotadores (kappa ponderado) y lista de desacuerdos a resolver.

Sirve para los recortes de ``Mixed`` y para las fotos propias: cada archivo tiene una columna con el
identificador (``crop_id`` o ``image_id``) y la columna ``calidad`` (Buena, Regular, Mala o Descartar).

Uso (desde la raíz del repositorio)::

    python 03_preparacion_datos/calcular_acuerdo.py etiquetas_mixed_NP.csv etiquetas_mixed_JP.csv
    python 03_preparacion_datos/calcular_acuerdo.py a.csv b.csv --id image_id --salida desacuerdos.csv

El kappa ponderado cuadráticamente trata las clases como ordinales (Buena < Regular < Mala): confundir
Buena con Mala pesa más que confundir Buena con Regular. Con pocos ítems es inestable: se reporta junto
con la concordancia y el número de ítems.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import labeling as lb  # noqa: E402


def main() -> None:
    """Línea de comandos: imprime el acuerdo entre dos anotadores y guarda los desacuerdos."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archivo_a")
    ap.add_argument("archivo_b")
    ap.add_argument("--id", default="crop_id", help="columna identificadora (crop_id o image_id)")
    ap.add_argument("--salida", default="desacuerdos.csv", help="CSV con los ítems en que difieren")
    args = ap.parse_args()
    a, b = (pd.read_csv(f, dtype=str, keep_default_na=False) for f in (args.archivo_a, args.archivo_b))
    for nombre, t in ((args.archivo_a, a), (args.archivo_b, b)):
        if args.id not in t.columns or "calidad" not in t.columns:
            sys.exit(f"{nombre} debe tener las columnas '{args.id}' y 'calidad'")
    r = lb.acuerdo(a, b, id_col=args.id)
    print(f"ítems etiquetados por ambos: {r['n_comunes']} (sin 'Descartar': {r['n_validos']})")
    print(f"concordancia exacta: {r['concordancia']:.1%}")
    print(f"kappa ponderado cuadrático: {r['kappa_w']:.3f}" if r["kappa_w"] == r["kappa_w"] else
          "kappa ponderado cuadrático: no definido (una sola clase o sin ítems en común)")
    print("\nMatriz de confusión (filas: A, columnas: B):")
    print(r["matriz"].to_string())
    r["desacuerdos"].rename(columns={"calidad_a": "calidad_A", "calidad_b": "calidad_B"}).assign(calidad_final="") \
        .to_csv(args.salida, index=False)
    print(f"\n{len(r['desacuerdos'])} desacuerdos guardados en {args.salida} (completar 'calidad_final' entre las dos personas)")


if __name__ == "__main__":
    main()
