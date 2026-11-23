"""Experimentos de la CNN desde cero (calidad Buena/Mala y especie) sobre el dataset de Kaggle.

Sigue la actividad ``CORTE 3/02-CNN_resuelto`` del curso: una **tabla de experimentos** que
varía una cosa a la vez (``Flatten`` frente a ``GlobalAveragePooling2D``, Dropout, aumento de
datos, L2, número de filtros, tasa de aprendizaje, resolución) y se mide en validación.

Reglas que cumple:
* solo se cargan ``train`` y ``val`` de ``data/processed/partition.pkl``: **el test no se usa**;
* el entrenamiento se submuestrea a <= 1 100 imágenes por celda especie x clase (``val`` no);
* parada temprana sobre la pérdida de validación (``restore_best_weights``): por eso la
  validación queda **ligeramente optimista** y no sustituye al test ni a las fotos propias;
* un experimento de **control "solo borde"** enmascara el centro de la imagen para medir cuánto
  del rendimiento puede venir del fondo (ver ``02_comprension_datos/eda.ipynb`` §7).

Es **reanudable**: cada resultado se agrega al CSV al terminar, y se omiten los ya hechos.

Uso (desde la raíz del repositorio)::

    python 04_modelado/entrenar_cnn.py --tarea calidad
    python 04_modelado/entrenar_cnn.py --tarea especie
    python 04_modelado/entrenar_cnn.py --tarea calidad --humo      # prueba rápida, no escribe resultados

Los tiempos varían con la carga de la máquina; se registran, pero no se deben citar como fijos.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import cnn  # noqa: E402
import features as ft  # noqa: E402
import models_classic as mc  # noqa: E402

PROC = REPO / "data" / "processed"
SEED, CAP, BATCH, MAX_EPOCHS, PATIENCE = 42, 1100, 32, 20, 3

BASE = dict(size=96, head="gap", filters=(32, 64, 128), dense=128, dropout=0.5, l2=0.0,
            augment=False, lr=1e-3, borde=False)
EXPERIMENTOS = [
    dict(BASE, id="E1", desc="Guía del curso: Flatten, Dense 64, sin regularización", head="flatten", dense=64, dropout=0.0),
    dict(BASE, id="E2", desc="E1 + Dropout 0.5", head="flatten", dense=64),
    dict(BASE, id="E3", desc="GAP + Dropout 0.5", head="gap"),
    dict(BASE, id="E4", desc="Flatten + Dropout 0.5 + aumento de datos", head="flatten", augment=True),
    dict(BASE, id="E5", desc="GAP + Dropout 0.5 + aumento de datos", head="gap", augment=True),
    dict(BASE, id="E6", desc="E5 + L2 1e-4", augment=True, l2=1e-4),
    dict(BASE, id="E7", desc="E5 con menos filtros (16-32-64)", augment=True, filters=(16, 32, 64)),
    dict(BASE, id="E8", desc="E5 con tasa de aprendizaje 3e-4", augment=True, lr=3e-4),
    dict(BASE, id="E9", desc="E5 a 128 px", augment=True, size=128),
    dict(BASE, id="E10", desc="CONTROL solo borde: E5 con el centro enmascarado", augment=True, borde=True),
]
EXPERIMENTOS_ESPECIE = [
    dict(BASE, id="S1", desc="Especie: GAP + Dropout 0.5 + aumento de datos", augment=True),
    dict(BASE, id="S2", desc="CONTROL solo borde (especie): centro enmascarado", augment=True, borde=True),
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cargar(size: int, tarea: str, humo: bool):
    """Imágenes (uint8) y etiquetas de train y val. **No carga el test.**"""
    part = pd.read_pickle(PROC / "partition.pkl")
    dev = part[part["split"].isin(["train", "val"])]
    train = mc.cap_per_cell(dev[dev["split"] == "train"], ["species", "label_weak"], CAP, SEED)
    val = dev[dev["split"] == "val"]
    if humo:
        train, val = train.sample(300, random_state=0), val.sample(150, random_state=0)
    cache = PROC / f"cnn{size}_{'humo' if humo else 'completo'}.npz"
    firma = np.concatenate([train["row"].to_numpy(), val["row"].to_numpy()])
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["firma"], firma):
            Xtr, Xva = z["Xtr"], z["Xva"]
        else:
            Xtr = Xva = None
    else:
        Xtr = Xva = None
    if Xtr is None:
        log(f"leyendo y reduciendo imágenes a {size} px ({len(train):,} train, {len(val):,} val)")
        Xtr = ft.thumbs_matrix(train["path"], size=size, workers=8)
        Xva = ft.thumbs_matrix(val["path"], size=size, workers=8)
        np.savez(cache, Xtr=Xtr, Xva=Xva, firma=firma)
    col = "label_weak" if tarea == "calidad" else "species"
    clases = sorted(part[col].dropna().unique()) if tarea == "especie" else ["Buena", "Mala"]
    idx = {c: i for i, c in enumerate(clases)}
    return (Xtr, train[col].map(idx).to_numpy(), Xva, val[col].map(idx).to_numpy(), clases, val["species"].to_numpy())


def enmascarar_centro(X: np.ndarray, size: int) -> np.ndarray:
    """Control "solo borde": reemplaza el interior por gris (queda visible solo el marco de la imagen)."""
    X = X.copy()
    X[:, ft.region_mask(size, "center", frame=size // 8), :] = 127
    return X


def pesos_de_clase(y: np.ndarray, k: int) -> dict:
    """w_k = N / (K * N_k) (docs/formulacion_matematica.md §2)."""
    n = np.bincount(y, minlength=k)
    return {i: float(len(y) / (k * n[i])) for i in range(k)}


def entrenar(exp: dict, datos, tarea: str, humo: bool, salida: Path) -> dict:
    import tensorflow as tf

    Xtr, ytr, Xva, yva, clases, especies_va = datos
    if exp["borde"]:
        Xtr, Xva = enmascarar_centro(Xtr, exp["size"]), enmascarar_centro(Xva, exp["size"])
    tf.keras.utils.set_random_seed(SEED)

    def ds(X, y, barajar):
        d = tf.data.Dataset.from_tensor_slices((X, y))
        if barajar:
            d = d.shuffle(len(X), seed=SEED)
        return (d.batch(BATCH).map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(2))

    model = cnn.build_cnn(exp["size"], len(clases), exp["head"], filters=exp["filters"], dense_units=exp["dense"],
                          dropout=exp["dropout"], l2=exp["l2"], augment=exp["augment"])
    model.compile(tf.keras.optimizers.Adam(exp["lr"]), "sparse_categorical_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    hist = model.fit(ds(Xtr, ytr, True), validation_data=ds(Xva, yva, False), epochs=1 if humo else MAX_EPOCHS,
                     class_weight=pesos_de_clase(ytr, len(clases)), verbose=0,
                     callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True)])
    segundos = time.time() - t0
    nombres = np.array(clases)
    pred_va = nombres[model.predict(ds(Xva, yva, False), verbose=0).argmax(1)]
    pred_tr = nombres[model.predict(ds(Xtr, ytr, False), verbose=0).argmax(1)]
    ev = mc.evaluate(nombres[yva], pred_va, especies_va)
    ev_tr = mc.evaluate(nombres[ytr], pred_tr)
    h = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    mejor = int(np.argmin(h["val_loss"]))
    fila = {"id": exp["id"], "descripcion": exp["desc"], "tarea": tarea, "size": exp["size"], "head": exp["head"],
            "filtros": "-".join(map(str, exp["filters"])), "dense": exp["dense"], "dropout": exp["dropout"], "l2": exp["l2"],
            "aumento": exp["augment"], "lr": exp["lr"], "solo_borde": exp["borde"], "parametros": int(model.count_params()),
            "epocas_corridas": len(h["loss"]), "mejor_epoca": mejor + 1,
            "F1_train": ev_tr["global"]["F1_macro"], "F1_val": ev["global"]["F1_macro"],
            "exactitud_val": ev["global"]["exactitud"], "EE_val": ev["global"]["EE"], "kappa_val": ev["global"]["kappa_w"],
            "brecha_F1": ev_tr["global"]["F1_macro"] - ev["global"]["F1_macro"], "loss_val": h["val_loss"][mejor],
            "segundos": round(segundos, 1), "s_por_epoca": round(segundos / len(h["loss"]), 1)}
    if not humo:
        salida.mkdir(parents=True, exist_ok=True)
        (salida / f"{tarea}_{exp['id']}.json").write_text(json.dumps({"fila": fila, "historia": h,
                                                          "por_especie": ev["per_species"].round(4).to_dict("index")},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
        model.save(salida / f"{tarea}_{exp['id']}.keras")
    return fila


def baselines_especie(salida: Path, humo: bool) -> list:
    """Línea base de ML tradicional para la especie: Random Forest sobre color (imagen completa y solo borde)."""
    from sklearn.ensemble import RandomForestClassifier
    part = pd.read_pickle(PROC / "partition.pkl")
    dev = part[part["split"].isin(["train", "val"])]
    train = mc.cap_per_cell(dev[dev["split"] == "train"], ["species", "label_weak"], CAP, SEED)
    val = dev[dev["split"] == "val"]
    if humo:
        train, val = train.sample(300, random_state=0), val.sample(150, random_state=0)
    thumbs = np.load(PROC / "thumbs64.npy", mmap_mode="r")
    filas = []
    for region in ("full", "border"):
        a = ft.color_features(np.asarray(thumbs[train["row"].to_numpy()]), region)
        b = ft.color_features(np.asarray(thumbs[val["row"].to_numpy()]), region)
        rf = RandomForestClassifier(300, random_state=SEED, n_jobs=-1, class_weight="balanced").fit(a, train["species"])
        ev = mc.evaluate(val["species"].to_numpy(), rf.predict(b), val["species"].to_numpy())
        filas.append({"id": f"RF_{region}", "descripcion": f"Random Forest sobre color, región {region}", "tarea": "especie",
                      "F1_val": ev["global"]["F1_macro"], "exactitud_val": ev["global"]["exactitud"],
                      "EE_val": ev["global"]["EE"], "kappa_val": ev["global"]["kappa_w"]})
    return filas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tarea", choices=["calidad", "especie"], required=True)
    ap.add_argument("--humo", action="store_true", help="prueba rápida con pocos datos; no escribe resultados")
    ap.add_argument("--solo", nargs="*", help="ids de experimentos a correr (por defecto, todos)")
    args = ap.parse_args()
    lista = EXPERIMENTOS if args.tarea == "calidad" else EXPERIMENTOS_ESPECIE
    if args.humo:
        lista = [e for e in lista if e["id"] in (("E1", "E10") if args.tarea == "calidad" else ("S1",))]
    if args.solo:
        lista = [e for e in lista if e["id"] in args.solo]
    csv = REPO / "04_modelado" / f"resultados_cnn_{args.tarea}.csv"
    salida = PROC / "cnn_runs"
    hechos = set(pd.read_csv(csv)["id"]) if (csv.exists() and not args.humo) else set()
    log(f"tarea={args.tarea} | experimentos pendientes: {[e['id'] for e in lista if e['id'] not in hechos]}")
    filas_nuevas = []
    if args.tarea == "especie" and "RF_full" not in hechos:
        filas_nuevas += baselines_especie(salida, args.humo)
        for f in filas_nuevas:
            log(f"  {f['id']}: F1 val {f['F1_val']:.3f}")
    caches = {}
    for exp in lista:
        if exp["id"] in hechos:
            continue
        if exp["size"] not in caches:
            caches = {exp["size"]: cargar(exp["size"], args.tarea, args.humo)}      # una resolución en memoria a la vez
        log(f"=== {exp['id']}: {exp['desc']}")
        fila = entrenar(exp, caches[exp["size"]], args.tarea, args.humo, salida)
        filas_nuevas.append(fila)
        log(f"    F1 val {fila['F1_val']:.3f} | F1 train {fila['F1_train']:.3f} | épocas {fila['epocas_corridas']} | {fila['segundos']:.0f} s")
        if not args.humo:
            previo = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
            pd.concat([previo, pd.DataFrame([fila])], ignore_index=True).to_csv(csv, index=False)
    if not args.humo and filas_nuevas and args.tarea == "especie":
        previo = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
        ids_previos = set(previo.get("id", []))
        nuevos = [f for f in filas_nuevas if f["id"].startswith("RF_") and f["id"] not in ids_previos]
        if nuevos:
            pd.concat([pd.DataFrame(nuevos), previo], ignore_index=True).to_csv(csv, index=False)
    log("terminado")


if __name__ == "__main__":
    main()
