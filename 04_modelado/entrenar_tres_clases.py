"""Entrenamiento y evaluación con tres clases (Buena, Regular, Mala) sobre la tabla unificada.

Usa ``data/processed/tres_clases.pkl`` (de ``03_preparacion_datos/construir_tres_clases.py --propias-solo-prueba``):

* ``train`` y ``val``: Kaggle (partición fija) y recortes de *Mixed* (partición por grupo);
* ``test``: la prueba de Kaggle y de *Mixed*;
* ``own_test``: las 69 fotos propias, **solo prueba**, de uso único.

Reglas que cumple:

* el entrenamiento se submuestrea a <= 1 100 imágenes por celda especie x clase (``val`` y pruebas no);
* *Regular* existe solo en *Mixed* (57 imágenes en entrenamiento): todos los modelos usan
  ``class_weight='balanced'`` (pesos w_k = N / (K N_k));
* los modelos se **eligen con la validación**; la prueba se evalúa **una sola vez** (fase ``test``), con todos los
  modelos ya declarados (línea base, regresión logística, SVM, Random Forest, control "solo borde" y dos CNN), y el
  script se niega a repetirla;
* el F1 macro de cada conjunto se calcula sobre las clases presentes en la verdad (la prueba de Kaggle no trae
  *Regular*), y el kappa ponderado usa el orden Buena < Regular < Mala.

Fases (desde la raíz del repositorio)::

    python 04_modelado/entrenar_tres_clases.py --fase datos       # lee y reduce las imágenes (una vez)
    python 04_modelado/entrenar_tres_clases.py --fase clasicos    # ML tradicional con GridSearchCV agrupado
    python 04_modelado/entrenar_tres_clases.py --fase cnn         # CNN desde cero (dos configuraciones)
    python 04_modelado/entrenar_tres_clases.py --fase test        # prueba, una sola vez
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import zlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GridSearchCV

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "04_modelado"))
import features as ft  # noqa: E402
import models_classic as mc  # noqa: E402

PROC = REPO / "data" / "processed"
MODELOS = PROC / "modelos_tres"
CLASES = mc.ORDEN_CALIDAD
SEED, CAP = 42, 1100
FOTOS_PROPIAS = REPO.parent / "etiquetado_23nov" / "fotos_para_etiquetar" / "recortes" / "foto"
CSV_VAL = REPO / "04_modelado" / "resultados_tres_clases.csv"
CSV_TEST = REPO / "04_modelado" / "resultados_tres_clases_test.csv"


def log(*a):
    """Imprime un mensaje con la hora."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cargar_tabla(fotos: Path) -> dict[str, pd.DataFrame]:
    """Tabla unificada por partición. Las fotos propias se leen de ``fotos`` (no están en el repositorio)."""
    df = pd.read_pickle(PROC / "tres_clases.pkl").reset_index(drop=True)
    propias = df["source"] == "own"
    df.loc[propias, "path"] = [str(fotos / Path(p).name) for p in df.loc[propias, "path"]]
    faltan = [p for p in df.loc[propias, "path"] if not Path(p).is_file()]
    if faltan:
        raise FileNotFoundError(f"faltan {len(faltan)} fotos propias, p. ej. {faltan[0]}")
    ann = pd.read_csv(REPO / "data" / "own_photos" / "annotations.csv", dtype=str)
    mano = dict(zip(ann["image_id"], ann["in_hand"]))
    df["in_hand"] = ""
    df.loc[propias, "in_hand"] = df.loc[propias, "path"].map(lambda p: mano.get(Path(p).name, ""))
    train = mc.cap_per_cell(df[df["split"] == "train"], ["species", "label"], CAP, SEED)
    return {"train": train, "val": df[df["split"] == "val"], "test": df[df["split"] == "test"],
            "own_test": df[df["split"] == "own_test"]}


def matrices(partes: dict, size: int) -> dict[str, np.ndarray]:
    """Miniaturas ``size`` x ``size`` de cada partición (con caché en ``data/processed``)."""
    cache = PROC / f"tres_{size}.npz"
    firma = np.array([zlib.crc32(str(p).encode()) for k in partes for p in partes[k]["path"]], dtype=np.int64)
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["firma"], firma):
            return {k: z[k] for k in partes}
    log(f"leyendo y reduciendo imágenes a {size} px")
    out = {k: ft.thumbs_matrix(partes[k]["path"], size=size, workers=8) for k in partes}
    np.savez(cache, firma=firma, **out)
    return out


def fila_conjunto(modelo: str, conjunto: str, sub: pd.DataFrame, pred: np.ndarray) -> dict:
    """Métricas de un conjunto: exactitud con su EE, F1 macro (clases presentes en la verdad), kappa ponderado y recall."""
    y = sub["label"].to_numpy()
    ev = mc.evaluate(y, pred)["global"]
    presentes = [c for c in CLASES if c in set(y)]
    fila = {"modelo": modelo, "conjunto": conjunto, "n": len(y)}
    fila.update({f"n_{c}": int((y == c).sum()) for c in CLASES})
    fila.update({"exactitud": ev["exactitud"], "EE": ev["EE"],
                 "F1_macro": f1_score(y, pred, labels=presentes, average="macro", zero_division=0),
                 "kappa_w": ev["kappa_w"]})
    rec = recall_score(y, pred, labels=CLASES, average=None, zero_division=0)
    for c, r in zip(CLASES, rec):
        fila[f"recall_{c}"] = float(r) if (y == c).any() else float("nan")
    fila["F1_Regular"] = float(f1_score(y, pred, labels=["Regular"], average=None, zero_division=0)[0]) if (y == "Regular").any() else float("nan")
    return fila


def filas_validacion(modelo: str, val: pd.DataFrame, pred: np.ndarray) -> list[dict]:
    """Métricas de validación: conjunto completo y por fuente (Kaggle y *Mixed*)."""
    filas = [fila_conjunto(modelo, "val", val, pred)]
    for fuente in ("kaggle", "mixed"):
        m = (val["source"] == fuente).to_numpy()
        filas.append(fila_conjunto(modelo, f"val_{fuente}", val[m], pred[m]))
    return filas


def guardar_filas(csv: Path, filas: list[dict]) -> None:
    """Agrega filas al CSV, reemplazando las del mismo modelo."""
    previo = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    previo = previo[~previo.get("modelo", pd.Series(dtype=str)).isin({f["modelo"] for f in filas})] if len(previo) else previo
    pd.concat([previo, pd.DataFrame(filas)], ignore_index=True).to_csv(csv, index=False)


def fase_datos(fotos: Path) -> None:
    """Lee y reduce las imágenes (64 y 96 px) y las guarda en caché."""
    partes = cargar_tabla(fotos)
    for k, v in partes.items():
        log(f"{k}: {len(v):,} imágenes | {v['label'].value_counts().to_dict()}")
    for size in (64, 96):
        matrices(partes, size)
    log("datos listos")


def fase_clasicos(fotos: Path) -> None:
    """Entrena los modelos de ML tradicional con GridSearchCV agrupado y mide la validación."""
    partes = cargar_tabla(fotos)
    X = matrices(partes, 64)
    MODELOS.mkdir(parents=True, exist_ok=True)
    tr, va = partes["train"], partes["val"]
    ftr, fva = ft.color_features(X["train"], "full"), ft.color_features(X["val"], "full")
    ytr = tr["label"].to_numpy()
    for nombre, (est, malla) in mc.model_zoo(SEED).items():
        t0 = time.time()
        if malla:
            gs = GridSearchCV(est, malla, scoring="f1_macro", cv=mc.make_cv(5, SEED), n_jobs=4)
            gs.fit(ftr, ytr, groups=tr["group"].to_numpy())
            modelo, params, cv = gs.best_estimator_, json.dumps(gs.best_params_), gs.best_score_
        else:
            modelo, params, cv = est.fit(ftr, ytr), "{}", float("nan")
        pred = modelo.predict(fva)
        filas = filas_validacion(nombre, va, pred)
        for f in filas:
            f.update({"hiperparametros": params, "F1_cv_train": cv, "segundos": round(time.time() - t0, 1)})
        guardar_filas(CSV_VAL, filas)
        joblib.dump(modelo, MODELOS / f"{nombre}.joblib")
        log(f"{nombre}: F1 val {filas[0]['F1_macro']:.3f} | recall Regular {filas[0]['recall_Regular']:.2f} | {time.time() - t0:.0f} s")
    # control "solo borde": cuánto del resultado puede venir del fondo
    t0 = time.time()
    btr, bva = ft.color_features(X["train"], "border"), ft.color_features(X["val"], "border")
    rf = RandomForestClassifier(300, class_weight="balanced", random_state=SEED, n_jobs=4).fit(btr, ytr)
    filas = filas_validacion("rf_solo_borde", va, rf.predict(bva))
    for f in filas:
        f.update({"hiperparametros": "{}", "F1_cv_train": float("nan"), "segundos": round(time.time() - t0, 1)})
    guardar_filas(CSV_VAL, filas)
    joblib.dump(rf, MODELOS / "rf_solo_borde.joblib")
    log(f"rf_solo_borde: F1 val {filas[0]['F1_macro']:.3f}")


# T4 y T5 (mismas configuraciones con el peso de clase completo, ~60 para Regular) colapsaron: la red predecía Regular para
# casi todo (exactitud de validación 0.0-0.02 en algunas épocas, F1 macro de validación 0.30 y 0.32). Se conservan en
# resultados_tres_clases.csv como experimentos fallidos y NO pasan a la prueba. T6 y T7 suavizan el peso (raíz cuadrada).
CNN_EXPERIMENTOS = [("T6", "Tres clases: Flatten + Dropout 0.5 + aumento + pesos de clase suavizados (raíz)", dict(head="flatten", augment=True, peso_potencia=0.5)),
                    ("T7", "Tres clases: GAP + Dropout 0.5 + aumento + pesos de clase suavizados (raíz)", dict(head="gap", augment=True, peso_potencia=0.5))]


def fase_cnn(fotos: Path, solo: list[str] | None = None) -> None:
    """Entrena las CNN T6 y T7 y mide la validación."""
    import entrenar_cnn as ec
    from tensorflow import keras

    partes = cargar_tabla(fotos)
    X = matrices(partes, 96)
    idx = {c: i for i, c in enumerate(CLASES)}
    ytr, yva = partes["train"]["label"].map(idx).to_numpy(), partes["val"]["label"].map(idx).to_numpy()
    datos = (X["train"], ytr, X["val"], yva, CLASES, partes["val"]["species"].to_numpy())
    salida = PROC / "cnn_runs"
    for id_, desc, cambios in CNN_EXPERIMENTOS:
        if solo and id_ not in solo:
            continue
        log(f"=== {id_}: {desc}")
        fila = ec.entrenar(dict(ec.BASE, id=id_, desc=desc, **cambios), datos, "tres_clases", False, salida)
        log(f"    F1 val {fila['F1_val']:.3f} | F1 train {fila['F1_train']:.3f} | épocas {fila['epocas_corridas']} | {fila['segundos']:.0f} s")
        modelo = keras.models.load_model(salida / f"tres_clases_{id_}.keras")
        pred = np.array(CLASES)[modelo.predict(X["val"].astype("float32") / 255.0, batch_size=64, verbose=0).argmax(1)]
        filas = filas_validacion(f"cnn_{id_}", partes["val"], pred)
        for f in filas:
            f.update({"hiperparametros": f"{desc}; épocas corridas {fila['epocas_corridas']}, mejor época {fila['mejor_epoca']}",
                      "F1_cv_train": float("nan"), "segundos": fila["segundos"]})
        guardar_filas(CSV_VAL, filas)


def predecir_todos(partes: dict, X64: dict, X96: dict, conjunto: str) -> dict[str, np.ndarray]:
    """Predicciones de todos los modelos declarados sobre un conjunto."""
    from tensorflow import keras
    preds = {}
    f64 = ft.color_features(X64[conjunto], "full")
    fb = ft.color_features(X64[conjunto], "border")
    for nombre in ("baseline_mayoritaria", "logreg", "svm_rbf", "random_forest"):
        preds[nombre] = joblib.load(MODELOS / f"{nombre}.joblib").predict(f64)
    preds["rf_solo_borde"] = joblib.load(MODELOS / "rf_solo_borde.joblib").predict(fb)
    for id_, _, _ in CNN_EXPERIMENTOS:
        m = keras.models.load_model(PROC / "cnn_runs" / f"tres_clases_{id_}.keras")
        preds[f"cnn_{id_}"] = np.array(CLASES)[m.predict(X96[conjunto].astype("float32") / 255.0, batch_size=64, verbose=0).argmax(1)]
    return preds


def fase_test(fotos: Path) -> None:
    """Evalúa todos los modelos en la prueba, una sola vez (se niega a repetirse)."""
    if CSV_TEST.exists():
        sys.exit(f"{CSV_TEST.name} ya existe: la prueba se usa una sola vez. No se repite ni se vuelve a ajustar nada con ella.")
    faltan = [n for n in ("baseline_mayoritaria", "logreg", "svm_rbf", "random_forest", "rf_solo_borde") if not (MODELOS / f"{n}.joblib").exists()]
    faltan += [i for i, _, _ in CNN_EXPERIMENTOS if not (PROC / "cnn_runs" / f"tres_clases_{i}.keras").exists()]
    if faltan:
        sys.exit(f"faltan modelos entrenados: {faltan}. Corre antes las fases clasicos y cnn.")
    partes = cargar_tabla(fotos)
    X64, X96 = matrices(partes, 64), matrices(partes, 96)
    filas, matrices_conf = [], {}
    for conjunto in ("test", "own_test"):
        sub = partes[conjunto].reset_index(drop=True)
        preds = predecir_todos(partes, X64, X96, conjunto)
        for modelo, pred in preds.items():
            subconjuntos = {conjunto: np.ones(len(sub), bool)}
            if conjunto == "test":
                for fuente in ("kaggle", "mixed"):
                    subconjuntos[f"test_{fuente}"] = (sub["source"] == fuente).to_numpy()
            else:
                subconjuntos["own_en_mano"] = (sub["in_hand"] == "1").to_numpy()
                subconjuntos["own_sin_mano"] = (sub["in_hand"] == "0").to_numpy()
                for esp in sorted(sub["species"].unique()):
                    subconjuntos[f"own_{esp}"] = (sub["species"] == esp).to_numpy()
            for nombre_c, m in subconjuntos.items():
                filas.append(fila_conjunto(modelo, nombre_c, sub[m], pred[m]))
            cruce = pd.crosstab(sub["label"], pd.Series(pred, name="pred")).reindex(index=CLASES, columns=CLASES, fill_value=0)
            matrices_conf[f"{modelo}|{conjunto}"] = cruce.values.tolist()
            if conjunto == "own_test":
                pd.DataFrame({"foto": sub["path"].map(lambda p: Path(p).name), "especie": sub["species"], "mano": sub["in_hand"],
                              "verdad": sub["label"], "prediccion": pred}).to_csv(MODELOS / f"predicciones_own_test_{modelo}.csv", index=False)
        log(f"{conjunto}: evaluado ({len(sub)} imágenes)")
    pd.DataFrame(filas).to_csv(CSV_TEST, index=False)
    (MODELOS / "matrices_confusion_test.json").write_text(json.dumps(matrices_conf, ensure_ascii=False), encoding="utf-8")
    log(f"resultados guardados en {CSV_TEST.name}; la prueba ya no se vuelve a usar")


def main() -> None:
    """Línea de comandos: corre la fase indicada."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fase", choices=["datos", "clasicos", "cnn", "test"], required=True)
    ap.add_argument("--fotos-propias", default=str(FOTOS_PROPIAS),
                    help="carpeta con F001.jpg ... (las fotos propias no están en el repositorio)")
    ap.add_argument("--solo", nargs="*", help="ids de CNN a correr (fase cnn)")
    args = ap.parse_args()
    if args.fase == "cnn":
        fase_cnn(Path(args.fotos_propias), args.solo)
    else:
        {"datos": fase_datos, "clasicos": fase_clasicos, "test": fase_test}[args.fase](Path(args.fotos_propias))


if __name__ == "__main__":
    main()
