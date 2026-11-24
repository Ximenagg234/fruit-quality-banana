"""Tabla train / val / test, matrices de confusión y análisis de errores de los modelos de tres clases.

No vuelve a evaluar la prueba para elegir nada: las métricas de validación y de prueba salen de
``resultados_tres_clases.csv`` y ``resultados_tres_clases_test.csv``; solo se calculan de nuevo las predicciones de
entrenamiento (para la brecha) y las que se necesitan para dibujar las matrices y los ejemplos de error. Con esto no se
cambia ni se vuelve a elegir ningún modelo.

Salidas (desde la raíz del repositorio: ``python 04_modelado/analisis_modelos.py``):

* ``04_modelado/tabla_train_val_test.csv``: F1 macro y exactitud por modelo en entrenamiento (submuestreado),
  validación, prueba (Kaggle + *Mixed*) y fotos propias, con la brecha entrenamiento - validación;
* ``04_modelado/figuras/matrices_confusion_tres_clases.pdf``: matrices de confusión (vectorial);
* ``04_modelado/figuras/errores_cnn_T6.pdf``: ejemplos de error de la CNN T6 en la prueba (imágenes de Kaggle y *Mixed*;
  las fotos propias no se muestran porque faltan los permisos de las personas cuyas manos aparecen);
* ``04_modelado/figuras/cnn_curvas_tres_clases.pdf``: curvas de validación por época de las CNN;
* ``04_modelado/errores_resumen.csv`` y ``errores_fotos_propias.csv``: conteos de errores por especie y por foto.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "04_modelado"))
import entrenar_tres_clases as et  # noqa: E402

CLASES = et.CLASES
MODELOS_TABLA = ["baseline_mayoritaria", "logreg", "svm_rbf", "random_forest", "rf_solo_borde", "cnn_T6", "cnn_T7"]
SALIDA = REPO / "04_modelado"


def tabla(partes, X64, X96) -> pd.DataFrame:
    """Tabla de F1 y exactitud por modelo en entrenamiento, validación, prueba y fotos propias."""
    val = pd.read_csv(et.CSV_VAL)
    val = val[val["conjunto"] == "val"].set_index("modelo")
    test = pd.read_csv(et.CSV_TEST)
    pred_tr = et.predecir_todos(partes, X64, X96, "train")
    tr = partes["train"].reset_index(drop=True)
    filas = []
    for m in MODELOS_TABLA:
        f = et.fila_conjunto(m, "train", tr, pred_tr[m])
        fila = {"modelo": m, "F1_train": f["F1_macro"], "exactitud_train": f["exactitud"],
                "F1_val": val.loc[m, "F1_macro"], "exactitud_val": val.loc[m, "exactitud"],
                "F1_test": test[(test.modelo == m) & (test.conjunto == "test")]["F1_macro"].iloc[0],
                "exactitud_test": test[(test.modelo == m) & (test.conjunto == "test")]["exactitud"].iloc[0],
                "F1_fotos_propias": test[(test.modelo == m) & (test.conjunto == "own_test")]["F1_macro"].iloc[0],
                "exactitud_fotos_propias": test[(test.modelo == m) & (test.conjunto == "own_test")]["exactitud"].iloc[0]}
        fila["brecha_F1_train_val"] = fila["F1_train"] - fila["F1_val"]
        filas.append(fila)
    return pd.DataFrame(filas)


def matriz(y, pred) -> np.ndarray:
    """Matriz de confusión 3 x 3 en el orden Buena, Regular, Mala."""
    return pd.crosstab(pd.Series(y, name="y"), pd.Series(pred, name="p")).reindex(index=CLASES, columns=CLASES, fill_value=0).values


def figura_matrices(partes, X64, X96) -> None:
    """Dibuja las matrices de confusión de tres modelos en validación, prueba y fotos propias."""
    modelos = ["cnn_T6", "svm_rbf", "random_forest"]
    conf = json.loads((et.MODELOS / "matrices_confusion_test.json").read_text(encoding="utf-8"))
    pred_val = et.predecir_todos(partes, X64, X96, "val")
    mats = {(m, "Validación"): matriz(partes["val"]["label"].to_numpy(), pred_val[m]) for m in modelos}
    for m in modelos:
        mats[(m, "Prueba (Kaggle y Mixed)")] = np.array(conf[f"{m}|test"])
        mats[(m, "Fotos propias")] = np.array(conf[f"{m}|own_test"])
    conjuntos = ["Validación", "Prueba (Kaggle y Mixed)", "Fotos propias"]
    nombres = {"cnn_T6": "CNN T6", "svm_rbf": "SVM", "random_forest": "Random Forest"}
    fig, ejes = plt.subplots(len(modelos), 3, figsize=(10, 9))
    for i, m in enumerate(modelos):
        for j, c in enumerate(conjuntos):
            M = mats[(m, c)]
            norm = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
            ax = ejes[i, j]
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, str(M[a, b]), ha="center", va="center", fontsize=9,
                            color="white" if norm[a, b] > 0.5 else "black")
            ax.set_xticks(range(3), CLASES, fontsize=8)
            ax.set_yticks(range(3), CLASES, fontsize=8)
            ax.set_title(f"{nombres[m]} · {c}", fontsize=9)
            if i == len(modelos) - 1:
                ax.set_xlabel("Clase predicha", fontsize=8)
            if j == 0:
                ax.set_ylabel("Clase real", fontsize=8)
    fig.suptitle("Matrices de confusión (conteos; el color es la proporción de cada clase real)", fontsize=10)
    fig.tight_layout()
    fig.savefig(SALIDA / "figuras" / "matrices_confusion_tres_clases.pdf")
    plt.close(fig)


def figura_errores(partes, X64, X96) -> None:
    """Dibuja ejemplos de error de la CNN T6 y guarda el resumen de errores por especie."""
    pred = et.predecir_todos(partes, X64, X96, "test")["cnn_T6"]
    t = partes["test"].reset_index(drop=True).assign(pred=pred)
    err = t[t["label"] != t["pred"]]
    elegidos = []
    for fuente, n in (("mixed", 8), ("kaggle", 4)):
        e = err[err["source"] == fuente]
        tipos = e.groupby(["label", "pred"]).size().sort_values(ascending=False).index
        ronda = 0
        while sum(1 for x in elegidos if x[1] == fuente) < n and ronda < 3:
            for tipo in tipos:
                cand = e[(e["label"] == tipo[0]) & (e["pred"] == tipo[1])].sort_values("path")
                if len(cand) > ronda and sum(1 for x in elegidos if x[1] == fuente) < n:
                    elegidos.append((cand.iloc[ronda], fuente))
            ronda += 1
    fig, ejes = plt.subplots(3, 4, figsize=(10, 8.5))
    for ax in ejes.ravel():
        ax.axis("off")
    for ax, (fila, fuente) in zip(ejes.ravel(), elegidos):
        with Image.open(fila["path"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"Real: {fila['label']} · Pred: {fila['pred']}\n{fila['species']} ({'Mixed' if fuente == 'mixed' else 'Kaggle'})", fontsize=8)
    fig.suptitle("Errores de la CNN T6 en la prueba (8 de Mixed y 4 de Kaggle)", fontsize=10)
    fig.tight_layout()
    fig.savefig(SALIDA / "figuras" / "errores_cnn_T6.pdf")
    plt.close(fig)
    # resumen de errores por especie y tipo, para T6 y SVM en la prueba
    resumen = []
    pred_all = et.predecir_todos(partes, X64, X96, "test")
    for m in ("cnn_T6", "svm_rbf"):
        d = partes["test"].reset_index(drop=True).assign(pred=pred_all[m])
        for (fuente, esp, real, p), n in d.groupby(["source", "species", "label", "pred"]).size().items():
            resumen.append({"modelo": m, "conjunto": "prueba", "fuente": fuente, "especie": esp, "real": real, "predicha": p, "n": n})
    pd.DataFrame(resumen).to_csv(SALIDA / "errores_resumen.csv", index=False)


def figura_curvas() -> None:
    """Curvas de entrenamiento y validación (pérdida y exactitud) de las CNN de tres clases.

    T4 y T5 (peso de clase completo, partición anterior) colapsaron y se muestran como contraste; T6 y T7 son las finales.
    """
    runs = et.PROC / "cnn_runs"
    nombres = {"T4": "T4 Flatten, peso completo (colapsó)", "T5": "T5 GAP, peso completo (colapsó)",
               "T6": "T6 Flatten, peso suavizado", "T7": "T7 GAP, peso suavizado"}
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    for id_, nombre in nombres.items():
        h = json.loads((runs / f"tres_clases_{id_}.json").read_text(encoding="utf-8"))["historia"]
        ep = range(1, len(h["loss"]) + 1)
        estilo = "--" if id_ in ("T4", "T5") else "-"
        ejes[0].plot(ep, h["val_loss"], estilo, marker="o", ms=3, label=nombre)
        ejes[1].plot(ep, h["val_accuracy"], estilo, marker="o", ms=3, label=nombre)
    ejes[0].set_ylabel("Pérdida de validación")
    ejes[1].set_ylabel("Exactitud de validación")
    for ax in ejes:
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3)
    ejes[0].legend(fontsize=7)
    fig.suptitle("CNN de tres clases: validación por época", fontsize=10)
    fig.tight_layout()
    fig.savefig(SALIDA / "figuras" / "cnn_curvas_tres_clases.pdf")
    plt.close(fig)


def errores_fotos_propias() -> None:
    """Guarda, por foto propia, la verdad y la predicción de la CNN T6 y del SVM."""
    d = None
    for m in ("cnn_T6", "svm_rbf"):
        p = pd.read_csv(et.MODELOS / f"predicciones_own_test_{m}.csv")
        p = p.rename(columns={"prediccion": f"pred_{m}"})
        d = p if d is None else d.merge(p[["foto", f"pred_{m}"]], on="foto")
    d["foto"] = d["foto"].str.replace(".jpg", "", regex=False)
    d.rename(columns={"mano": "en_mano"}).to_csv(SALIDA / "errores_fotos_propias.csv", index=False)


def main() -> None:
    """Genera la tabla, las figuras y los resúmenes de errores."""
    (SALIDA / "figuras").mkdir(exist_ok=True)
    partes = et.cargar_tabla(et.FOTOS_PROPIAS)
    X64, X96 = et.matrices(partes, 64), et.matrices(partes, 96)
    t = tabla(partes, X64, X96)
    t.to_csv(SALIDA / "tabla_train_val_test.csv", index=False)
    print(t.round(3).to_string(index=False))
    figura_matrices(partes, X64, X96)
    figura_errores(partes, X64, X96)
    errores_fotos_propias()
    figura_curvas()
    print("listo")


if __name__ == "__main__":
    main()
