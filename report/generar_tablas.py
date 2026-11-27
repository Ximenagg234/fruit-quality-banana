"""Genera ``report/tablas_informe.md``: las tablas del informe con los números finales, leídos de los CSV del repositorio.

Así ninguna cifra se escribe a mano: si se vuelve a entrenar, se vuelve a correr este script.

Uso (desde la raíz del repositorio)::

    python report/generar_tablas.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MOD = REPO / "04_modelado"
NOMBRES = {"baseline_mayoritaria": "Línea base (clase más frecuente)", "logreg": "Regresión logística", "svm_rbf": "SVM (RBF)",
           "random_forest": "Random Forest", "rf_solo_borde": "Random Forest solo borde (control)",
           "cnn_T6": "CNN T6 (Flatten)", "cnn_T7": "CNN T7 (GAP)"}
ORDEN = list(NOMBRES)


def f(x, d=3) -> str:
    """Formatea un número con ``d`` decimales (guion si falta)."""
    return "—" if pd.isna(x) else f"{x:.{d}f}"


def tabla_md(encabezado: list[str], filas: list[list[str]]) -> str:
    """Tabla en formato Markdown."""
    return "\n".join(["| " + " | ".join(encabezado) + " |", "|" + "|".join("---" for _ in encabezado) + "|"] +
                     ["| " + " | ".join(r) + " |" for r in filas])


def main() -> None:
    """Lee los CSV y escribe ``report/tablas_informe.md``."""
    val = pd.read_csv(MOD / "resultados_tres_clases.csv")
    test = pd.read_csv(MOD / "resultados_tres_clases_test.csv")
    brecha = pd.read_csv(MOD / "tabla_train_val_test.csv").set_index("modelo")
    acuerdo = pd.read_csv(REPO / "data" / "mixed_labels" / "acuerdo_anotadores.csv")
    hw = json.loads((REPO / "05_evaluacion" / "hardware.json").read_text(encoding="utf-8"))
    costo_ent = pd.read_csv(REPO / "05_evaluacion" / "costo_entrenamiento.csv")
    costo_inf = pd.read_csv(REPO / "05_evaluacion" / "costo_inferencia.csv")
    sal = ["# Tablas del informe (generadas; no editar a mano)\n",
           "Se generan con `python report/generar_tablas.py` a partir de los CSV de `04_modelado/`, `05_evaluacion/` y `data/mixed_labels/`. "
           "Cada tabla dice de qué archivo sale. *Regular* tiene pocos casos (ver los n): las diferencias entre modelos en esa clase no son concluyentes.\n"]

    def conjunto(df, c):
        """Filas de un conjunto, una por modelo, en el orden de la tabla."""
        return df[df["conjunto"] == c].set_index("modelo").loc[ORDEN]

    v = conjunto(val, "val")
    sal.append("## Tabla 1. Validación (Kaggle y *Mixed*), tres clases\n")
    sal.append(f"Fuente: `04_modelado/resultados_tres_clases.csv`. n = {int(v['n'].iloc[0])} imágenes, de ellas {int(v['n_Regular'].iloc[0])} *Regular*. "
               "F1 macro y kappa ponderado (orden Buena < Regular < Mala); EE = error estándar binomial de la exactitud.\n")
    sal.append(tabla_md(["Modelo", "Exactitud ± EE", "F1 macro", "Kappa ponderado", "Recall *Regular*", "F1 *Regular*"],
                        [[NOMBRES[m], f"{r.exactitud:.3f} ± {r.EE:.3f}", f(r.F1_macro), f(r.kappa_w), f(r.recall_Regular, 2), f(r.F1_Regular)]
                         for m, r in v.iterrows()]) + "\n")

    t = conjunto(test, "test")
    sal.append("## Tabla 2. Prueba (Kaggle y *Mixed*), usada una sola vez\n")
    sal.append(f"Fuente: `04_modelado/resultados_tres_clases_test.csv`. n = {int(t['n'].iloc[0])} imágenes, de ellas {int(t['n_Regular'].iloc[0])} *Regular*.\n")
    sal.append(tabla_md(["Modelo", "Exactitud ± EE", "F1 macro", "Kappa ponderado", "Recall *Regular*", "F1 *Regular*"],
                        [[NOMBRES[m], f"{r.exactitud:.3f} ± {r.EE:.3f}", f(r.F1_macro), f(r.kappa_w), f(r.recall_Regular, 2), f(r.F1_Regular)]
                         for m, r in t.iterrows()]) + "\n")

    o = conjunto(test, "own_test")
    sal.append("## Tabla 3. Fotos propias (prueba final de uso único)\n")
    sal.append(f"Fuente: `04_modelado/resultados_tres_clases_test.csv`, conjunto `own_test`. n = {int(o['n'].iloc[0])} fotos de 34 frutas: "
               f"{int(o['n_Buena'].iloc[0])} *Buena*, {int(o['n_Regular'].iloc[0])} *Regular*, {int(o['n_Mala'].iloc[0])} *Mala*. "
               "La línea base de decir siempre *Buena* tiene exactitud 0.710.\n")
    sal.append(tabla_md(["Modelo", "Exactitud ± EE", "F1 macro", "Recall *Buena*", "Recall *Regular*", "Recall *Mala*"],
                        [[NOMBRES[m], f"{r.exactitud:.3f} ± {r.EE:.3f}", f(r.F1_macro), f(r.recall_Buena, 2), f(r.recall_Regular, 2), f(r.recall_Mala, 2)]
                         for m, r in o.iterrows()]) + "\n")

    sal.append("## Tabla 4. Fotos propias con y sin mano, y por especie (CNN T6)\n")
    sal.append("Fuente: `04_modelado/resultados_tres_clases_test.csv`. Las 6 *Mala* están todas en mano; sin mano solo hay 14 fotos.\n")
    filas = []
    for c, nombre in (("own_en_mano", "En mano"), ("own_sin_mano", "Sin mano"), ("own_apple", "Manzana"), ("own_banana", "Banano"),
                      ("own_guava", "Guayaba"), ("own_lime", "Lima"), ("own_orange", "Naranja")):
        r = test[(test["modelo"] == "cnn_T6") & (test["conjunto"] == c)].iloc[0]
        filas.append([nombre, str(int(r.n)), f"{r.exactitud:.3f} ± {r.EE:.3f}", f(r.F1_macro, 2)])
    sal.append(tabla_md(["Subconjunto", "n", "Exactitud ± EE", "F1 macro"], filas) + "\n")

    sal.append("## Tabla 5. Entrenamiento, validación, prueba y fotos propias (F1 macro) y sobreajuste\n")
    sal.append("Fuente: `04_modelado/tabla_train_val_test.csv`. El entrenamiento está submuestreado (<= 1 100 imágenes por especie y clase).\n")
    sal.append(tabla_md(["Modelo", "Entrenamiento", "Validación", "Prueba", "Fotos propias", "Brecha entren. − val."],
                        [[NOMBRES[m], f(r.F1_train), f(r.F1_val), f(r.F1_test), f(r.F1_fotos_propias), f(r.brecha_F1_train_val, 2)]
                         for m, r in brecha.loc[ORDEN].iterrows()]) + "\n")

    sal.append("## Tabla 6. Acuerdo entre anotadores (recortes de *Mixed*)\n")
    sal.append("Fuente: `data/mixed_labels/acuerdo_anotadores.csv`. Dos personas por ronda, etiquetando por separado; los desacuerdos se resolvieron en grupo.\n")
    sal.append(tabla_md(["Ronda", "Recortes", "Sin descartar", "Concordancia exacta", "Kappa ponderado", "Desacuerdos"],
                        [[str(int(r.ronda)), str(int(r.recortes_etiquetados_por_ambos)), str(int(r.sin_descartar)), f"{r.concordancia_exacta:.1%}",
                          f(r.kappa_ponderado), str(int(r.desacuerdos))] for r in acuerdo.itertuples()]) + "\n")

    sal.append("## Tabla 7. Costo de cómputo\n")
    sal.append(f"Fuente: `05_evaluacion/`. Equipo: {hw['procesador']}, {hw['nucleos']} núcleos, {hw['memoria_GB']} GB, sin GPU. "
               "La energía es **estimada** (potencia del fabricante, 15 a 55 W, por el tiempo), no medida.\n")
    final = costo_ent[costo_ent["etapa"].str.contains("ronda 2")]
    sal.append(tabla_md(["Concepto", "Valor"], [
        ["Entrenamiento de los modelos finales (procesos)", f"{final['segundos'].sum():.0f} s ({final['horas'].sum():.2f} h)"],
        ["Entrenamiento de todo el proyecto (procesos)", f"{costo_ent['segundos'].sum():.0f} s ({costo_ent['horas'].sum():.1f} h)"],
        ["Energía estimada de todo el proyecto", f"{costo_ent['kWh_estimado_min'].sum():.2f} a {costo_ent['kWh_estimado_max'].sum():.2f} kWh"],
    ] + [[r.paso, f"{r.mediana_ms:.0f} ms (p95 {r.p95_ms:.0f} ms)" if pd.notna(r.p95_ms) else f"{r.mediana_ms:.0f} ms"] for r in costo_inf.itertuples()]) + "\n")
    tl = pd.read_csv(MOD / "resultados_transfer_validacion.csv")
    base = val[(val["modelo"] == "cnn_T6")]
    comp = pd.concat([base, tl])
    nombres = {"cnn_T6": "CNN propia (la de la app)", "tl_congelada": "MobileNetV2 congelada", "tl_ajuste_fino": "MobileNetV2 con ajuste fino"}
    filas = []
    for m, nombre in nombres.items():
        a = comp[(comp["modelo"] == m) & (comp["conjunto"] == "val")].iloc[0]
        b = comp[(comp["modelo"] == m) & (comp["conjunto"] == "val_mixed")].iloc[0]
        filas.append([nombre, f(a.F1_macro), f(a.recall_Regular, 2), f(a.F1_Regular), f(b.F1_macro)])
    sal.append("## Tabla 8. Exploración: transfer learning (solo validación)\n")
    sal.append("Fuente: `04_modelado/resultados_transfer_validacion.csv`. Se hizo **después** de usar la prueba, así que solo se mide en "
               "validación (35 *Regular*; la diferencia no es concluyente). No cambia el modelo de la app.\n")
    sal.append(tabla_md(["Modelo", "F1 macro", "Recall *Regular*", "F1 *Regular*", "F1 macro en *Mixed*"], filas) + "\n")
    (REPO / "report" / "tablas_informe.md").write_text("\n".join(sal), encoding="utf-8")
    print("report/tablas_informe.md generado")


if __name__ == "__main__":
    main()
