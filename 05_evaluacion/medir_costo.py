"""Costo de cómputo del proyecto: hardware, tiempos de entrenamiento, latencia de inferencia y energía estimada.

Mide (con reloj) lo que se puede medir y **estima** lo demás, dejando claro cuál es cuál:

* **Hardware**: se lee del equipo donde se corre (nombre del procesador, núcleos, memoria).
* **Entrenamiento**: tiempos de reloj registrados en ``04_modelado/resultados_*.csv``; la primera ronda (partición anterior,
  descartada al ampliar las etiquetas) se tomó de los registros de la ejecución del 24-nov-2026 y se declara abajo.
* **Inferencia**: latencia por imagen medida aquí (mediana y percentil 95) sobre las fotos propias, para cada modelo y
  para el flujo completo de la app (calidad con la CNN + tamaño).
* **Energía**: **no se midió con vatímetro**. Se estima como potencia x tiempo con el rango que declara el fabricante del
  procesador (potencia base 15 W y potencia máxima en turbo 55 W, Intel ARK, i7-1255U). Es una cota, no un dato medido.
  El factor de emisión del sistema eléctrico colombiano (UPME) **no se usó**: sigue sin verificar (``[VERIFICAR]``).

Uso (desde la raíz del repositorio)::

    python 05_evaluacion/medir_costo.py
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import inference as inf  # noqa: E402

SALIDA = REPO / "05_evaluacion"
FOTOS = REPO.parent / "etiquetado_23nov" / "fotos_para_etiquetar" / "recortes" / "foto"
MODELOS = REPO / "data" / "processed" / "modelos_tres"
#: Potencia (W) del procesador según Intel ARK para el i7-1255U: potencia base y máxima en turbo.
POTENCIA_W = {"base": 15.0, "turbo_max": 55.0}
#: Tiempos (s) de la primera ronda de tres clases (partición anterior, descartada), de los registros de la ejecución.
RONDA1_S = {"baseline_mayoritaria": 2, "logreg": 864, "svm_rbf": 588, "random_forest": 849, "rf_solo_borde": 79,
            "cnn_T4": 1324, "cnn_T5": 1151, "cnn_T6": 6624, "cnn_T7": 7242}


def hardware() -> dict:
    """Datos del equipo: procesador, núcleos, hilos y memoria."""
    info = {"sistema": platform.platform(), "python": platform.python_version()}
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "$p=Get-CimInstance Win32_Processor; $m=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory;"
                              "\"$($p.Name)|$($p.NumberOfCores)|$($p.NumberOfLogicalProcessors)|$([math]::Round($m/1GB,1))\""],
                             capture_output=True, text=True, timeout=30).stdout.strip().split("|")
        info.update({"procesador": out[0].strip(), "nucleos": int(out[1]), "hilos": int(out[2]), "memoria_GB": float(out[3])})
    except Exception:
        info["procesador"] = platform.processor()
    info["GPU_usada"] = "no (TensorFlow para Windows corre en CPU)"
    return info


def energia_kwh(segundos: float) -> tuple[float, float]:
    """Cota de energía (kWh) en un tiempo dado, con la potencia base y la de turbo del fabricante."""
    return tuple(round(POTENCIA_W[k] * segundos / 3.6e6, 4) for k in ("base", "turbo_max"))


def entrenamiento() -> pd.DataFrame:
    """Tabla de tiempos de entrenamiento y su energía estimada."""
    filas = []
    val = pd.read_csv(REPO / "04_modelado" / "resultados_tres_clases.csv")
    val = val[val["conjunto"] == "val"]
    for r in val.itertuples():
        filas.append({"etapa": "tres clases, ronda 2 (modelos finales)", "modelo": r.modelo, "segundos": float(r.segundos)})
    for m, s in RONDA1_S.items():
        filas.append({"etapa": "tres clases, ronda 1 (descartada)", "modelo": m, "segundos": float(s)})
    for tarea in ("calidad", "especie"):
        d = pd.read_csv(REPO / "04_modelado" / f"resultados_cnn_{tarea}.csv")
        for r in d.dropna(subset=["segundos"]).itertuples():
            filas.append({"etapa": f"dos clases, CNN ({tarea})", "modelo": r.id, "segundos": float(r.segundos)})
    t = pd.DataFrame(filas)
    t["horas"] = (t["segundos"] / 3600).round(3)
    t["kWh_estimado_min"], t["kWh_estimado_max"] = zip(*t["segundos"].map(energia_kwh))
    return t


def tiempos(f, n_calentamiento: int = 3, repeticiones: int = 1) -> dict:
    """Mediana y percentil 95 (ms) de una función aplicada a cada imagen, tras unas pasadas de calentamiento."""
    ts = []
    for k, im in enumerate(IMAGENES):
        t0 = time.perf_counter()
        f(im)
        dt = time.perf_counter() - t0
        if k >= n_calentamiento:
            ts.append(dt)
    ts = np.array(ts) * 1000
    return {"mediana_ms": round(float(np.median(ts)), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1),
            "n": len(ts)}


def inferencia() -> pd.DataFrame:
    """Latencia por imagen de cada modelo y del flujo completo de la app."""
    global IMAGENES
    archivos = sorted(FOTOS.glob("F*.jpg"))
    # decodificación (apertura y orientación), medida aparte de los modelos
    t0 = time.perf_counter()
    IMAGENES = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in archivos]
    decod = (time.perf_counter() - t0) / len(archivos) * 1000
    filas = [{"paso": "abrir y decodificar la foto (1 024 px)", "mediana_ms": round(decod, 1), "p95_ms": float("nan"), "n": len(archivos)}]
    cnn = inf.load_model(REPO / "06_despliegue" / "modelo" / "calidad_cnn.keras")
    modelos = {n: joblib.load(MODELOS / f"{n}.joblib") for n in ("logreg", "svm_rbf", "random_forest")}
    for n, m in modelos.items():
        filas.append({"paso": f"{n}: características de color + predicción", **tiempos(lambda im, m=m: m.predict(inf.extract_features(im)))})
    filas.append({"paso": "CNN T6: reducir a 96 px + predicción", **tiempos(lambda im: inf.predict_quality(cnn, im))})
    filas.append({"paso": "tamaño: segmentar y medir la longitud normalizada", **tiempos(lambda im: inf.measure_size(im))})
    filas.append({"paso": "flujo completo de la app (CNN T6 + tamaño)", **tiempos(lambda im: (inf.predict_quality(cnn, im), inf.measure_size(im)))})
    t = pd.DataFrame(filas)
    t["Wh_por_1000_min"] = (POTENCIA_W["base"] * t["mediana_ms"] / 1000 * 1000 / 3600).round(3)
    t["Wh_por_1000_max"] = (POTENCIA_W["turbo_max"] * t["mediana_ms"] / 1000 * 1000 / 3600).round(3)
    return t, cnn


def tamanos(cnn) -> pd.DataFrame:
    """Tamaño en disco y parámetros de los modelos."""
    filas = [{"modelo": n, "MB": round((MODELOS / f"{n}.joblib").stat().st_size / 1e6, 2), "parametros": ""}
             for n in ("logreg", "svm_rbf", "random_forest")]
    filas.append({"modelo": "cnn_T6 (calidad_cnn.keras)", "MB": round((REPO / "06_despliegue/modelo/calidad_cnn.keras").stat().st_size / 1e6, 2),
                  "parametros": int(cnn.model.count_params())})
    return pd.DataFrame(filas)


def main() -> None:
    """Mide y guarda hardware, tiempos, latencias y tamaños."""
    hw = hardware()
    (SALIDA / "hardware.json").write_text(json.dumps({**hw, "potencia_W_fabricante": POTENCIA_W}, ensure_ascii=False, indent=1), encoding="utf-8")
    ent = entrenamiento()
    ent.to_csv(SALIDA / "costo_entrenamiento.csv", index=False)
    inf_t, cnn = inferencia()
    inf_t.to_csv(SALIDA / "costo_inferencia.csv", index=False)
    tam = tamanos(cnn)
    tam.to_csv(SALIDA / "tamano_modelos.csv", index=False)
    print(json.dumps(hw, ensure_ascii=False))
    print(ent.groupby("etapa")[["segundos", "horas", "kWh_estimado_min", "kWh_estimado_max"]].sum().round(3).to_string())
    print("TOTAL", ent["segundos"].sum(), "s =", round(ent["segundos"].sum() / 3600, 2), "h;", [round(x, 3) for x in energia_kwh(ent["segundos"].sum())], "kWh")
    print(inf_t.to_string(index=False))
    print(tam.to_string(index=False))


if __name__ == "__main__":
    main()
