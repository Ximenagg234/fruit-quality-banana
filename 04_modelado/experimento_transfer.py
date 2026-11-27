"""Experimento exploratorio: transfer learning con MobileNetV2 sobre la partición de tres clases.

MobileNetV2 (Sandler et al., CVPR 2018) con pesos de ImageNet y entrada de 96 x 96, más capas propias
(promedio global, densa de 128 y Dropout). Dos fases: base **congelada** y **ajuste fino** de las últimas 40 capas
(las de normalización por lotes siguen congeladas). Pesos de clase suavizados (raíz cuadrada).

Es **exploratorio**: se hizo después de usar la prueba, así que solo usa entrenamiento y validación, no cambia el modelo
de la app (la CNN propia) y sus cifras no se comparan en la prueba ni en las fotos propias.

Requiere ``data/processed/tres_clases.pkl`` y descargar los pesos preentrenados (Keras los baja la primera vez).

Uso (desde la raíz del repositorio)::

    python 04_modelado/experimento_transfer.py

Salidas: ``04_modelado/resultados_transfer_validacion.csv`` y los modelos en ``data/processed/transfer/``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "04_modelado"))
import entrenar_cnn as ec  # noqa: E402
import entrenar_tres_clases as et  # noqa: E402

SALIDA = REPO / "data" / "processed" / "transfer"
CSV = REPO / "04_modelado" / "resultados_transfer_validacion.csv"
FASES = (("tl_congelada", 1e-3, 15), ("tl_ajuste_fino", 1e-5, 12))
CAPAS_AJUSTE = 40


def main() -> None:
    """Entrena las dos fases y guarda las métricas de validación."""
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.utils.set_random_seed(42)
    clases = et.CLASES
    partes = et.cargar_tabla(et.FOTOS_PROPIAS)
    X = et.matrices(partes, 96)
    idx = {c: i for i, c in enumerate(clases)}
    tr, va = partes["train"], partes["val"].reset_index(drop=True)
    ytr, yva = tr["label"].map(idx).to_numpy(), va["label"].map(idx).to_numpy()
    Xtr, Xva = X["train"].astype("float32"), X["val"].astype("float32")
    pesos = ec.pesos_de_clase(ytr, 3, 0.5)

    base = keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights="imagenet")
    base.trainable = False
    entrada = keras.Input((96, 96, 3))
    x = layers.RandomFlip("horizontal_and_vertical")(entrada)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.2)(x)
    x = layers.RandomContrast(0.3)(x)
    x = layers.RandomBrightness(0.25, value_range=(0, 255))(x)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)          # MobileNetV2 espera píxeles en [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    modelo = keras.Model(entrada, layers.Dense(3, activation="softmax")(x))

    SALIDA.mkdir(parents=True, exist_ok=True)
    filas = []
    for fase, lr, epocas in FASES:
        if fase == "tl_ajuste_fino":
            base.trainable = True
            for capa in base.layers[:-CAPAS_AJUSTE]:
                capa.trainable = False
            for capa in base.layers:
                if isinstance(capa, layers.BatchNormalization):
                    capa.trainable = False
        modelo.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
        t0 = time.time()
        modelo.fit(Xtr, ytr, validation_data=(Xva, yva), batch_size=32, epochs=epocas, class_weight=pesos, verbose=2,
                   callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=4, restore_best_weights=True)])
        pred = np.array(clases)[modelo.predict(Xva, batch_size=64, verbose=0).argmax(1)]
        nuevas = et.filas_validacion(fase, va, pred)
        for f in nuevas:
            f["segundos"] = round(time.time() - t0, 1)
        filas += nuevas
        modelo.save(SALIDA / f"{fase}.keras")
    pd.DataFrame(filas).to_csv(CSV, index=False)
    print(pd.DataFrame(filas)[["modelo", "conjunto", "F1_macro", "recall_Regular"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
