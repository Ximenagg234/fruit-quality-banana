"""Inferencia para la app: calidad de una fruta y medición de su tamaño.

Todo se hace **en memoria**: ninguna función guarda la imagen en disco ni la envía a
terceros (medida de privacidad E7, ``docs/etica_tabla.md``).

* Calidad: una CNN de Keras (``.keras``, la imagen reducida a 96 x 96 como en el entrenamiento) o un modelo de
  scikit-learn con ``predict_proba`` sobre características de color (``features.color_features``).
* Tamaño: segmentación por saturación y longitud normalizada ``ell = 4*sqrt(lambda_1)/W``
  (``segmentation.py``). Es una medida **relativa**: depende de la distancia cámara-fruta.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from PIL import Image

import features as ft
import segmentation as sg

#: Por debajo de esta probabilidad máxima, la app avisa que la predicción es poco fiable.
LOW_CONFIDENCE = 0.70
#: Lado de la miniatura que recibe la CNN (el mismo del entrenamiento) y orden de sus clases.
CNN_SIZE = 96
CNN_CLASSES = ["Buena", "Regular", "Mala"]
#: La imagen se reduce a este lado máximo antes de segmentar (rapidez; no cambia ``ell``).
SEGMENT_MAX_SIDE = 400
#: Si más de esta fracción del borde de la imagen cae en la máscara, se avisa de una posible mano,
#: brazo o fruta cortada. Calibrado en las fotos propias: 0.000 en 7 fotos sobre superficie y de
#: 0.065 a 0.122 en 22 fotos con mano (revisión visual).
HAND_BORDER_FRAC = 0.02


class CnnQuality:
    """Envuelve una CNN de Keras entrenada con miniaturas ``size`` x ``size`` y clases ``classes`` (en ese orden)."""

    def __init__(self, keras_model, size: int = CNN_SIZE, classes=CNN_CLASSES):
        self.model, self.size, self.classes_ = keras_model, size, np.array(classes)

    def predict_proba_image(self, im: Image.Image) -> np.ndarray:
        """Probabilidades por clase de una imagen PIL (la misma reducción que en el entrenamiento)."""
        x = ft.thumb_from_image(im, self.size)[None].astype("float32") / 255.0
        return np.asarray(self.model.predict(x, verbose=0)[0], dtype=float)


def load_model(path: Path | str):
    """Carga el modelo de calidad: una CNN de Keras (``.keras``) o un ``Pipeline`` de scikit-learn (``joblib``).

    Raises:
        FileNotFoundError: si el archivo no existe.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {path}")
    if path.suffix == ".keras":
        from tensorflow import keras                       # se importa aquí: solo hace falta con la CNN
        return CnnQuality(keras.models.load_model(path))
    return joblib.load(path)


def extract_features(im: Image.Image) -> np.ndarray:
    """Características de color ``(1, 136)`` de una imagen PIL, con el mismo método del entrenamiento."""
    return ft.color_features(ft.thumb_from_image(im, 64)[None], region="full")


def predict_quality(model, im: Image.Image, low_confidence: float = LOW_CONFIDENCE) -> dict:
    """Predice la calidad de una imagen.

    Returns:
        ``{'label', 'confidence', 'probabilities', 'low_confidence'}``; ``probabilities`` es
        un diccionario clase -> probabilidad (suma 1) y ``low_confidence`` es ``True`` si la
        probabilidad máxima es menor que ``low_confidence``.
    """
    if hasattr(model, "predict_proba_image"):               # CNN: recibe la imagen
        proba = model.predict_proba_image(im)
    else:                                                    # scikit-learn: recibe las características de color
        proba = model.predict_proba(extract_features(im))[0]
    classes = [str(c) for c in model.classes_]
    k = int(np.argmax(proba))
    return {"label": classes[k], "confidence": float(proba[k]),
            "probabilities": dict(zip(classes, map(float, proba))),
            "low_confidence": bool(proba[k] < low_confidence)}


def measure_size(im: Image.Image, method: str = "saturation") -> Optional[dict]:
    """Segmenta la fruta y mide su longitud normalizada.

    Returns:
        ``None`` si no se encuentra un objeto; si no, un diccionario con ``ell`` (longitud
        del eje mayor dividida por el ancho de la imagen), ``length_px``, ``width_px``,
        ``area_frac`` y ``mask`` (booleana, a la resolución reducida usada para segmentar).
    """
    rgb = im.convert("RGB")
    rgb.thumbnail((SEGMENT_MAX_SIDE, SEGMENT_MAX_SIDE))
    mask = sg.segment_fruit(np.asarray(rgb), method=method)
    if mask is None:
        return None
    return {**sg.mask_geometry(mask), "mask": mask}


def overlay_mask(im: Image.Image, mask: np.ndarray, color=(255, 0, 255), alpha: float = 0.45) -> Image.Image:
    """Imagen con la máscara resaltada (para mostrarla en la app), a la resolución de ``mask``."""
    rgb = im.convert("RGB").resize((mask.shape[1], mask.shape[0]))
    arr = np.asarray(rgb).copy()
    arr[mask] = ((1 - alpha) * arr[mask] + alpha * np.array(color)).astype(np.uint8)
    return Image.fromarray(arr)
