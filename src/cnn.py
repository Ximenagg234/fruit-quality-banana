"""CNN pequeña desde cero (32-64-128) y conteo analítico de parámetros.

Arquitectura de la guía del curso ``CORTE 3/02-CNN_resuelto`` (bloques Conv + MaxPool
con 32, 64 y 128 filtros, luego capa densa con Dropout), con la opción de cambiar
``Flatten`` por ``GlobalAveragePooling2D`` para comparar el número de parámetros y el
sobreajuste. La fórmula analítica se verifica contra ``model.count_params()`` en
``tests/test_cnn.py`` (``docs/formulacion_matematica.md`` §6).
"""
from __future__ import annotations

from typing import Sequence

FILTERS = (32, 64, 128)


def conv_params(kernel: int, c_in: int, c_out: int) -> int:
    """Parámetros de una Conv2D: ``(k^2 * c_in + 1) * c_out`` (pesos más sesgos)."""
    return (kernel * kernel * c_in + 1) * c_out


def dense_params(n_in: int, n_out: int) -> int:
    """Parámetros de una capa densa: ``(n_in + 1) * n_out``."""
    return (n_in + 1) * n_out


def analytic_param_count(size: int = 128, n_classes: int = 3, head: str = "flatten",
                         filters: Sequence[int] = FILTERS, dense_units: int = 128,
                         kernel: int = 3) -> int:
    """Número de parámetros calculado a mano (padding ``same``, pooling 2x2 con paso 2).

    Args:
        size: lado de la imagen cuadrada de entrada (RGB).
        head: ``'flatten'`` o ``'gap'`` (``GlobalAveragePooling2D``).
    """
    if head not in ("flatten", "gap"):
        raise ValueError("head debe ser 'flatten' o 'gap'")
    total, c_in, side = 0, 3, size
    for f in filters:
        total += conv_params(kernel, c_in, f)
        c_in, side = f, side // 2
    flat = side * side * c_in if head == "flatten" else c_in
    return total + dense_params(flat, dense_units) + dense_params(dense_units, n_classes)


def build_cnn(size: int = 128, n_classes: int = 3, head: str = "flatten",
              filters: Sequence[int] = FILTERS, dense_units: int = 128, dropout: float = 0.5,
              l2: float = 0.0, augment: bool = False):
    """Construye la CNN (API funcional de Keras, como en la guía del curso).

    Las imágenes de entrada deben venir escaladas a [0, 1]. Devuelve un modelo sin
    compilar. Keras se importa aquí para no cargar TensorFlow al importar el módulo.

    Args:
        l2: coeficiente de regularización L2 en las capas convolucionales y densas (0 = sin L2).
        augment: si es ``True``, antepone aumento de datos (volteo horizontal, rotación, zoom y
            contraste aleatorios). Son capas **sin parámetros** que solo actúan al entrenar,
            así que no cambian el conteo de parámetros ni la inferencia.
    """
    if head not in ("flatten", "gap"):
        raise ValueError("head debe ser 'flatten' o 'gap'")
    from tensorflow import keras
    from tensorflow.keras import layers

    reg = keras.regularizers.l2(l2) if l2 > 0 else None
    inputs = keras.Input((size, size, 3))
    x = inputs
    if augment:
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.08)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomContrast(0.15)(x)
    for f in filters:
        x = layers.Conv2D(f, (3, 3), activation="relu", padding="same", kernel_regularizer=reg)(x)
        x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x) if head == "flatten" else layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)               # sin parámetros: no cambia el conteo
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)
