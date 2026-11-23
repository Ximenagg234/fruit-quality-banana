"""Pruebas de ``src/segmentation.py`` con formas sintéticas de tamaño conocido."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import segmentation as sg  # noqa: E402

BG = (235, 235, 230)


def _scene(shape=(200, 300), blobs=(), bg=BG, color=(210, 170, 30)):
    """Fondo liso con elipses ``(cy, cx, a, b)`` (semiejes a horizontal, b vertical)."""
    img = np.empty((*shape, 3), dtype=np.uint8)
    img[:] = bg
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for cy, cx, a, b in blobs:
        img[((yy - cy) / b) ** 2 + ((xx - cx) / a) ** 2 <= 1] = color
    return img


def test_ellipse_major_axis_is_recovered():
    img = _scene(blobs=[(100, 150, 80, 30)])               # eje mayor teórico 2a = 160 px
    mask = sg.segment_fruit(img)
    g = sg.mask_geometry(mask)
    assert g["length_px"] == pytest.approx(160, rel=0.03)
    assert g["width_px"] == pytest.approx(60, rel=0.05)
    assert g["ell"] == pytest.approx(160 / 300, rel=0.03)  # normalizado por el ancho de la imagen


def test_disk_gives_its_diameter():
    img = _scene(shape=(240, 240), blobs=[(120, 120, 50, 50)])
    g = sg.mask_geometry(sg.segment_fruit(img))
    assert g["length_px"] == pytest.approx(100, rel=0.03)  # diámetro = 2r = 4*sqrt(r^2/4)
    assert g["length_px"] == pytest.approx(g["width_px"], rel=0.05)


def test_rotation_does_not_change_length():
    # una elipse inclinada se dibuja con una matriz de rotación de 40 grados
    yy, xx = np.mgrid[:300, :300]
    th = np.deg2rad(40)
    u = (xx - 150) * np.cos(th) + (yy - 150) * np.sin(th)
    v = -(xx - 150) * np.sin(th) + (yy - 150) * np.cos(th)
    img = np.empty((300, 300, 3), dtype=np.uint8); img[:] = BG
    img[(u / 90) ** 2 + (v / 30) ** 2 <= 1] = (210, 170, 30)
    assert sg.mask_geometry(sg.segment_fruit(img))["length_px"] == pytest.approx(180, rel=0.03)


def test_keeps_only_the_largest_object():
    img = _scene(blobs=[(100, 90, 50, 50), (30, 260, 8, 8)])    # un objeto grande y una mota
    mask = sg.segment_fruit(img)
    assert mask[100, 90] and not mask[30, 260]


def test_uniform_or_tiny_objects_return_none():
    assert sg.segment_fruit(_scene()) is None                   # sin objeto
    assert sg.segment_fruit(_scene(blobs=[(100, 150, 4, 4)]), min_area_frac=0.01) is None
    with pytest.raises(ValueError):
        sg.segment_fruit(np.zeros((10, 10), dtype=np.uint8))


def test_background_estimation_and_geometry_errors():
    assert np.allclose(sg.estimate_background(_scene(blobs=[(100, 150, 40, 40)])), BG)
    with pytest.raises(ValueError):
        sg.mask_geometry(np.zeros((5, 5), dtype=bool))


def test_size_class_and_terciles():
    assert [sg.size_class(x, 0.3, 0.5) for x in (0.1, 0.3, 0.45, 0.5, 0.9)] == \
        ["Pequeño", "Mediano", "Mediano", "Grande", "Grande"]
    with pytest.raises(ValueError):
        sg.size_class(0.4, 0.6, 0.5)
    t1, t2 = sg.terciles(np.arange(1, 10))
    assert (t1, t2) == (pytest.approx(3.667, abs=1e-3), pytest.approx(6.333, abs=1e-3))
