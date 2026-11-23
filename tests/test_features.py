"""Pruebas de ``src/features.py`` con imágenes sintéticas."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features as ft  # noqa: E402


def _img(border_rgb, center_rgb, size=32, frame=8):
    arr = np.empty((size, size, 3), dtype=np.uint8)
    arr[:] = border_rgb
    arr[frame:size - frame, frame:size - frame] = center_rgb
    return arr


def test_region_masks_partition_the_image():
    border, center = ft.region_mask(32, "border", 8), ft.region_mask(32, "center", 8)
    assert border.shape == center.shape == (32, 32)
    assert not (border & center).any() and (border | center).all()
    assert ft.region_mask(32, "full").all()
    assert center.sum() == 16 * 16
    for malo in (("otra", 8), ("border", 0), ("border", 16)):
        with pytest.raises(ValueError):
            ft.region_mask(32, malo[0], malo[1])


def test_color_stats_separates_border_from_center():
    # borde rojo, centro verde
    thumbs = np.stack([_img((255, 0, 0), (0, 255, 0))])
    b = ft.color_stats(thumbs, "border", 8)[0]
    c = ft.color_stats(thumbs, "center", 8)[0]
    assert b.shape == (8,) and len(ft.COLOR_FEATURES) == 8
    assert (b[0], b[1], b[2]) == (1.0, 0.0, 0.0)          # medias R, G, B del borde
    assert (c[0], c[1], c[2]) == (0.0, 1.0, 0.0)
    assert np.allclose(b[3:6], 0) and np.allclose(c[3:6], 0)   # regiones uniformes: desviación 0
    assert b[6] == c[6] == 1.0 and b[7] == c[7] == 1.0         # saturación y valor máximos


def test_color_stats_full_is_weighted_mean_and_chunking_is_equivalent():
    thumbs = np.stack([_img((255, 255, 255), (0, 0, 0)), _img((0, 0, 0), (255, 255, 255))] * 3)
    full = ft.color_stats(thumbs, "full", 8)
    frac_center = 16 * 16 / (32 * 32)
    assert np.allclose(full[0, 0], 1 - frac_center)            # blanco en el borde, negro al centro
    assert np.allclose(ft.color_stats(thumbs, "full", 8, chunk=1), full)


def test_thumbs_matrix_shapes_and_empty(tmp_path):
    for k in range(3):
        Image.fromarray(np.full((50, 80, 3), 40 * k, dtype=np.uint8)).save(tmp_path / f"{k}.png")
    t = ft.thumbs_matrix(sorted(tmp_path.glob("*.png")), size=16)
    assert t.shape == (3, 16, 16, 3) and t.dtype == np.uint8
    assert ft.thumbs_matrix([], size=16).shape == (0, 16, 16, 3)
    assert ft.load_thumb(tmp_path / "1.png", 16).mean() == pytest.approx(40, abs=1)
