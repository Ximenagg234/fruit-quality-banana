"""Pruebas de ``src/models_classic.py`` y del histograma HSV de ``src/features.py``."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features as ft  # noqa: E402
import models_classic as mc  # noqa: E402


def test_hsv_histogram_rows_sum_to_one_and_locate_colors():
    rojo = np.zeros((16, 16, 3), dtype=np.uint8); rojo[:] = (255, 0, 0)
    verde = np.zeros((16, 16, 3), dtype=np.uint8); verde[:] = (0, 255, 0)
    h = ft.hsv_histogram(np.stack([rojo, verde]), bins=(8, 4, 4), region="full")
    assert h.shape == (2, 128) and np.allclose(h.sum(axis=1), 1.0)
    # cada imagen es de un solo color: todo el histograma cae en un único intervalo...
    assert (h.max(axis=1) == 1.0).all()
    # ...y rojo y verde caen en intervalos distintos
    assert h[0].argmax() != h[1].argmax()
    # el borde de una imagen con centro distinto solo ve el color del borde
    mix = rojo.copy(); mix[4:12, 4:12] = (0, 255, 0)
    borde = ft.hsv_histogram(mix[None], region="border", frame=4)[0]
    assert borde.argmax() == h[0].argmax() and borde.max() == 1.0


def test_hsv_histogram_chunking_equivalence():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 256, (7, 16, 16, 3), dtype=np.uint8)
    assert np.allclose(ft.hsv_histogram(t, chunk=2), ft.hsv_histogram(t, chunk=1000))


def test_cap_per_cell_limits_each_cell_only():
    df = pd.DataFrame({"species": ["a"] * 10 + ["b"] * 3, "label": ["x"] * 13})
    out = mc.cap_per_cell(df, ["species", "label"], cap=4, seed=1)
    assert out.groupby("species").size().to_dict() == {"a": 4, "b": 3}
    assert mc.cap_per_cell(df, ["species", "label"], 4, 1).equals(out)         # reproducible


def test_evaluate_global_and_per_species():
    y = ["Buena", "Buena", "Mala", "Mala", "Buena", "Mala"]
    p = ["Buena", "Mala", "Mala", "Mala", "Buena", "Mala"]
    sp = ["apple", "apple", "apple", "banana", "banana", "banana"]
    r = mc.evaluate(y, p, sp)
    assert r["global"]["n"] == 6 and r["global"]["exactitud"] == pytest.approx(5 / 6)
    assert r["global"]["EE"] == pytest.approx(np.sqrt(5 / 6 * 1 / 6 / 6))
    assert list(r["per_species"].index) == ["apple", "banana"]
    assert r["per_species"].loc["banana", "exactitud"] == 1.0
    assert r["per_species"].loc["apple", "n"] == 3
    # una especie con una sola clase no rompe el kappa
    solo = mc.evaluate(["Buena"] * 3, ["Buena"] * 3, ["x"] * 3)
    assert np.isnan(solo["global"]["kappa_w"])


def test_binomial_se_edge_cases():
    assert mc.binomial_se(0.5, 100) == pytest.approx(0.05)
    assert np.isnan(mc.binomial_se(0.5, 0))


def test_model_zoo_fits_and_grids_are_valid():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 5)); y = (x[:, 0] > 0).astype(int)
    for nombre, (est, malla) in mc.model_zoo(0).items():
        est.fit(x, y)
        assert est.predict(x).shape == (60,), nombre
        for k in malla:                         # cada hiperparámetro de la malla existe en el estimador
            assert k in est.get_params(), (nombre, k)


def test_cv_has_no_group_overlap_between_folds():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(40), 5)
    y = np.repeat(rng.integers(0, 2, 40), 5)
    for tr, te in mc.make_cv(5, 0).split(np.zeros((200, 1)), y, g):
        assert not set(g[tr]) & set(g[te])


def test_evaluate_kappa_uses_ordinal_order_buena_regular_mala():
    """Confundir Buena con Regular debe pesar menos que confundir Buena con Mala (no el orden alfabético)."""
    verdad = ["Buena"] * 10 + ["Regular"] * 10 + ["Mala"] * 10
    cerca = ["Buena"] * 10 + ["Mala"] * 10 + ["Mala"] * 10            # Regular -> Mala: un escalón
    lejos = ["Buena"] * 10 + ["Regular"] * 10 + ["Buena"] * 10        # Mala -> Buena: dos escalones
    k_cerca = mc.evaluate(verdad, cerca)["global"]["kappa_w"]
    k_lejos = mc.evaluate(verdad, lejos)["global"]["kappa_w"]
    assert k_cerca > k_lejos
