"""Pruebas de ``src/inference.py`` (lógica de la app) con imágenes sintéticas."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features as ft  # noqa: E402
import inference as inf  # noqa: E402


def _fruit_image(color, size=(300, 200), blob=(150, 100, 70, 45)):
    arr = np.empty((size[1], size[0], 3), dtype=np.uint8)
    arr[:] = (235, 235, 230)
    yy, xx = np.mgrid[: size[1], : size[0]]
    cx, cy, a, b = blob
    arr[((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1] = color
    return Image.fromarray(arr)


@pytest.fixture(scope="module")
def tiny_model():
    """Modelo mínimo: amarillo = Buena, marrón oscuro = Mala (con ligeras variaciones)."""
    rng = np.random.default_rng(0)
    imgs, y = [], []
    for k in range(20):
        jitter = rng.integers(-12, 12, 3)
        imgs.append(_fruit_image(np.clip((220, 190, 30) + jitter, 0, 255).astype(np.uint8))); y.append("Buena")
        imgs.append(_fruit_image(np.clip((70, 40, 20) + jitter, 0, 255).astype(np.uint8))); y.append("Mala")
    X = np.vstack([inf.extract_features(i) for i in imgs])
    return RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)


def test_color_features_matches_the_training_recipe():
    thumbs = np.random.default_rng(1).integers(0, 256, (3, 64, 64, 3), dtype=np.uint8)
    esperado = np.hstack([ft.hsv_histogram(thumbs), ft.color_stats(thumbs)])
    assert np.array_equal(ft.color_features(thumbs), esperado) and esperado.shape == (3, 136)
    im = _fruit_image((220, 190, 30))
    assert inf.extract_features(im).shape == (1, 136)
    assert np.array_equal(ft.thumb_from_image(im, 64), ft.thumb_from_image(im.convert("RGBA"), 64))


def test_predict_quality_structure_and_labels(tiny_model):
    buena = inf.predict_quality(tiny_model, _fruit_image((220, 190, 30)))
    mala = inf.predict_quality(tiny_model, _fruit_image((70, 40, 20)))
    assert (buena["label"], mala["label"]) == ("Buena", "Mala")
    assert sum(buena["probabilities"].values()) == pytest.approx(1.0)
    assert set(buena) == {"label", "confidence", "probabilities", "low_confidence"}
    assert buena["confidence"] == max(buena["probabilities"].values())


def test_low_confidence_flag(tiny_model):
    im = _fruit_image((220, 190, 30))
    assert inf.predict_quality(tiny_model, im, low_confidence=1.01)["low_confidence"] is True
    assert inf.predict_quality(tiny_model, im, low_confidence=0.0)["low_confidence"] is False


def test_measure_size_recovers_the_known_length():
    r = inf.measure_size(_fruit_image((220, 190, 30), blob=(150, 100, 80, 30)))
    assert r["length_px"] == pytest.approx(160, rel=0.04)
    assert r["ell"] == pytest.approx(160 / 300, rel=0.04)
    # la medida relativa no cambia si la misma foto viene con el doble de resolución
    grande = _fruit_image((220, 190, 30), size=(600, 400), blob=(300, 200, 160, 60))
    assert inf.measure_size(grande)["ell"] == pytest.approx(r["ell"], rel=0.05)


def test_measure_size_without_object_and_overlay():
    assert inf.measure_size(Image.fromarray(np.full((100, 100, 3), 235, dtype=np.uint8))) is None
    im = _fruit_image((220, 190, 30))
    r = inf.measure_size(im)
    out = inf.overlay_mask(im, r["mask"])
    assert out.size == (r["mask"].shape[1], r["mask"].shape[0])
    arr = np.asarray(out)
    # la fruta (220,190,30) mezclada al 45 % con magenta pierde verde (190 -> ~104); el fondo (235) no cambia
    assert arr[r["mask"]][:, 1].mean() < 120
    assert arr[~r["mask"]][:, 1].mean() > 220


def test_load_model_roundtrip_and_missing_file(tmp_path, tiny_model):
    import joblib
    joblib.dump(tiny_model, tmp_path / "m.joblib")
    assert hasattr(inf.load_model(tmp_path / "m.joblib"), "predict_proba")
    with pytest.raises(FileNotFoundError):
        inf.load_model(tmp_path / "no_existe.joblib")


def test_inference_does_not_write_files(tmp_path, tiny_model, monkeypatch):
    """Privacidad por diseño (E7): predecir y medir no crea ningún archivo."""
    monkeypatch.chdir(tmp_path)
    im = _fruit_image((220, 190, 30))
    inf.predict_quality(tiny_model, im); inf.measure_size(im)
    assert list(tmp_path.iterdir()) == []
