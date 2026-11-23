"""Prueba del etiquetador Streamlit con ``AppTest``: guarda etiquetas, avanza y retoma donde quedó."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "03_preparacion_datos" / "etiquetador_mixed.py"


@pytest.fixture
def carpeta(tmp_path, monkeypatch):
    """Carpeta de recortes mínima: 3 recortes y su muestra."""
    filas = []
    for k in range(3):
        cid = f"apple_IMG{k}_00"
        (tmp_path / "recortes" / "apple").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((128, 128, 3), 60 * k + 40, dtype=np.uint8)).save(tmp_path / "recortes" / "apple" / f"{cid}.jpg")
        filas.append({"crop_id": cid, "species": "apple", "source_image": f"IMG{k}.jpg", "x0": 5, "y0": 5, "x1": 60, "y1": 60,
                      "orden": k + 1})
    pd.DataFrame(filas).to_csv(tmp_path / "muestra_etiquetado.csv", index=False)
    monkeypatch.setenv("ETIQUETADOR_CARPETA", str(tmp_path))
    monkeypatch.setenv("ETIQUETADOR_INICIALES", "TS")
    return tmp_path


def _boton(at, texto):
    return next(b for b in at.button if b.label == texto)


def test_labels_are_saved_and_the_session_resumes(carpeta):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, at.exception
    assert [b.label for b in at.button[:4]] == ["Buena", "Regular", "Mala", "Descartar"]
    _boton(at, "Buena").click().run()
    _boton(at, "Descartar").click().run()
    guardado = pd.read_csv(carpeta / "etiquetas_mixed_TS.csv")
    assert guardado["crop_id"].tolist() == ["apple_IMG0_00", "apple_IMG1_00"]
    assert guardado["calidad"].tolist() == ["Buena", "Descartar"]
    # sesión nueva (otra persona de la misma máquina, mismas iniciales): retoma en el recorte 3
    nuevo = AppTest.from_file(str(APP), default_timeout=60).run()
    assert "2 de 3 etiquetados" in str(nuevo.get("progress")[0].proto)
    _boton(nuevo, "Mala").click().run()                        # la etiqueta cae en el tercer recorte, no en el primero
    assert pd.read_csv(carpeta / "etiquetas_mixed_TS.csv")["calidad"].tolist() == ["Buena", "Descartar", "Mala"]
    assert any("Terminaste" in s.value for s in nuevo.success)


def test_each_annotator_has_a_separate_file_and_never_sees_the_other(carpeta, monkeypatch):
    AppTest.from_file(str(APP), default_timeout=60).run()
    at_a = AppTest.from_file(str(APP), default_timeout=60).run()
    _boton(at_a, "Regular").click().run()
    monkeypatch.setenv("ETIQUETADOR_INICIALES", "JP")
    at_b = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at_b.info or all("etiqueta actual" not in i.value for i in at_b.info)   # no ve lo del otro
    _boton(at_b, "Mala").click().run()
    a = pd.read_csv(carpeta / "etiquetas_mixed_TS.csv")["calidad"].tolist()
    b = pd.read_csv(carpeta / "etiquetas_mixed_JP.csv")["calidad"].tolist()
    assert a == ["Regular"] and b == ["Mala"]


def test_missing_initials_or_folder_show_a_clear_message(carpeta, monkeypatch, tmp_path):
    monkeypatch.setenv("ETIQUETADOR_INICIALES", "")
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception and any("iniciales" in i.value for i in at.info)
    monkeypatch.setenv("ETIQUETADOR_INICIALES", "TS")
    monkeypatch.setenv("ETIQUETADOR_CARPETA", str(tmp_path / "no_existe"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception and any("muestra_etiquetado.csv" in e.value for e in at.error)
