"""Prueba de humo de la app Streamlit con el arnés ``AppTest`` (sin navegador).

Comprueba que la app arranca sin errores con el modelo incluido en el repositorio, que
ofrece las dos entradas exigidas (cargar imagen y usar la cámara) y que, sin imagen, solo
pide una. La lógica de predicción y de tamaño se prueba en ``test_inference.py``.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
APP = RAIZ / "06_despliegue" / "app.py"
MODELO = RAIZ / "06_despliegue" / "modelo" / "calidad_cnn.keras"


@pytest.mark.skipif(not MODELO.exists(), reason="falta el modelo de la app (se genera en 04_modelado)")
def test_app_starts_and_offers_upload_and_camera():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Calidad y tamaño de frutas"
    assert at.radio[0].options == ["Subir imagen", "Usar la cámara"]
    assert len(at.get("file_uploader")) == 1 and len(at.get("camera_input")) == 0   # la cámara no se monta sola
    assert any("Esperando una imagen" in i.value for i in at.info)       # sin imagen no predice nada
    at.radio[0].set_value("Usar la cámara").run()                        # al elegirla, aparece el widget de cámara
    assert not at.exception, at.exception
    assert len(at.get("camera_input")) == 1 and len(at.get("file_uploader")) == 0
    assert not at.error                                                  # el modelo se encontró


def test_app_reports_a_missing_model_without_crashing(monkeypatch, tmp_path):
    """Sin el archivo del modelo, la app muestra un error claro y se detiene (no lanza excepción)."""
    import streamlit as st
    st.cache_resource.clear()          # la prueba anterior dejó el modelo real en la caché de recursos
    copia = tmp_path / "06_despliegue"
    copia.mkdir()
    (copia / "app.py").write_text(APP.read_text(encoding="utf-8").replace(
        "REPO = Path(__file__).resolve().parents[1]", f"REPO = Path(r'{RAIZ}')"), encoding="utf-8")
    at = AppTest.from_file(str(copia / "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert any("No se encontró el modelo" in e.value for e in at.error)
