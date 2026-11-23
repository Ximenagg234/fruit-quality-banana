"""Pruebas de ``src/labeling.py``: guardado atómico, reanudación, deshacer y acuerdo entre anotadores."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import labeling as lb  # noqa: E402


def test_initials_and_paths(tmp_path):
    assert lb.normalizar_iniciales(" n.d ") == "ND"
    assert lb.ruta_etiquetas(tmp_path, "jp").name == "etiquetas_mixed_JP.csv"
    with pytest.raises(ValueError):
        lb.normalizar_iniciales("  ..  ")


def test_save_overwrite_resume_and_undo(tmp_path):
    ruta = lb.ruta_etiquetas(tmp_path, "NP")
    assert lb.cargar_etiquetas(ruta) == {}
    lb.guardar_etiqueta(ruta, "a_01", "Buena", "se ve sana")
    lb.guardar_etiqueta(ruta, "a_02", "Mala")
    lb.guardar_etiqueta(ruta, "a_01", "Regular", " con golpe ")       # reemplaza, no duplica
    et = lb.cargar_etiquetas(ruta)
    assert et == {"a_01": {"calidad": "Regular", "notas": "con golpe"}, "a_02": {"calidad": "Mala", "notas": ""}}
    lb.borrar_etiqueta(ruta, "a_02"); lb.borrar_etiqueta(ruta, "no_existe")
    assert list(lb.cargar_etiquetas(ruta)) == ["a_01"]
    assert not list(tmp_path.glob("*.tmp"))                            # sin archivos temporales a medias
    with pytest.raises(ValueError):
        lb.guardar_etiqueta(ruta, "a_03", "Excelente")


def test_notes_with_commas_quotes_and_accents_survive(tmp_path):
    ruta = lb.ruta_etiquetas(tmp_path, "X")
    nota = 'moho, "verde" y golpe; ¿dudosa?'
    lb.guardar_etiqueta(ruta, "z", "Mala", nota)
    assert lb.cargar_etiquetas(ruta)["z"]["notas"] == nota


def test_first_pending_and_progress():
    muestra = pd.DataFrame({"crop_id": ["a", "b", "c"], "orden": [1, 2, 3]})
    assert lb.primer_pendiente(muestra, {}) == 0
    assert lb.primer_pendiente(muestra, {"a": {}, "c": {}}) == 1           # salta lo hecho y vuelve al hueco
    assert lb.primer_pendiente(muestra, {"a": {}, "b": {}, "c": {}}) is None
    assert lb.progreso(muestra, {"a": {}, "zzz": {}}) == (1, 3)


def test_load_sample_sorted_and_missing_file(tmp_path):
    pd.DataFrame({"crop_id": ["b", "a"], "orden": [2, 1]}).to_csv(tmp_path / "muestra_etiquetado.csv", index=False)
    assert list(lb.cargar_muestra(tmp_path)["crop_id"]) == ["a", "b"]
    with pytest.raises(FileNotFoundError):
        lb.cargar_muestra(tmp_path / "otra")


def _tabla(pares):
    return pd.DataFrame({"crop_id": [f"c{i}" for i in range(len(pares))], "calidad": pares})


def test_agreement_perfect_partial_and_discarded():
    a = _tabla(["Buena", "Regular", "Mala", "Buena", "Descartar"])
    perfecto = lb.acuerdo(a, a)
    assert perfecto["kappa_w"] == pytest.approx(1.0) and perfecto["concordancia"] == 1.0 and perfecto["desacuerdos"].empty
    b = _tabla(["Buena", "Mala", "Mala", "Regular", "Buena"])           # difiere en 3 ítems; el último, solo uno descartó
    r = lb.acuerdo(a, b)
    assert r["n_comunes"] == 5 and r["n_validos"] == 4                 # lo descartado no entra al kappa
    assert len(r["desacuerdos"]) == 3 and r["concordancia"] == pytest.approx(2 / 5)
    # kappa ponderado cuadrático castiga más Buena<->Mala que Buena<->Regular
    cerca = lb.acuerdo(_tabla(["Buena", "Regular", "Mala", "Buena"]), _tabla(["Regular", "Regular", "Mala", "Buena"]))
    lejos = lb.acuerdo(_tabla(["Buena", "Regular", "Mala", "Buena"]), _tabla(["Mala", "Regular", "Mala", "Buena"]))
    assert cerca["kappa_w"] > lejos["kappa_w"]
    assert r["matriz"].shape == (4, 4)


def test_agreement_handles_degenerate_cases():
    u = _tabla(["Buena", "Buena", "Buena"])
    assert np.isnan(lb.acuerdo(u, u)["kappa_w"])                       # una sola clase: kappa no definido, no rompe
    vacio = lb.acuerdo(_tabla(["Buena"]), pd.DataFrame({"crop_id": ["otro"], "calidad": ["Mala"]}))
    assert vacio["n_comunes"] == 0 and np.isnan(vacio["kappa_w"])
