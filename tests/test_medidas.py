import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import medidas as md  # noqa: E402


def test_parsear_cm_acepta_coma_punto_y_unidad():
    assert md.parsear_cm("12,5") == 12.5
    assert md.parsear_cm(" 7.4 cm ") == 7.4
    assert md.parsear_cm("18") == 18.0


@pytest.mark.parametrize("malo", ["", "abc", "0", "-3", "120", "nan"])
def test_parsear_cm_rechaza_valores_invalidos(malo):
    with pytest.raises(ValueError):
        md.parsear_cm(malo)


def test_guardar_y_cargar_medidas_con_no_medida_y_reemplazo(tmp_path):
    ruta = md.ruta_medidas(tmp_path, "xg")
    assert ruta.name == "medidas_fotos_XG.csv" and md.cargar_medidas(ruta) == {}
    md.guardar_medida(ruta, "F001", 12.5, "central")
    md.guardar_medida(ruta, "F002", None)
    md.guardar_medida(ruta, "F001", 13.0)                       # reemplaza
    cargado = md.cargar_medidas(ruta)
    assert cargado["F001"] == {"max_dim_cm": "13", "notas": ""}
    assert cargado["F002"]["max_dim_cm"] == ""
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError):
        md.guardar_medida(ruta, "F003", 500)


def test_primer_pendiente_cuenta_no_medida_como_registrada():
    ids = ["F001", "F002", "F003"]
    assert md.primer_pendiente(ids, {"F001": {}, "F002": {}}) == 2
    assert md.primer_pendiente(ids, {i: {} for i in ids}) is None
