"""Pruebas de ``src/unified_dataset.py`` con datos sintéticos."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unified_dataset as ud  # noqa: E402


def _tabla(ids, cal):
    return pd.DataFrame({"crop_id": ids, "calidad": cal})


def test_resolve_labels_agreement_resolution_and_exclusions():
    a = _tabla(["c1", "c2", "c3", "c4", "c5"], ["Buena", "Regular", "Mala", "Descartar", "Buena"])
    b = _tabla(["c1", "c2", "c3", "c4", "c5"], ["Buena", "Mala", "Mala", "Buena", "Regular"])
    resuelto = pd.DataFrame({"crop_id": ["c2"], "calidad_final": ["Regular"]})     # c5 no se resolvió
    out = ud.resolve_labels(a, b, resuelto).set_index("crop_id")
    assert out["calidad"].to_dict() == {"c1": "Buena", "c2": "Regular", "c3": "Mala"}
    assert out["origen"].to_dict() == {"c1": "acuerdo", "c2": "resuelto", "c3": "acuerdo"}
    # c4 (descartado por uno) y c5 (desacuerdo sin resolver) quedan fuera: no se adivina
    assert set(ud.resolve_labels(a, b)["crop_id"]) == {"c1", "c3"}
    # una resolución que dice "Descartar" o algo inválido tampoco entra
    invalida = pd.DataFrame({"crop_id": ["c2"], "calidad_final": ["Descartar"]})
    assert "c2" not in set(ud.resolve_labels(a, b, invalida)["crop_id"])


def test_from_kaggle_keeps_only_good_and_bad_as_weak_labels():
    part = pd.DataFrame({"path": ["a", "b", "c"], "species": ["apple", "apple", "lime"],
                         "label_weak": ["Buena", "Mala", None], "group": ["g1", "g2", "g3"]})
    k = ud.from_kaggle(part)
    assert list(k["path"]) == ["a", "b"] and set(k["label_origin"]) == {"debil_kaggle"} and set(k["source"]) == {"kaggle"}
    with pytest.raises(KeyError):
        ud.from_kaggle(part.drop(columns="group"))


def test_from_mixed_uses_photo_group_and_crop_paths(tmp_path):
    final = pd.DataFrame({"crop_id": ["apple_IMG1_00", "apple_IMG1_01"], "calidad": ["Regular", "Mala"], "origen": ["acuerdo", "resuelto"]})
    man = pd.DataFrame({"crop_id": ["apple_IMG1_00", "apple_IMG1_01", "otro"], "species": ["apple"] * 3, "group": ["g7", "g7", "g9"]})
    m = ud.from_mixed(final, man, tmp_path)
    assert set(m["group"]) == {"g7"}                                    # las frutas de una foto comparten grupo
    assert m["path"].iloc[0].endswith("recortes\\apple\\apple_IMG1_00.jpg") or m["path"].iloc[0].endswith("recortes/apple/apple_IMG1_00.jpg")
    assert set(m["label_origin"]) == {"manual_acuerdo", "manual_resuelto"}


def test_from_own_groups_by_object_and_ignores_unlabeled(tmp_path):
    ann = pd.DataFrame({"image_id": ["B01_d0_top.jpg", "B01_d0_side.jpg", "B02_d0_top.jpg"], "species": ["banana"] * 3,
                        "object_id": ["B01", "B01", "B02"], "quality_final": ["Regular", "Regular", ""]})
    own = ud.from_own(ann, tmp_path)
    assert len(own) == 2 and set(own["group"]) == {"own_B01"}
    with pytest.raises(KeyError):
        ud.from_own(ann.drop(columns="object_id"), tmp_path)


def _fuente(src, etiquetas, esp="apple"):
    return pd.DataFrame({"path": [f"{src}_{i}" for i in range(len(etiquetas))], "species": esp, "label": etiquetas,
                         "source": src, "group": [f"{src}_g{i}" for i in range(len(etiquetas))], "label_origin": "x"})


def test_combine_validates_labels_and_duplicates_and_maps_lemon():
    k, m = _fuente("kaggle", ["Buena"] * 6 + ["Mala"] * 6), _fuente("mixed", ["Regular"] * 6 + ["Buena"] * 6, esp="lemon")
    df = ud.combine(k, m)
    assert len(df) == 24 and set(df["species"]) == {"apple", "lemon"}
    assert set(ud.combine(k, m, lemon_as_lime=True)["species"]) == {"apple", "lime"}
    with pytest.raises(ValueError, match="etiquetas"):
        ud.combine(k, _fuente("x", ["Excelente"]))
    with pytest.raises(ValueError, match="repetidas"):
        ud.combine(k, k)


def test_source_class_report_flags_the_regular_only_in_one_source_risk():
    k = _fuente("kaggle", ["Buena"] * 8 + ["Mala"] * 8)                  # sin Regular (como en Kaggle real)
    m = _fuente("mixed", ["Regular"] * 8 + ["Buena"] * 8 + ["Mala"] * 8)
    o = _fuente("own", ["Regular"] * 6 + ["Buena"] * 6 + ["Mala"] * 6)
    rep = ud.source_class_report(ud.combine(k, m, o))
    assert rep.loc["kaggle", "completa"] == False and rep.loc["mixed", "completa"] and rep.loc["own", "completa"]   # noqa: E712
    assert rep.loc["kaggle", "Regular"] == 0
    esp = ud.species_class_report(ud.combine(k, m, o))
    assert esp.loc["apple", "completa"]
