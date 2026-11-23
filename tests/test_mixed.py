"""Pruebas de ``src/mixed.py`` con escenas sintéticas de frutas conocidas."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mixed as mx  # noqa: E402

NEUTRO = (235, 235, 230)


def _escena(discos, shape=(192, 256), bg=NEUTRO, color=(215, 120, 30)):
    """Fondo liso con discos ``(cy, cx, r)``."""
    img = np.empty((*shape, 3), dtype=np.uint8)
    img[:] = bg
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for cy, cx, r in discos:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = color
    return img


def test_three_separate_fruits_give_three_instances():
    inst, metodo = mx.extract_instances(_escena([(50, 50, 20), (50, 130, 20), (130, 90, 25)]))
    assert metodo == "saturation" and len(inst) == 3
    assert all(i["solidity"] > 0.9 and not i["touches_border"] for i in inst)
    # el orden es estable: de arriba abajo y de izquierda a derecha
    assert [i["bbox"][1] for i in inst] == sorted(i["bbox"][1] for i in inst)


def test_touching_fruits_are_split_with_watershed():
    # dos discos de radio 22 que se solapan un poco: una sola mancha conectada
    inst, _ = mx.extract_instances(_escena([(96, 100, 22), (96, 138, 22)]))
    assert len(inst) == 2
    centros = sorted((i["bbox"][0] + i["bbox"][2]) / 2 for i in inst)
    assert centros[0] == pytest.approx(100, abs=8) and centros[1] == pytest.approx(138, abs=8)


def test_coloured_background_uses_the_distance_method():
    img = _escena([(96, 128, 30)], bg=(0, 170, 200), color=(230, 210, 40))
    assert mx.choose_method(img) == "distance"
    inst, metodo = mx.extract_instances(img)
    assert metodo == "distance" and len(inst) == 1
    assert mx.choose_method(_escena([(96, 128, 30)])) == "saturation"


def test_border_flag_and_area_filters():
    inst, _ = mx.extract_instances(_escena([(10, 128, 25), (140, 80, 20)]))
    banderas = {tuple(i["bbox"]): i["touches_border"] for i in inst}
    assert sorted(banderas.values()) == [False, True]                 # una fruta toca el borde superior
    # una mota diminuta se descarta por área
    assert len(mx.extract_instances(_escena([(96, 128, 30), (30, 30, 2)]))[0]) == 1
    # sin contraste: no hay instancias
    assert mx.extract_instances(_escena([]))[0] == []


def test_crop_square_shape_and_stays_inside_the_image():
    img = _escena([(96, 128, 30)])
    assert mx.crop_square(img, (98, 66, 158, 126)).size == (128, 128)
    esquina = mx.crop_square(img, (0, 0, 40, 40), out_size=64)        # fruta pegada a la esquina
    assert esquina.size == (64, 64)
    assert np.asarray(esquina).std() >= 0                              # no lanza ni devuelve vacío


def test_quality_flags_reject_dark_strips_background_patches_and_odd_shapes():
    ok = {"solidity": 0.95, "area_frac": 0.02, "touches_border": False, "mean_v": 0.7, "bg_dist": 120.0,
          "aspect": 1.3, "min_side": 40}
    assert all(mx.quality_flags(ok).values())
    assert not mx.quality_flags({**ok, "mean_v": 0.05})["color_ok"]              # franja negra
    assert not mx.quality_flags({**ok, "bg_dist": 10.0})["color_ok"]             # parche del color del fondo
    assert not mx.quality_flags({**ok, "aspect": 9.0})["forma_ok"]               # tira delgada
    assert not mx.quality_flags({**ok, "min_side": 5})["forma_ok"]
    assert not mx.quality_flags({**ok, "solidity": 0.5})["compacta"]
    assert not mx.quality_flags({**ok, "touches_border": True})["no_borde"]
    assert not mx.quality_flags({**ok, "area_frac": 0.001})["grande"]


def test_instances_carry_color_and_shape_metrics():
    inst, _ = mx.extract_instances(_escena([(96, 128, 30)]))
    i = inst[0]
    assert i["mean_v"] > 0.6 and i["bg_dist"] > 80 and i["aspect"] == pytest.approx(1.0, abs=0.15) and i["min_side"] >= 50
    # una "franja negra" sobre fondo cian se detecta pero no pasa el filtro de color
    img = _escena([(96, 128, 30)], bg=(0, 170, 200), color=(230, 210, 40))
    img[0:34, :, :] = 5
    todas, _ = mx.extract_instances(img)
    franjas = [x for x in todas if x["bbox"][1] == 0]
    assert franjas and not any(mx.quality_flags(x)["color_ok"] for x in franjas)


def _manifiesto():
    filas = []
    base = {"compacta": True, "no_borde": True, "grande": True, "color_ok": True, "forma_ok": True, "n_inst_imagen": 6}
    for esp in ("apple", "banana", "lemon"):
        for k in range(40):
            filas.append({**base, "crop_id": f"{esp}_{k}", "species": esp, "source_image": f"{esp}_{k // 5}.jpg"})
    filas.append({**base, "crop_id": "malo1", "species": "apple", "source_image": "x.jpg", "compacta": False})
    filas.append({**base, "crop_id": "malo2", "species": "apple", "source_image": "y.jpg", "no_borde": False})
    filas.append({**base, "crop_id": "ruido", "species": "apple", "source_image": "z.jpg", "n_inst_imagen": 26})
    return pd.DataFrame(filas)


def test_label_sample_is_reproducible_balanced_and_diverse():
    m = _manifiesto()
    a = mx.pick_label_sample(m, per_species=10, max_per_image=2, seed=1)
    b = mx.pick_label_sample(m, per_species=10, max_per_image=2, seed=1)
    assert a.equals(b)                                                  # reproducible: mismo orden para todos los anotadores
    assert a["species"].value_counts().to_dict() == {"apple": 10, "banana": 10, "lemon": 10}
    assert a.groupby("source_image").size().max() <= 2                  # diversidad: a lo sumo 2 por foto
    assert not {"malo1", "malo2", "ruido"} & set(a["crop_id"])          # no compactos, de borde o de fotos ruidosas: fuera
    assert list(a["orden"]) == list(range(1, 31))
    with pytest.raises(KeyError):
        mx.pick_label_sample(m.drop(columns="color_ok"))
