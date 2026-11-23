"""Pruebas de ``src/data_utils.py`` y ``src/splits.py`` con datos sintéticos.

Ejecutar desde la raíz del repo:  ``python -m pytest tests -q``
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data_utils as du  # noqa: E402
import splits  # noqa: E402

# Nombres reales observados en Banana_Good (dos "frutas" separadas por un salto de 5 s)
NAMES = ["IMG20200729190220.jpg", "IMG20200729190220_01.jpg", "IMG20200729190221.jpg",
         "IMG20200729190222.jpg", "IMG20200729190223.jpg", "IMG20200729190224.jpg",
         "IMG20200729190229.jpg", "IMG20200729190230.jpg", "IMG20200729190231.jpg",
         "IMG20200729190231_01.jpg", "IMG20200729190232.jpg", "IMG20200729190233.jpg"]


def _df_from_names(names):
    rows = []
    for n in names:
        ts, suf = du.parse_timestamp(n)
        rows.append({"path": n, "filename": n, "timestamp": ts, "dup_suffix": suf})
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_parse_timestamp():
    ts, suf = du.parse_timestamp("IMG20200729190220_01.jpg")
    assert (ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second) == (2020, 7, 29, 19, 2, 20)
    assert suf == 1
    assert du.parse_timestamp("IMG20200729190220.jpg")[1] == 0
    assert du.parse_timestamp("banana_01.png") == (None, 0)


def test_assign_groups_splits_on_gap():
    grouped = du.assign_groups(_df_from_names(NAMES), gap_seconds=3)
    assert grouped["group"].nunique() == 2
    assert sorted(grouped.groupby("group").size()) == [6, 6]
    # con un umbral muy grande todo es un solo grupo
    assert du.assign_groups(_df_from_names(NAMES), gap_seconds=60)["group"].nunique() == 1


def test_images_without_timestamp_get_own_group():
    df = _df_from_names(NAMES[:3] + ["foto_rara.jpg", "otra.jpg"])
    grouped = du.assign_groups(df, gap_seconds=3)
    odd = grouped[grouped["filename"].isin(["foto_rara.jpg", "otra.jpg"])]
    assert odd["group"].nunique() == 2
    assert grouped[~grouped["filename"].isin(odd["filename"])]["group"].nunique() == 1


def test_list_images_and_stats(tmp_path):
    for i, name in enumerate(NAMES[:4]):
        Image.fromarray(np.full((40, 60, 3), 30 * i + 60, dtype=np.uint8)).save(tmp_path / name)
    (tmp_path / "notas.txt").write_text("no es imagen")
    df = du.list_images(tmp_path)
    assert len(df) == 4 and df["timestamp"].notna().all()
    stats = du.image_stats(df.loc[0, "path"])
    assert (stats["width"], stats["height"]) == (60, 40)
    assert 0 <= stats["sat"] <= 1 and 0 <= stats["val"] <= 1
    with pytest.raises(FileNotFoundError):
        du.list_images(tmp_path / "no_existe")


def test_ahash_near_duplicates(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(base).save(tmp_path / "a.png")
    Image.fromarray(base).save(tmp_path / "a_copy.png")
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp_path / "b.png")
    hashes = {p.name: du.ahash(p) for p in tmp_path.iterdir()}
    pairs = du.near_duplicates(hashes, max_dist=2)
    assert any({a, b} == {"a.png", "a_copy.png"} for a, b, _ in pairs)


def test_grouped_split_has_no_leakage_and_keeps_classes():
    rng = np.random.default_rng(1)
    rows = []
    for label, n_groups in (("Buena", 40), ("Mala", 40), ("Regular", 10)):
        for g in range(n_groups):
            for _ in range(5):                       # 5 ángulos por banana
                rows.append({"label": label, "group": f"{label}_{g}"})
    df = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    train, test = splits.grouped_split(df, test_size=0.2, seed=3)
    splits.assert_no_group_overlap(train, test)
    assert set(test["label"]) == {"Buena", "Mala", "Regular"}
    tr, va, te = splits.grouped_train_val_test(df, seed=3)
    assert len(tr) + len(va) + len(te) == len(df)
    splits.assert_no_group_overlap(tr, te)


def test_parse_timestamp_second_camera_format():
    ts, suf = du.parse_timestamp("IMG_20190910_175131_1.jpg")
    assert (ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second) == (2019, 9, 10, 17, 51, 31)
    assert suf == 1
    assert du.parse_timestamp("IMG_20190910_175131.jpg")[1] == 0
    # contadores sin fecha: no se puede inferir un instante
    for sin_fecha in ("IMG_0111.JPG", "123_45.jpg", "7_1_2.jpg"):
        assert du.parse_timestamp(sin_fecha) == (None, 0)


def test_parse_timestamp_without_img_prefix_and_with_counter_suffix():
    ts, suf = du.parse_timestamp("20190809_115439.jpg")
    assert (ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second) == (2019, 8, 9, 11, 54, 39)
    assert suf == 0
    ts, suf = du.parse_timestamp("20190820_143522_22896.jpg")
    assert (ts.hour, ts.minute, ts.second) == (14, 35, 22) and suf == 22896


def test_parse_counter_only_for_plain_img_counters():
    assert du.parse_counter("IMG_0111.JPG") == 111
    assert du.parse_counter("IMG_8131.jpg") == 8131
    for otro in ("IMG_20190910_175131.jpg", "20190809_115439.jpg", "IMG20200729190220.jpg", "foto.jpg"):
        assert du.parse_counter(otro) is None


def test_name_pattern_counts_conventions():
    assert du.name_pattern("IMG_0111.JPG") == "IMG_N.jpg"
    assert du.name_pattern("IMG20200729190220_01.jpg") == "IMGN_N.jpg"
    assert du.name_pattern("12_345_6.jpg") == "N_N_N.jpg"


def _make_tree(root):
    """Árbol mínimo con la estructura original de Kaggle (nombres reales de carpeta)."""
    layout = {
        "Good Quality_Fruits/Apple_Good": ["IMG_0111.JPG", "IMG_0112.JPG"],
        "Bad Quality_Fruits/Banana_Bad": ["IMG_20190910_175131.jpg"],
        "Mixed Qualit_Fruits/Lemon": ["IMG20200729183039.jpg"],
        "Mixed Qualit_Fruits/Banana": ["IMG20200729183040.jpg", "leeme.txt"],
        "Otra carpeta/Apple_Good": ["IMG_9999.JPG"],          # no es carpeta de calidad
    }
    for folder, names in layout.items():
        (root / folder).mkdir(parents=True)
        for n in names:
            if n.endswith(".txt"):
                (root / folder / n).write_text("no es imagen")
            else:
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / folder / n)


def test_index_dataset_species_source_and_weak_labels(tmp_path):
    _make_tree(tmp_path)
    df = du.index_dataset(tmp_path)
    assert list(df.columns) == du.INDEX_COLUMNS
    assert len(df) == 5                                   # ignora .txt y la carpeta ajena
    por_especie = df.groupby(["source", "species"]).size().to_dict()
    assert por_especie == {("good", "apple"): 2, ("bad", "banana"): 1,
                           ("mixed", "lemon"): 1, ("mixed", "banana"): 1}
    assert df[df["source"] == "good"]["label_weak"].tolist() == ["Buena", "Buena"]
    assert df[df["source"] == "bad"]["label_weak"].tolist() == ["Mala"]
    assert df[df["source"] == "mixed"]["label_weak"].isna().all()
    # solo los nombres con fecha traen timestamp; solo los IMG_NNNN traen contador
    por_nombre = df.set_index("filename")
    assert por_nombre.loc["IMG_0111.JPG", "timestamp"] is pd.NaT
    assert por_nombre.loc["IMG_20190910_175131.jpg", "timestamp"].year == 2019
    assert por_nombre.loc["IMG_0111.JPG", "counter"] == 111
    assert pd.isna(por_nombre.loc["IMG_20190910_175131.jpg", "counter"])


def test_index_dataset_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        du.index_dataset(tmp_path / "no_existe")


def test_overlap_is_detected():
    a = pd.DataFrame({"group": ["x", "y"]})
    b = pd.DataFrame({"group": ["y", "z"]})
    with pytest.raises(AssertionError):
        splits.assert_no_group_overlap(a, b)
