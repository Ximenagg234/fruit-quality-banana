"""Pruebas de ``src/groups.py`` y de la estratificación multifruta de ``src/splits.py``.

Datos sintéticos: no dependen del dataset real. Ejecutar con ``python -m pytest tests -q``.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import groups as gr  # noqa: E402
import splits  # noqa: E402

T0 = pd.Timestamp("2020-07-29 19:02:20")


def _bits(seed: int, n_bytes: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, n_bytes, dtype=np.uint8)


def test_hash_edges_finds_only_close_pairs():
    a = _bits(0)
    b = a.copy(); b[0] ^= 0b00000011            # 2 bits distintos de a
    c = _bits(1)                                # aleatorio: lejos de a y b
    packed = np.vstack([a, b, c])
    i, j = gr.hash_edges(packed, max_dist=2)
    assert set(zip(i.tolist(), j.tolist())) == {(0, 1)}
    i, j = gr.hash_edges(packed, max_dist=1)    # con umbral 1, a y b ya no se unen
    assert len(i) == 0


def test_hash_edges_chunking_gives_same_result():
    packed = np.vstack([_bits(s) for s in range(20)] + [_bits(0)])   # el último repite al 0
    full = gr.hash_edges(packed, max_dist=0, chunk=1000)
    small = gr.hash_edges(packed, max_dist=0, chunk=3)
    assert set(zip(*map(np.ndarray.tolist, full))) == set(zip(*map(np.ndarray.tolist, small))) == {(0, 20)}


def test_time_edges_splits_on_gap_and_ignores_missing():
    ts = pd.Series([T0, T0 + pd.Timedelta(seconds=1), pd.NaT,
                    T0 + pd.Timedelta(seconds=30), T0 + pd.Timedelta(seconds=31)])
    i, j = gr.time_edges(ts, gap_seconds=3)
    assert set(zip(i.tolist(), j.tolist())) == {(0, 1), (3, 4)}
    assert len(gr.time_edges(pd.Series([pd.NaT, pd.NaT]), 3)[0]) == 0


def _index(rows):
    return pd.DataFrame(rows, columns=["species", "source", "timestamp"])


def test_build_groups_chains_time_and_hash():
    # 0~1 por tiempo; 1~2 por hash (la 2 no tiene fecha); 3 está sola
    df = _index([("apple", "good", T0), ("apple", "good", T0 + pd.Timedelta(seconds=1)),
                 ("apple", "good", pd.NaT), ("apple", "good", pd.NaT)])
    base = _bits(5)
    packed = np.vstack([_bits(10), base, base, _bits(11)])
    out = gr.build_groups(df, packed, gap_seconds=3, max_hash_dist=0)
    assert out.loc[0, "group"] == out.loc[1, "group"] == out.loc[2, "group"]
    assert out.loc[3, "group"] != out.loc[0, "group"]
    assert out["group_size"].tolist() == [3, 3, 3, 1]


def test_build_groups_never_merges_across_cells():
    # misma hora e hash idéntico, pero distinta especie o distinta fuente
    df = _index([("apple", "good", T0), ("banana", "good", T0), ("apple", "bad", T0)])
    packed = np.vstack([_bits(7)] * 3)
    out = gr.build_groups(df, packed, gap_seconds=3, max_hash_dist=0)
    assert out["group"].nunique() == 3


def test_build_groups_without_signals_gives_singletons_and_validates_input():
    df = _index([("apple", "good", T0), ("apple", "good", T0)])
    assert gr.build_groups(df, None, gap_seconds=None, max_hash_dist=None)["group"].nunique() == 2
    with pytest.raises(ValueError):
        gr.build_groups(df, np.zeros((5, 8), dtype=np.uint8))


def test_counter_edges_require_consecutive_counter_and_close_hash():
    base = _bits(4)
    near = base.copy(); near[0] ^= 0b00000111           # 3 bits de diferencia
    packed = np.vstack([base, near, _bits(9), _bits(9)])
    counters = pd.Series([100, 101, 102, 150], dtype="Int64")
    i, j = gr.counter_edges(counters, packed, max_gap=1, max_dist=5)
    # (100,101): consecutivos y hash cercano -> sí; (101,102): consecutivos pero hash lejano -> no;
    # (102,150): hash idéntico pero contadores lejanos -> no
    assert set(zip(i.tolist(), j.tolist())) == {(0, 1)}
    assert len(gr.counter_edges(pd.Series([pd.NA, pd.NA], dtype="Int64"), packed[:2])[0]) == 0


def test_build_groups_uses_counter_only_when_asked():
    df = _index([("apple", "good", pd.NaT)] * 3)
    df["counter"] = pd.Series([10, 11, 12], dtype="Int64")
    base = _bits(2)
    near = base.copy(); near[0] ^= 0b00000011
    packed = np.vstack([base, near, _bits(8)])
    assert gr.build_groups(df, packed, gap_seconds=None, max_hash_dist=None)["group"].nunique() == 3
    con = gr.build_groups(df, packed, gap_seconds=None, max_hash_dist=None, counter_hash_dist=5)
    assert con["group"].nunique() == 2 and con["group_size"].max() == 2


def test_grouping_sensitivity_is_monotonic_in_threshold():
    df = _index([("apple", "good", T0 + pd.Timedelta(seconds=s)) for s in (0, 2, 4, 20, 22, 60)])
    sens = gr.grouping_sensitivity(df, None, [(1, None), (3, None), (30, None)])
    assert sens["n_groups"].tolist() == [6, 3, 2]
    assert sens["max_size"].tolist() == [1, 3, 5]


def test_hash_matrix_identical_images_and_shapes(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(base).save(tmp_path / "a.png")
    Image.fromarray(base).save(tmp_path / "a_copia.png")
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp_path / "b.png")
    packed = gr.hash_matrix([tmp_path / "a.png", tmp_path / "a_copia.png", tmp_path / "b.png"])
    assert packed.shape == (3, 32)               # 16x16 bits = 32 bytes
    i, j = gr.hash_edges(packed, max_dist=0)
    assert set(zip(i.tolist(), j.tolist())) == {(0, 1)}
    assert gr.hash_matrix([]).shape == (0, 32)
    with pytest.raises(ValueError):
        gr.hash_matrix([tmp_path / "a.png"], size=3)


def test_stratified_split_by_species_and_quality_keeps_every_cell():
    rows = []
    for especie in ("apple", "banana", "pomegranate"):
        for calidad, n_groups in (("Buena", 30), ("Mala", 30), ("Regular", 10)):
            for g in range(n_groups):
                rows += [{"species": especie, "label": calidad, "group": f"{especie}_{calidad}_{g}"}] * 4
    df = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    df["strata"] = splits.make_strata(df, ["species", "label"])
    tr, va, te = splits.grouped_train_val_test(df, seed=2, strata_col="strata")
    assert len(tr) + len(va) + len(te) == len(df)
    splits.assert_no_group_overlap(tr, te)
    splits.assert_no_group_overlap(tr, va)
    for parte in (tr, va, te):                   # cada partición conserva las 9 celdas
        assert parte["strata"].nunique() == 9
