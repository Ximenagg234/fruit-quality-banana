"""Pruebas de ``src/prepare_photos.py`` con fotos sintéticas que traen EXIF y GPS."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prepare_photos as pp  # noqa: E402


def _photo_with_metadata(path, size=(300, 200), orientation=None):
    """JPEG sintético con marca de cámara y un GPS falso en el EXIF."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, : size[0] // 2] = (200, 30, 30)                 # mitad izquierda roja
    im = Image.fromarray(arr)
    exif = Image.Exif()
    exif[0x010F] = "MarcaDeCamara"
    exif[0x0112] = orientation or 1
    exif.get_ifd(0x8825)[1] = "N"                           # GPS: referencia de latitud
    im.save(path, format="JPEG", exif=exif)


def test_has_metadata_detects_exif_and_clean_files(tmp_path):
    _photo_with_metadata(tmp_path / "con.jpg")
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(tmp_path / "limpia.jpg")
    assert pp.has_metadata(tmp_path / "con.jpg") is True
    assert pp.has_metadata(tmp_path / "limpia.jpg") is False


def test_prepare_photo_strips_metadata_and_downsizes(tmp_path):
    _photo_with_metadata(tmp_path / "o.jpg", size=(2000, 1500))
    size = pp.prepare_photo(tmp_path / "o.jpg", tmp_path / "salida" / "a.jpg", max_side=1000)
    assert size == (1000, 750)
    assert not pp.has_metadata(tmp_path / "salida" / "a.jpg")
    with Image.open(tmp_path / "salida" / "a.jpg") as im:
        assert im.size == (1000, 750) and im.format == "JPEG"


def test_prepare_photo_never_upscales(tmp_path):
    _photo_with_metadata(tmp_path / "o.jpg", size=(300, 200))
    assert pp.prepare_photo(tmp_path / "o.jpg", tmp_path / "a.jpg", max_side=1024) == (300, 200)


def test_orientation_is_applied_before_removing_exif(tmp_path):
    # orientación 6 = hay que girar 90° para verla bien: 300x200 pasa a 200x300 y el rojo queda arriba
    _photo_with_metadata(tmp_path / "o.jpg", size=(300, 200), orientation=6)
    size = pp.prepare_photo(tmp_path / "o.jpg", tmp_path / "a.jpg")
    assert size == (200, 300)
    with Image.open(tmp_path / "a.jpg") as im:
        assert np.asarray(im)[10, 100, 0] > 150 and np.asarray(im)[290, 100, 0] < 60


def _map(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_prepare_folder_writes_annotations_and_appends(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    for n in ("x1.jpg", "x2.jpg", "x3.jpg"):
        _photo_with_metadata(src / n)
    _map(tmp_path / "m1.csv", [{"original": "x1.jpg", "image_id": "A01_d0_top.jpg", "species": "apple",
                                "object_id": "A01", "quality_a": "Buena"},
                               {"original": "x2.jpg", "image_id": "A01_d0_side.jpg", "species": "apple",
                                "object_id": "A01"}])
    nuevas = pp.prepare_folder(src, tmp_path / "m1.csv", dst)
    ann = pd.read_csv(dst / "annotations.csv", dtype=str, keep_default_na=False)
    assert list(ann.columns) == pp.ANNOTATION_COLUMNS and len(nuevas) == len(ann) == 2
    assert ann.loc[0, "quality_a"] == "Buena" and ann.loc[1, "quality_a"] == ""      # lo que el mapa no trae queda vacío
    assert not any(pp.has_metadata(dst / i) for i in ann["image_id"])
    # segunda tanda: se agrega sin pisar lo anterior
    _map(tmp_path / "m2.csv", [{"original": "x3.jpg", "image_id": "G01_d0_top.jpg", "species": "guava"}])
    pp.prepare_folder(src, tmp_path / "m2.csv", dst)
    assert len(pd.read_csv(dst / "annotations.csv")) == 3
    # repetir un image_id ya registrado se rechaza
    with pytest.raises(ValueError, match="ya registrados"):
        pp.prepare_folder(src, tmp_path / "m2.csv", dst)


def test_map_validation(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _photo_with_metadata(src / "x1.jpg")
    _map(tmp_path / "sin_col.csv", [{"original": "x1.jpg"}])
    with pytest.raises(ValueError, match="Faltan columnas"):
        pp.load_photo_map(tmp_path / "sin_col.csv")
    _map(tmp_path / "dup.csv", [{"original": "a.jpg", "image_id": "B.jpg"}, {"original": "b.jpg", "image_id": "B.jpg"}])
    with pytest.raises(ValueError, match="repetidos"):
        pp.load_photo_map(tmp_path / "dup.csv")
    _map(tmp_path / "ext.csv", [{"original": "x1.jpg", "image_id": "B.png"}])
    with pytest.raises(ValueError, match="terminar en"):
        pp.load_photo_map(tmp_path / "ext.csv")
    _map(tmp_path / "falta.csv", [{"original": "no_esta.jpg", "image_id": "C.jpg"}])
    with pytest.raises(FileNotFoundError):
        pp.prepare_folder(src, tmp_path / "falta.csv", tmp_path / "dst")
