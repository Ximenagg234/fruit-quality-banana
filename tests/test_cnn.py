"""Verifica el conteo analítico de parámetros contra Keras (formulación §6)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cnn  # noqa: E402


def test_layer_formulas_match_hand_calculation():
    assert cnn.conv_params(3, 3, 32) == 896
    assert cnn.conv_params(3, 32, 64) == 18_496
    assert cnn.conv_params(3, 64, 128) == 73_856
    assert cnn.dense_params(32_768, 128) == 4_194_432
    assert cnn.dense_params(128, 3) == 387


def test_analytic_count_example_of_the_formulation():
    assert cnn.analytic_param_count(128, 3, "flatten") == 4_288_067
    assert cnn.analytic_param_count(128, 3, "gap") == 110_147
    with pytest.raises(ValueError):
        cnn.analytic_param_count(head="otro")


@pytest.mark.parametrize("size,n_classes,head", [(128, 3, "flatten"), (128, 3, "gap"),
                                                 (96, 6, "flatten"), (96, 6, "gap")])
def test_analytic_count_matches_keras(size, n_classes, head):
    pytest.importorskip("tensorflow")
    model = cnn.build_cnn(size, n_classes, head)
    assert model.count_params() == cnn.analytic_param_count(size, n_classes, head)
    assert model.output_shape == (None, n_classes)


@pytest.mark.parametrize("head", ["flatten", "gap"])
def test_augmentation_and_l2_do_not_change_the_parameter_count(head):
    pytest.importorskip("tensorflow")
    base = cnn.build_cnn(96, 2, head)
    extra = cnn.build_cnn(96, 2, head, l2=1e-4, augment=True)
    assert extra.count_params() == base.count_params() == cnn.analytic_param_count(96, 2, head)
    nombres = [type(l).__name__ for l in extra.layers]
    assert {"RandomFlip", "RandomRotation", "RandomZoom", "RandomContrast"} <= set(nombres)
    assert not any("Random" in type(l).__name__ for l in base.layers)
    assert len(extra.losses) > 0 and len(base.losses) == 0          # la regularización L2 agrega términos a la pérdida


def test_augmentation_is_inactive_at_inference():
    pytest.importorskip("tensorflow")
    import numpy as np
    m = cnn.build_cnn(64, 3, "gap", augment=True)
    x = np.random.default_rng(0).random((4, 64, 64, 3), dtype=np.float32)
    assert np.allclose(m.predict(x, verbose=0), m.predict(x, verbose=0))      # determinista al predecir
