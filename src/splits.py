"""Particiones train/val/test **por grupo** (banana física) para evitar fuga de datos.

Imágenes de una misma banana (distintos ángulos, mismo instante) están fuertemente
correlacionadas; si se reparten al azar entre train y test, las métricas salen
infladas. Aquí toda la banana va a una sola partición. Ver
``docs/formulacion_matematica.md`` §7.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assert_no_group_overlap(a: pd.DataFrame, b: pd.DataFrame, group_col: str = "group") -> None:
    """Lanza ``AssertionError`` si algún grupo aparece en ambos DataFrames."""
    shared = set(a[group_col]) & set(b[group_col])
    assert not shared, f"Fuga de datos: {len(shared)} grupos en ambas particiones, p. ej. {sorted(shared)[:3]}"


def grouped_split(df: pd.DataFrame, label_col: str = "label", group_col: str = "group",
                  test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide ``df`` en (train, test) sin compartir grupos y estratificando por clase.

    Usa ``StratifiedGroupKFold`` con ``round(1/test_size)`` pliegues y toma el primero
    como test, por lo que la proporción de test es aproximada (los grupos no se parten).

    Raises:
        AssertionError: si por algún motivo un grupo queda en ambas particiones.
    """
    n_splits = max(2, int(round(1 / test_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df[label_col], df[group_col]))
    train, test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    assert_no_group_overlap(train, test, group_col)
    return train, test


def grouped_train_val_test(df: pd.DataFrame, label_col: str = "label", group_col: str = "group",
                           val_size: float = 0.15, test_size: float = 0.15,
                           seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partición (train, val, test) por grupo; ``val_size`` es la fracción sobre el total."""
    train_val, test = grouped_split(df, label_col, group_col, test_size, seed)
    rel_val = val_size / (1 - test_size)
    train, val = grouped_split(train_val, label_col, group_col, rel_val, seed + 1)
    assert_no_group_overlap(train, val, group_col)
    assert_no_group_overlap(val, test, group_col)
    return train, val, test
