"""Modelos de ML tradicional: línea base, regresión logística, SVM y Random Forest.

Cada modelo es un ``Pipeline`` de scikit-learn con estandarización **dentro** del
pipeline (los estadísticos se calculan solo con los datos de entrenamiento de cada
pliegue, sin fuga) y se ajusta con ``GridSearchCV`` y ``StratifiedGroupKFold`` agrupado
por objeto físico. Ver ``docs/formulacion_matematica.md`` §4, §5 y §7.

Vistos en clase: regresión logística/softmax, SVM, Random Forest, ``GridSearchCV``,
``Pipeline``, F1 y kappa. **No visto en clase** (se declara en el informe):
``StratifiedGroupKFold`` y el histograma de color HSV como característica.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


#: Orden ordinal de las clases de calidad (para el kappa ponderado).
ORDEN_CALIDAD = ["Buena", "Regular", "Mala"]


def cap_per_cell(df: pd.DataFrame, cell_cols: Sequence[str], cap: int, seed: int = 42) -> pd.DataFrame:
    """Submuestrea cada celda (p. ej. especie x clase) a lo sumo ``cap`` filas.

    Evita que una celda enorme (la granada *Good*) domine el entrenamiento. Se aplica
    **solo a entrenamiento**: validación y prueba conservan su distribución natural.
    """
    parts = [g.sample(n=min(len(g), cap), random_state=seed) for _, g in df.groupby(list(cell_cols), sort=True)]
    return pd.concat(parts).sort_index()


def binomial_se(p: float, n: int) -> float:
    """Error estándar binomial de una proporción ``p`` con ``n`` casos: sqrt(p(1-p)/n)."""
    return float(np.sqrt(p * (1 - p) / n)) if n > 0 else float("nan")


def evaluate(y_true, y_pred, species=None) -> dict:
    """Métricas globales y, si se da ``species``, por especie.

    Returns:
        ``{'global': {...}, 'per_species': DataFrame}`` con exactitud, F1 macro, kappa
        de Cohen ponderado cuadráticamente (para 2 clases coincide con el kappa simple),
        error estándar binomial de la exactitud y número de imágenes.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    def one(yt, yp):
        """Métricas de un subconjunto (exactitud, EE, F1 macro y kappa ponderado)."""
        acc = accuracy_score(yt, yp)
        # con las clases de calidad, el orden ordinal es Buena < Regular < Mala (no el alfabético)
        presentes = set(yt) | set(yp)
        orden = [c for c in ORDEN_CALIDAD if c in presentes] if presentes <= set(ORDEN_CALIDAD) else None
        return {"n": len(yt), "exactitud": acc, "EE": binomial_se(acc, len(yt)),
                "F1_macro": f1_score(yt, yp, average="macro"),
                "kappa_w": cohen_kappa_score(yt, yp, weights="quadratic", labels=orden) if len(set(yt)) > 1 else float("nan")}

    out = {"global": one(y_true, y_pred)}
    if species is not None:
        species = np.asarray(species)
        out["per_species"] = pd.DataFrame(
            {s: one(y_true[species == s], y_pred[species == s]) for s in sorted(set(species))}).T
    return out


def make_cv(n_splits: int = 5, seed: int = 42) -> StratifiedGroupKFold:
    """Validación cruzada estratificada y agrupada por objeto (pliegues sin grupos compartidos)."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def model_zoo(seed: int = 42) -> dict:
    """Modelos y mallas de hiperparámetros: ``{nombre: (estimador, malla)}``.

    * ``baseline_mayoritaria``: predice siempre la clase más frecuente.
    * ``logreg``: regresión logística (softmax) con L2; ``C = 1/lambda``.
    * ``svm_rbf``: SVM con núcleo RBF; ``C`` y ``gamma``.
    * ``random_forest``: número de árboles, profundidad y hojas mínimas.

    Todos usan ``class_weight='balanced'`` (pesos ``w_k = N / (K * N_k)``) salvo la
    línea base mayoritaria.
    """
    return {
        "baseline_mayoritaria": (DummyClassifier(strategy="most_frequent"), {}),
        "logreg": (Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(class_weight="balanced", max_iter=2000,
                                                        random_state=seed))]),
                   {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
        "svm_rbf": (Pipeline([("scaler", StandardScaler()),
                              ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=seed))]),
                    {"clf__C": [1.0, 10.0, 100.0], "clf__gamma": [0.003, 0.01, 0.03]}),
        "random_forest": (RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1),
                          {"n_estimators": [200, 400], "max_depth": [None, 20], "min_samples_leaf": [1, 3]}),
    }
