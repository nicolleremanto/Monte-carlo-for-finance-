"""Interface des produits et utilitaires communs."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.results import Paths

__all__ = ["Product", "time_index", "time_indices"]


def time_indices(paths: Paths, times) -> np.ndarray:
    """Indices des dates ``times`` dans ``paths.times`` (tolérance 1e-9)."""
    t_arr = np.atleast_1d(np.asarray(times, dtype=float))
    idx = np.clip(np.searchsorted(paths.times, t_arr - 1e-9), 0, paths.times.size - 1)
    if np.any(np.abs(paths.times[idx] - t_arr) > 1e-9):
        raise ValueError(f"date(s) {t_arr} absente(s) de la grille d'observation")
    return idx


def time_index(paths: Paths, t: float) -> int:
    """Indice de la date ``t`` dans ``paths.times``."""
    return int(time_indices(paths, t)[0])


class Product(ABC):
    """Produit dérivé évalué par Monte Carlo.

    ``observation_times`` : dates dont le produit a besoin (> 0).
    ``payoff(paths)`` : flux actualisés (valeur en 0) par trajectoire.
    """

    @property
    @abstractmethod
    def observation_times(self) -> np.ndarray: ...

    @abstractmethod
    def payoff(self, paths: Paths) -> np.ndarray: ...
