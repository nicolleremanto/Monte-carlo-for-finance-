"""Interface des produits et utilitaires communs."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.results import Paths

__all__ = ["Product", "time_index"]


def time_index(paths: Paths, t) -> np.ndarray | int:
    """Indice(s) de la (des) date(s) ``t`` dans ``paths.times`` (tolérance 1e-9)."""
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    idx = np.searchsorted(paths.times, t_arr - 1e-9)
    idx = np.clip(idx, 0, paths.times.size - 1)
    if np.any(np.abs(paths.times[idx] - t_arr) > 1e-9):
        raise ValueError(f"date(s) {t_arr} absente(s) de la grille d'observation")
    return idx if np.ndim(t) else int(idx[0])


class Product(ABC):
    """Produit dérivé évalué par Monte Carlo.

    ``observation_times`` : dates dont le produit a besoin (> 0).
    ``payoff(paths)`` : flux actualisés (valeur en 0) par trajectoire.
    """

    @property
    @abstractmethod
    def observation_times(self) -> np.ndarray:
        ...

    @abstractmethod
    def payoff(self, paths: Paths) -> np.ndarray:
        ...
