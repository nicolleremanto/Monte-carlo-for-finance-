"""Interface commune des modèles de diffusion (actions / indices / change)."""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod

import numpy as np

from ..core.results import Paths
from ..core.timegrid import TimeGrid
from ..market.curves import Curve, as_curve

__all__ = ["Model", "bump_model"]


class Model(ABC):
    """Modèle simulable.

    Les sous-classes sont des dataclasses : un choc de paramètre (Greeks par
    différences finies) s'obtient par ``dataclasses.replace`` et conserve la
    même graine => nombres aléatoires communs (CRN).

    Attributs de classe
    -------------------
    n_factors : nombre de gaussiennes consommées par pas de temps.
    n_assets : nombre de sous-jacents.
    uniform_grid : le schéma exige un pas constant (rough Bergomi).
    """
    n_factors: int = 1
    n_assets: int = 1
    uniform_grid: bool = False
    dt: float | None = None  # pas maximal par défaut (None = schéma exact)

    # ---------------------------------------------------------------
    @property
    def rate_curve(self) -> Curve:
        return as_curve(getattr(self, "rate", 0.0))

    def div_curve(self, asset: int = 0) -> Curve:
        div = getattr(self, "div", 0.0)
        if isinstance(div, (list, tuple, np.ndarray)) and not isinstance(div, Curve):
            return as_curve(div[asset])
        return as_curve(div)

    def discount(self, t):
        return self.rate_curve.df(t)

    def forward(self, t, asset: int = 0):
        s0 = np.atleast_1d(np.asarray(self.spot, dtype=float))[asset]
        return s0 * self.div_curve(asset).df(t) / self.rate_curve.df(t)

    def log_drift(self, times: np.ndarray, asset: int = 0) -> np.ndarray:
        """ln(F(t_{i+1})/F(t_i)) : dérive risque-neutre exacte par pas."""
        f = np.log(self.div_curve(asset).df(times)) - np.log(self.rate_curve.df(times))
        return np.diff(f)

    def build_grid(self, obs_times, max_dt: float | None = None) -> TimeGrid:
        dt = self.dt if max_dt is None else max_dt
        if self.uniform_grid:
            return TimeGrid.uniform(obs_times, dt)
        return TimeGrid.build(obs_times, dt)

    @abstractmethod
    def simulate(self, grid: TimeGrid, z: np.ndarray, rng=None) -> Paths:
        """Simule les trajectoires. ``z`` : (n_paths, n_steps, n_factors)."""

    def _deterministic_discount(self, grid: TimeGrid) -> np.ndarray:
        return self.rate_curve.df(np.concatenate(([0.0], grid.obs_times)))


def bump_model(model: Model, param: str, h: float, relative: bool = False) -> Model:
    """Renvoie une copie du modèle avec ``param`` choqué de ``h``.

    Gère les flottants, tableaux (choc de tous les éléments) et courbes
    (choc parallèle des taux zéro).
    """
    value = getattr(model, param)
    if isinstance(value, Curve):
        new = value.shift(h)
    else:
        arr = np.asarray(value, dtype=float)
        new = arr * (1 + h) if relative else arr + h
        if np.ndim(value) == 0:
            new = float(new)
    return dataclasses.replace(model, **{param: new})
