"""Grilles de temps de simulation.

Un produit déclare ses dates d'observation (fixings, dates d'exercice,
dates de constatation d'un autocall...). Le modèle impose éventuellement un
pas maximal (schémas d'Euler, volatilité locale...). La grille de simulation
est la réunion des dates d'observation et d'une subdivision régulière de
chaque intervalle ; on ne conserve en mémoire que les valeurs aux dates
d'observation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["TimeGrid"]

_TOL = 1e-10


@dataclass(frozen=True)
class TimeGrid:
    times: np.ndarray  # grille complète, times[0] = 0
    obs_idx: np.ndarray  # indices des dates d'observation dans `times`

    @property
    def dt(self) -> np.ndarray:
        return np.diff(self.times)

    @property
    def n_steps(self) -> int:
        return self.times.size - 1

    @property
    def obs_times(self) -> np.ndarray:
        return self.times[self.obs_idx]

    @staticmethod
    def _clean(obs_times) -> np.ndarray:
        t = np.unique(np.round(np.asarray(obs_times, dtype=float).ravel(), 12))
        if t.size == 0 or t[0] <= 0:
            raise ValueError("les dates d'observation doivent être > 0")
        return t

    @classmethod
    def build(cls, obs_times, max_dt: float | None = None) -> TimeGrid:
        """Réunion des dates d'observation et d'un raffinement de pas <= max_dt."""
        t_obs = cls._clean(obs_times)
        pts = [0.0]
        obs_idx = []
        prev = 0.0
        for t in t_obs:
            n_sub = 1 if max_dt is None else max(1, int(np.ceil((t - prev) / max_dt - _TOL)))
            pts.extend(prev + (t - prev) * np.arange(1, n_sub + 1) / n_sub)
            obs_idx.append(len(pts) - 1)
            prev = t
        times = np.asarray(pts)
        times[np.asarray(obs_idx)] = t_obs  # évite les erreurs d'arrondi
        return cls(times=times, obs_idx=np.asarray(obs_idx, dtype=int))

    @classmethod
    def uniform(cls, obs_times, max_dt: float) -> TimeGrid:
        """Grille uniforme (requise par le schéma hybride du rough Bergomi).

        Cherche le plus grand pas Δ <= max_dt tel que toutes les dates
        d'observation soient des multiples de Δ ; sinon, projette les dates
        sur la grille la plus proche (avec avertissement).
        """
        t_obs = cls._clean(obs_times)
        horizon = t_obs[-1]
        n0 = max(1, int(np.ceil(horizon / max_dt - _TOL)))
        for n in range(n0, n0 + 2000):
            k = t_obs / horizon * n
            if np.all(np.abs(k - np.round(k)) < 1e-8):
                break
        else:
            n = n0
            warnings.warn("dates d'observation projetées sur une grille uniforme", stacklevel=2)
        times = horizon * np.arange(n + 1) / n
        obs_idx = np.round(t_obs / horizon * n).astype(int)
        if np.any(obs_idx == 0):
            raise ValueError("pas trop grossier pour la première date d'observation")
        return cls(times=times, obs_idx=obs_idx)
