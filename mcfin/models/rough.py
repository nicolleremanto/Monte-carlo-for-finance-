"""Rough Bergomi (Bayer, Friz & Gatheral 2016) — schéma hybride.

    v_t = ξ_0(t) exp(η Y_t - η²/2 t^{2H}),   Y_t = sqrt(2H) ∫_0^t (t-s)^{H-1/2} dW_s
    dS_t/S_t = (r - q) dt + sqrt(v_t) (ρ dW_t + sqrt(1-ρ²) dW⊥_t)

Y est un processus de Volterra (Riemann-Liouville), non markovien, de
rugosité H ≈ 0.1 (Gatheral, Jaisson & Rosenbaum 2018, « Volatility is
rough »). Il reproduit l'explosion du skew ATM en T^{H-1/2} aux maturités
courtes, que les modèles markoviens (Heston) ne capturent pas.

Schéma hybride (Bennedsen, Lunde & Pakkanen 2017), κ = 1, α = H - 1/2 :
    Y(t_i) ≈ sqrt(2α+1) [ W̃_i + Σ_{k=2}^{i} (b_k Δ)^α ΔW_{i-k+1} ]
où W̃_i = ∫_{t_{i-1}}^{t_i} (t_i - s)^α dW_s est simulé exactement avec ΔW_i
(vecteur gaussien 2D), b_k = ((k^{α+1} - (k-1)^{α+1})/(α+1))^{1/α} (points
d'évaluation optimaux) ; la somme de Riemann est une convolution calculée
par FFT : coût O(N log N) par trajectoire.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.signal import fftconvolve

from ..core.results import Paths
from ..core.timegrid import TimeGrid
from ..market.curves import Curve
from .base import Model

__all__ = ["RoughBergomi"]


@dataclass
class RoughBergomi(Model):
    spot: float = 100.0
    xi0: float | Callable = 0.04
    eta: float = 1.9
    hurst: float = 0.1
    rho: float = -0.9
    rate: float | Curve = 0.0
    div: float | Curve = 0.0
    dt: float | None = 1.0 / 252
    n_factors: int = field(default=3, init=False)
    uniform_grid: bool = field(default=True, init=False)

    def _xi(self, t):
        return self.xi0(t) if callable(self.xi0) else np.full_like(t, float(self.xi0))

    def volterra(self, dt: float, n_steps: int, z: np.ndarray):
        """Renvoie (Y aux points de grille (n, N+1), ΔW (n, N))."""
        a = self.hurst - 0.5
        cov = np.array([[dt, dt ** (a + 1) / (a + 1)],
                        [dt ** (a + 1) / (a + 1), dt ** (2 * a + 1) / (2 * a + 1)]])
        chol = np.linalg.cholesky(cov)
        dw = chol[0, 0] * z[:, :, 0]
        w_tilde = chol[1, 0] * z[:, :, 0] + chol[1, 1] * z[:, :, 1]
        k = np.arange(n_steps + 1, dtype=float)
        gamma = np.zeros(n_steps + 1)
        if abs(a) < 1e-12:
            gamma[2:] = 1.0
        else:
            b = ((k[2:] ** (a + 1) - (k[2:] - 1) ** (a + 1)) / (a + 1)) ** (1 / a)
            gamma[2:] = (b * dt) ** a
        conv = fftconvolve(dw, gamma[None, :], axes=1)[:, : n_steps + 1]
        y1 = np.zeros((z.shape[0], n_steps + 1))
        y1[:, 1:] = w_tilde
        return np.sqrt(2 * a + 1) * (y1 + conv), dw

    def simulate(self, grid: TimeGrid, z: np.ndarray, rng=None) -> Paths:
        dt = float(grid.dt[0])
        if not np.allclose(grid.dt, dt):
            raise ValueError("le rough Bergomi exige une grille uniforme")
        n, N = z.shape[0], grid.n_steps
        t = grid.times
        Y, dw = self.volterra(dt, N, z)
        v = self._xi(t)[None, :] * np.exp(self.eta * Y - 0.5 * self.eta**2
                                          * t[None, :] ** (2 * self.hurst))
        db = self.rho * dw + np.sqrt(1 - self.rho**2) * np.sqrt(dt) * z[:, :, 2]
        incr = self.log_drift(t)[None, :] - 0.5 * v[:, :-1] * dt + np.sqrt(v[:, :-1]) * db
        logs = np.log(self.spot) + np.concatenate([np.zeros((n, 1)), np.cumsum(incr, axis=1)],
                                                  axis=1)
        idx = np.concatenate(([0], grid.obs_idx))
        cumv = np.concatenate([np.zeros((n, 1)), np.cumsum(v[:, :-1] * dt, axis=1)], axis=1)
        return Paths(times=t[idx], spot=np.exp(logs[:, idx]),
                     discount=self._deterministic_discount(grid),
                     int_var=np.diff(cumv[:, idx], axis=1), variance=v[:, idx],
                     extra={"Y": Y[:, idx]})
