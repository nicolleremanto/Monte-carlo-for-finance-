"""Échantillonnage préférentiel par changement de dérive gaussien.

Pour Z ~ N(0, I) et un payoff G(Z) :  E[G(Z)] = E_μ[ G(Z) e^{-μ·Z + |μ|²/2} ],
Z ~ N(μ, I) sous la nouvelle mesure (Girsanov discret). Choix de μ de
Glasserman, Heidelberger & Shahabuddin (1999) : le mode de G(z)φ(z),

    μ* = argmax_z [ ln G(z) - |z|²/2 ],

approximation (asymptotiquement optimale) de la densité d'échantillonnage
à variance nulle ∝ G(z)φ(z). Gain spectaculaire pour les événements rares :
options très en dehors de la monnaie, digitales lointaines, queue de
distribution pour la VaR.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..models.base import Model

__all__ = ["optimal_drift"]


def optimal_drift(model: Model, product, max_dt: float | None = None, eps: float = 1e-300) -> np.ndarray:
    """Dérive μ* (n_steps, n_factors) maximisant ln G(z) - |z|²/2.

    G(z) est le flux actualisé d'une trajectoire unique simulée à partir de z
    (même grille que le moteur). Initialisation par recherche sur la droite
    z = c·1 (trajectoire « montante » ou « descendante ») pour partir d'un
    point où G > 0.
    """
    grid = model.build_grid(product.observation_times, max_dt)
    shape = (grid.n_steps, model.n_factors)
    dim = shape[0] * shape[1]

    def g(zflat):
        paths = model.simulate(grid, zflat.reshape((1, *shape)))
        return float(product.payoff(paths)[0])

    def objective(zflat):
        return -(np.log(max(g(zflat), eps)) - 0.5 * zflat @ zflat)

    best, best_val = np.zeros(dim), np.inf
    scale = np.sqrt(grid.dt / grid.times[-1])  # direction « brownien terminal »
    direction = np.repeat(scale, shape[1])
    for c in np.linspace(-6, 6, 49):
        x0 = c * direction
        val = objective(x0)
        if val < best_val:
            best, best_val = x0, val
    res = minimize(objective, best, method="L-BFGS-B")
    return res.x.reshape(shape)
