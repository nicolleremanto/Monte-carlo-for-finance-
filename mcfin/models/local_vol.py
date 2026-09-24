"""Volatilité locale (Dupire 1994) et volatilité locale-stochastique (LSV).

Volatilité locale
-----------------
    dS/S = (r - q) dt + σ_loc(t, S) dW

σ_loc est déduite de la nappe implicite (formule de Dupire en variance
totale). Par construction le modèle reprice toutes les vanilles : c'est le
modèle standard des desks actions pour les exotiques « faiblement »
path-dépendants, mais il sous-estime la volatilité du smile forward (le smile
forward s'aplatit), ce qui biaise cliquets et options sur variance.

LSV (Heston local) et méthode particulaire
------------------------------------------
    dS/S = (r - q) dt + L(t, S) sqrt(v) dW_S
    dv   = κ(θ - v) dt + ξ sqrt(v) dW_v,   d<W_S, W_v> = ρ dt

Condition de calibration (Gyöngy 1986 / Dupire) : le modèle reprice les
vanilles ssi   L²(t, K) · E[v_t | S_t = K] = σ_loc²(t, K).
L'espérance conditionnelle dépend de la loi de (S_t, v_t), donc de L :
l'équation est de type McKean-Vlasov. Guyon & Henry-Labordère (2012,
« Being particular about calibration ») la résolvent par un système de N
particules en interaction, E[v | S = K] étant estimée par régression à noyau
de Nadaraya-Watson :

    E[v_t | S_t = K] ≈ Σ_i v_t^i δ_h(S_t^i - K) / Σ_i δ_h(S_t^i - K)

avec un noyau gaussien de fenêtre h = 1.5 σ_loc(t, S0) sqrt(max(t, 1/4)) N^{-1/5}
(en log-spot). Une seule simulation calibre toute la fonction de levier.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from ..core.results import Paths
from ..core.timegrid import TimeGrid
from ..market.curves import Curve
from ..market.volsurface import ImpliedVolSurface
from .base import Model

__all__ = ["LocalVol", "LeverageFunction", "LocalStochasticVol"]


@dataclass
class LocalVol(Model):
    """Volatilité locale ; schéma log-Euler (exactement martingale à chaque pas)."""
    spot: float = 100.0
    surface: ImpliedVolSurface | None = None
    rate: float | Curve = 0.0
    div: float | Curve = 0.0
    dt: float | None = 1.0 / 100
    t_min: float = 1e-3
    n_factors: int = field(default=1, init=False)

    def local_vol(self, t: float, s: np.ndarray) -> np.ndarray:
        y = np.log(s / self.forward(t))
        return np.sqrt(self.surface.local_variance(y, max(t, self.t_min)))

    def simulate(self, grid: TimeGrid, z: np.ndarray, rng=None) -> Paths:
        n = z.shape[0]
        dt = grid.dt
        drift = self.log_drift(grid.times)
        n_obs = grid.obs_idx.size
        spot = np.empty((n, n_obs + 1))
        iv = np.zeros((n, n_obs))
        spot[:, 0] = self.spot
        s = np.full(n, float(self.spot))
        logs = np.log(s)
        acc = np.zeros(n)
        obs = {k: j + 1 for j, k in enumerate(grid.obs_idx.tolist())}
        for i in range(grid.n_steps):
            sig = self.local_vol(grid.times[i], s)
            logs += drift[i] - 0.5 * sig**2 * dt[i] + sig * np.sqrt(dt[i]) * z[:, i, 0]
            s = np.exp(logs)
            acc += sig**2 * dt[i]
            if i + 1 in obs:
                j = obs[i + 1]
                spot[:, j], iv[:, j - 1] = s, acc
                acc = np.zeros(n)
        return Paths(times=np.concatenate(([0.0], grid.obs_times)), spot=spot,
                     discount=self._deterministic_discount(grid), int_var=iv)


@dataclass
class LeverageFunction:
    """L(t, y) tabulée : temps (n_t,), grilles de log-moneyness (n_t, M), valeurs (n_t, M).
    Interpolation linéaire en y (plate hors grille), constante par morceaux en t."""
    times: np.ndarray
    y_grid: np.ndarray
    values: np.ndarray

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        k = int(np.clip(np.searchsorted(self.times, t + 1e-12, side="right") - 1,
                        0, self.times.size - 1))
        return np.interp(y, self.y_grid[k], self.values[k])


@dataclass
class LocalStochasticVol(Model):
    """Heston local (LSV) calibré par méthode particulaire."""
    spot: float = 100.0
    surface: ImpliedVolSurface | None = None
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7
    rate: float | Curve = 0.0
    div: float | Curve = 0.0
    dt: float | None = 1.0 / 100
    leverage: LeverageFunction | None = None
    n_factors: int = field(default=2, init=False)

    # ------------------------------------------------------------------
    def _loc_var(self, t, y):
        return self.surface.local_variance(y, max(t, 1e-3))

    def calibrate(self, horizon: float, n_particles: int = 50_000, n_grid: int = 41,
                  bandwidth_scale: float = 1.5, seed: int | None = 0,
                  chunk: int = 20_000) -> "LocalStochasticVol":
        """Calibre la fonction de levier jusqu'à ``horizon`` ; renvoie un nouveau modèle."""
        rng = np.random.default_rng(seed)
        grid = TimeGrid.build([horizon], self.dt)
        times, dts = grid.times, grid.dt
        y = np.zeros(n_particles)           # ln(S/F_t) : martingale exponentielle
        v = np.full(n_particles, self.v0)
        L_times, L_grids, L_vals = [], [], []
        rc = np.sqrt(1 - self.rho**2)
        for i in range(grid.n_steps):
            t = times[i]
            if i == 0:
                yg = np.linspace(-0.5, 0.5, n_grid)
                lev = np.full(n_grid, np.sqrt(self._loc_var(t, np.array(0.0)) / self.v0))
            else:
                lo, hi = np.quantile(y, [0.001, 0.999])
                yg = np.linspace(lo, hi, n_grid)
                sig_atm = np.sqrt(self._loc_var(t, np.array(0.0)))
                h = bandwidth_scale * sig_atm * np.sqrt(max(t, 0.25)) * n_particles ** (-0.2)
                num = np.zeros(n_grid)
                den = np.zeros(n_grid)
                vp = np.maximum(v, 0.0)
                for c in range(0, n_particles, chunk):
                    ker = np.exp(-0.5 * ((y[c:c + chunk, None] - yg[None, :]) / h) ** 2)
                    num += vp[c:c + chunk] @ ker
                    den += ker.sum(axis=0)
                cond_v = np.maximum(num / np.maximum(den, 1e-300), 1e-6)
                lev = np.sqrt(self._loc_var(t, yg) / cond_v)
            L_times.append(t)
            L_grids.append(yg)
            L_vals.append(lev)
            # évolution des particules (Euler full truncation + log-Euler)
            zv = rng.standard_normal(n_particles)
            zs = self.rho * zv + rc * rng.standard_normal(n_particles)
            vp = np.maximum(v, 0.0)
            lp = np.interp(y, yg, lev)
            sq = np.sqrt(vp * dts[i])
            y += -0.5 * lp * lp * vp * dts[i] + lp * sq * zs
            v = v + self.kappa * (self.theta - vp) * dts[i] + self.xi * sq * zv
        lev_fn = LeverageFunction(np.asarray(L_times), np.asarray(L_grids), np.asarray(L_vals))
        return dataclasses.replace(self, leverage=lev_fn)

    def simulate(self, grid: TimeGrid, z: np.ndarray, rng=None) -> Paths:
        if self.leverage is None:
            raise RuntimeError("appeler calibrate() avant de simuler")
        n = z.shape[0]
        dt = grid.dt
        n_obs = grid.obs_idx.size
        spot = np.empty((n, n_obs + 1))
        var = np.empty((n, n_obs + 1))
        iv = np.zeros((n, n_obs))
        spot[:, 0], var[:, 0] = self.spot, self.v0
        y = np.zeros(n)
        v = np.full(n, self.v0)
        acc = np.zeros(n)
        rc = np.sqrt(1 - self.rho**2)
        fwd = self.forward(grid.times)
        obs = {k: j + 1 for j, k in enumerate(grid.obs_idx.tolist())}
        for i in range(grid.n_steps):
            vp = np.maximum(v, 0.0)
            lp = self.leverage(grid.times[i], y)
            sq = np.sqrt(vp * dt[i])
            zv = z[:, i, 0]
            y += -0.5 * lp * lp * vp * dt[i] + lp * sq * (self.rho * zv + rc * z[:, i, 1])
            v = v + self.kappa * (self.theta - vp) * dt[i] + self.xi * sq * zv
            acc += lp * lp * vp * dt[i]
            if i + 1 in obs:
                j = obs[i + 1]
                spot[:, j] = fwd[i + 1] * np.exp(y)
                var[:, j] = np.maximum(v, 0.0)
                iv[:, j - 1] = acc
                acc = np.zeros(n)
        return Paths(times=np.concatenate(([0.0], grid.obs_times)), spot=spot,
                     discount=self._deterministic_discount(grid), int_var=iv, variance=var)
