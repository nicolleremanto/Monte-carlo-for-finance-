"""Modèles actions : Black-Scholes multi-actifs, Merton, Heston/Bates, SABR.

Schémas de discrétisation
-------------------------
* Black-Scholes : solution exacte du log-prix (aucun biais de discrétisation).
* Merton : exacte aux dates de la grille (Poisson par inversion, somme de
  N sauts gaussiens = N μ + sqrt(N) δ Z).
* Heston :
    - "qe" : Quadratic-Exponential d'Andersen (2008) avec correction de
      martingale (QE-M). Appariement des deux premiers moments de la loi
      conditionnelle (χ² non centrée) de v_{t+Δ} ; schéma de référence en
      production (biais très faible avec des pas hebdomadaires).
    - "euler" : Euler « full truncation » (Lord, Koekkoek & van Dijk 2010),
      le meilleur des schémas d'Euler biaisés pour la CIR.
* SABR : α lognormal exact, F par Euler absorbé en 0 (β < 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

from ..core.results import Paths
from ..core.rng import poisson_inverse
from ..core.timegrid import TimeGrid
from ..market.curves import Curve
from .base import Model

__all__ = ["SABR", "BlackScholes", "Heston", "MertonJumpDiffusion"]


def _obs_mask(grid: TimeGrid) -> np.ndarray:
    mask = np.zeros(grid.n_steps + 1, dtype=bool)
    mask[grid.obs_idx] = True
    return mask


@dataclass
class BlackScholes(Model):
    """Black-Scholes multi-actifs corrélés (Cholesky).

    dS_i/S_i = (r - q_i) dt + σ_i dW_i,  d<W_i, W_j> = ρ_ij dt
    """

    spot: float | np.ndarray = 100.0
    vol: float | np.ndarray = 0.2
    rate: float | Curve = 0.0
    div: float | Curve | np.ndarray = 0.0
    corr: np.ndarray | None = None
    dt: float | None = None

    def __post_init__(self):
        self.n_assets = int(np.size(self.spot))
        self.n_factors = self.n_assets
        if self.n_assets > 1:
            corr = np.eye(self.n_assets) if self.corr is None else np.asarray(self.corr, float)
            self._chol = np.linalg.cholesky(corr)

    def simulate(self, grid: TimeGrid, z: np.ndarray, rng=None) -> Paths:
        dt = grid.dt
        s0 = np.atleast_1d(np.asarray(self.spot, float))
        vol = np.broadcast_to(np.asarray(self.vol, float), s0.shape)
        n = z.shape[0]
        d = self.n_assets
        # corrélation (Cholesky) puis incréments du log-spot, calcul en place
        if d == 1:
            incr = z.copy()
        elif d <= 8:  # combinaisons explicites : bien plus rapide qu'un matmul batché
            incr = np.empty_like(z)
            for i in range(d):
                incr[..., i] = self._chol[i, 0] * z[..., 0]
                for j in range(1, i + 1):
                    incr[..., i] += self._chol[i, j] * z[..., j]
        else:
            incr = (z.reshape(-1, d) @ self._chol.T).reshape(z.shape)
        incr *= (vol[None, :] * np.sqrt(dt)[:, None])[None]
        drift = np.stack([self.log_drift(grid.times, a) for a in range(d)], axis=-1)
        incr += (drift - 0.5 * vol**2 * dt[:, None])[None]
        np.cumsum(incr, axis=1, out=incr)
        spot = np.empty((n, grid.obs_idx.size + 1, d))
        spot[:, 0] = s0
        if grid.obs_idx.size == grid.n_steps:  # toutes les dates observées : pas de copie
            logs = incr
        else:
            logs = np.take(incr, grid.obs_idx - 1, axis=1)
        logs += np.log(s0)
        np.exp(logs, out=spot[:, 1:])
        t_all = np.concatenate(([0.0], grid.obs_times))
        # variance intégrée déterministe : vue diffusée (aucune copie)
        iv = np.broadcast_to(
            (vol**2)[None, None, :] * np.diff(t_all)[None, :, None], (n, t_all.size - 1, self.n_assets)
        )
        if self.n_assets == 1:
            spot, iv = spot[..., 0], iv[..., 0]
        return Paths(times=t_all, spot=spot, discount=self._deterministic_discount(grid), int_var=iv)


@dataclass
class MertonJumpDiffusion(Model):
    """dS/S = (r - q - λk̄) dt + σ dW + (J - 1) dN,  ln J ~ N(μ_J, δ²)."""

    spot: float = 100.0
    vol: float = 0.2
    lam: float = 0.5
    mu_j: float = -0.1
    sigma_j: float = 0.15
    rate: float | Curve = 0.0
    div: float | Curve = 0.0
    dt: float | None = None
    n_factors: int = field(default=3, init=False)

    def simulate(self, grid: TimeGrid, z: np.ndarray, rng=None) -> Paths:
        dt = grid.dt
        kbar = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0
        n_jumps = poisson_inverse(ndtr(z[:, :, 1]), self.lam * dt[None, :])
        jumps = n_jumps * self.mu_j + np.sqrt(n_jumps) * self.sigma_j * z[:, :, 2]
        incr = (
            (self.log_drift(grid.times) - (0.5 * self.vol**2 + self.lam * kbar) * dt)[None]
            + self.vol * np.sqrt(dt)[None] * z[:, :, 0]
            + jumps
        )
        logs = np.log(self.spot) + np.concatenate(
            [np.zeros((z.shape[0], 1)), np.cumsum(incr, axis=1)], axis=1
        )
        idx = np.concatenate(([0], grid.obs_idx))
        t_all = np.concatenate(([0.0], grid.obs_times))
        iv = np.broadcast_to(self.vol**2 * np.diff(t_all), (z.shape[0], t_all.size - 1))
        return Paths(
            times=t_all,
            spot=np.exp(logs[:, idx]),
            discount=self._deterministic_discount(grid),
            int_var=np.array(iv),
        )


@dataclass
class Heston(Model):
    """Heston (1993), avec sauts de Merton optionnels (Bates 1996).

    dS/S = (r - q - λk̄) dt + sqrt(v) dW_S + (J-1) dN
    dv   = κ(θ - v) dt + ξ sqrt(v) dW_v,   d<W_S, W_v> = ρ dt
    """

    spot: float = 100.0
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7
    rate: float | Curve = 0.0
    div: float | Curve = 0.0
    lam: float = 0.0
    mu_j: float = 0.0
    sigma_j: float = 0.0
    scheme: str = "qe"
    dt: float | None = 1.0 / 64
    martingale_correction: bool = True
    psi_c: float = 1.5

    def __post_init__(self):
        if self.scheme not in ("qe", "euler"):
            raise ValueError("scheme ∈ {'qe', 'euler'}")
        self.n_factors = 4 if self.lam > 0 else 2

    @property
    def params(self):
        from ..analytics.heston import HestonParams

        return HestonParams(
            self.v0, self.kappa, self.theta, self.xi, self.rho, self.lam, self.mu_j, self.sigma_j
        )

    # --- schéma QE ------------------------------------------------------
    def _qe_step(self, v, dt, zv, zs):
        """Un pas QE-M : renvoie (v_next, incrément de ln S hors dérive)."""
        k, th, xi, rho = self.kappa, self.theta, self.xi, self.rho
        e = np.exp(-k * dt)
        m = th + (v - th) * e
        s2 = v * xi**2 * e * (1 - e) / k + th * xi**2 * (1 - e) ** 2 / (2 * k)
        psi = s2 / (m * m)
        quad = psi <= self.psi_c
        # branche quadratique : v' = a (b + Z)²
        psi_q = np.where(quad, psi, 1.0)
        b2 = 2 / psi_q - 1 + np.sqrt(2 / psi_q) * np.sqrt(2 / psi_q - 1)
        a = m / (1 + b2)
        v_quad = a * (np.sqrt(b2) + zv) ** 2
        # branche exponentielle : masse p en 0 + queue exponentielle
        psi_e = np.where(quad, 2.0, psi)
        p = (psi_e - 1) / (psi_e + 1)
        beta = (1 - p) / m
        u = ndtr(zv)
        with np.errstate(divide="ignore"):
            v_exp = np.where(u <= p, 0.0, np.log((1 - p) / np.maximum(1 - u, 1e-300)) / beta)
        v_next = np.where(quad, v_quad, v_exp)

        g1 = g2 = 0.5
        K0 = -rho * k * th * dt / xi
        K1 = g1 * dt * (k * rho / xi - 0.5) - rho / xi
        K2 = g2 * dt * (k * rho / xi - 0.5) + rho / xi
        K3 = g1 * dt * (1 - rho**2)
        K4 = g2 * dt * (1 - rho**2)
        if self.martingale_correction:
            A = K2 + 0.5 * K4
            with np.errstate(invalid="ignore", divide="ignore"):
                k0_q = -A * b2 * a / (1 - 2 * A * a) + 0.5 * np.log(1 - 2 * A * a)
                k0_e = -np.log(p + beta * (1 - p) / (beta - A))
            ok = np.where(quad, A < 1 / (2 * a), A < beta)
            k0 = np.where(quad, k0_q, k0_e) - (K1 + 0.5 * K3) * v
            K0 = np.where(ok & np.isfinite(k0), k0, K0)
        dlog = K0 + K1 * v + K2 * v_next + np.sqrt(np.maximum(K3 * v + K4 * v_next, 0.0)) * zs
        return v_next, dlog

    def _euler_step(self, v, dt, zv, zs):
        vp = np.maximum(v, 0.0)
        sq = np.sqrt(vp * dt)
        dlog = -0.5 * vp * dt + sq * (self.rho * zv + np.sqrt(1 - self.rho**2) * zs)
        v_next = v + self.kappa * (self.theta - vp) * dt + self.xi * sq * zv
        return v_next, dlog

    def simulate(self, grid: TimeGrid, z: np.ndarray, rng=None) -> Paths:
        n = z.shape[0]
        dt = grid.dt
        drift = self.log_drift(grid.times)
        kbar = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0
        n_obs = grid.obs_idx.size
        spot = np.empty((n, n_obs + 1))
        var = np.empty((n, n_obs + 1))
        iv = np.zeros((n, n_obs))
        logs = np.full(n, np.log(self.spot))
        v = np.full(n, self.v0)
        spot[:, 0], var[:, 0] = self.spot, self.v0
        step_fn = self._qe_step if self.scheme == "qe" else self._euler_step
        acc = np.zeros(n)
        j = 0
        obs = set(grid.obs_idx.tolist())
        for i in range(grid.n_steps):
            v_next, dlog = step_fn(v, dt[i], z[:, i, 0], z[:, i, 1])
            if self.lam > 0:
                nj = poisson_inverse(ndtr(z[:, i, 2]), self.lam * dt[i])
                dlog = (
                    dlog + nj * self.mu_j + np.sqrt(nj) * self.sigma_j * z[:, i, 3] - self.lam * kbar * dt[i]
                )
            logs += drift[i] + dlog
            if self.scheme == "qe":
                acc += 0.5 * (v + v_next) * dt[i]
            else:
                acc += np.maximum(v, 0.0) * dt[i]
            v = v_next
            if i + 1 in obs:
                j += 1
                spot[:, j] = np.exp(logs)
                var[:, j] = np.maximum(v, 0.0)
                iv[:, j - 1] = acc
                acc = np.zeros(n)
        return Paths(
            times=np.concatenate(([0.0], grid.obs_times)),
            spot=spot,
            discount=self._deterministic_discount(grid),
            int_var=iv,
            variance=var,
        )


@dataclass
class SABR(Model):
    """SABR (Hagan et al. 2002) sous la mesure forward T.

    ``spot`` est le forward F0 ; ``rate`` sert uniquement à l'actualisation.
    """

    spot: float = 0.03
    alpha: float = 0.03
    beta: float = 0.5
    rho: float = -0.3
    nu: float = 0.4
    rate: float | Curve = 0.0
    dt: float | None = 1.0 / 250
    n_factors: int = field(default=2, init=False)

    def forward(self, t, asset: int = 0):
        return np.full_like(np.asarray(t, dtype=float), self.spot)

    def simulate(self, grid: TimeGrid, z: np.ndarray, rng=None) -> Paths:
        n = z.shape[0]
        dt = grid.dt
        f = np.full(n, float(self.spot))
        a = np.full(n, float(self.alpha))
        n_obs = grid.obs_idx.size
        out = np.empty((n, n_obs + 1))
        iv = np.zeros((n, n_obs))
        out[:, 0] = f
        acc = np.zeros(n)
        j = 0
        obs = set(grid.obs_idx.tolist())
        rc = np.sqrt(1 - self.rho**2)
        for i in range(grid.n_steps):
            z2 = z[:, i, 1]
            z1 = self.rho * z2 + rc * z[:, i, 0]
            sq = np.sqrt(dt[i])
            if self.beta == 1.0:
                f = f * np.exp(a * sq * z1 - 0.5 * a * a * dt[i])
                loc_var = a * a
            else:
                fb = np.power(np.maximum(f, 0.0), self.beta)
                # variance lognormale locale en début de pas : (α F^{β-1})² (0 si absorbé)
                loc_var = np.where(f > 0, (a * fb / np.maximum(f, 1e-300)) ** 2, 0.0)
                f = np.maximum(f + a * fb * sq * z1, 0.0)  # absorption en 0
            acc += np.minimum(loc_var, 1e6) * dt[i]
            a = a * np.exp(self.nu * sq * z2 - 0.5 * self.nu**2 * dt[i])
            if i + 1 in obs:
                j += 1
                out[:, j] = f
                iv[:, j - 1] = acc
                acc = np.zeros(n)
        return Paths(
            times=np.concatenate(([0.0], grid.obs_times)),
            spot=out,
            discount=self._deterministic_discount(grid),
            int_var=iv,
        )
