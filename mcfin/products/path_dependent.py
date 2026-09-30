"""Produits path-dépendants : asiatiques, barrières, lookbacks, cliquets,
swaps de variance.

Barrières et correction de pont brownien
----------------------------------------
Surveiller une barrière continue sur une grille discrète surestime la
survie (on « rate » les franchissements entre deux dates) : biais en
O(sqrt(Δt)). Conditionnellement aux extrémités, le log-spot entre deux dates
est un pont brownien de variance intégrée IV = ∫σ²dt et

    P(min_{[t_i, t_{i+1}]} X <= b | X_i = x_i, X_{i+1} = x_{i+1})
        = exp(-2 (x_i - b)(x_{i+1} - b) / IV)        (x = ln S, b = ln H)

On remplace l'indicatrice de survie par le produit des probabilités de
non-franchissement (estimateur conditionnel : sans biais pour GBM et de
variance plus faible). Pour les modèles à volatilité locale/stochastique,
IV est la variance intégrée simulée sur l'intervalle (approximation
standard).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analytics.black_scholes import option_sign
from ..core.results import Paths
from .base import Product, time_indices

__all__ = ["AsianOption", "BarrierOption", "Cliquet", "LookbackOption", "VarianceSwap"]


@dataclass
class AsianOption(Product):
    """Asiatique sur moyenne arithmétique (ou géométrique) de fixings.

    strike_type="fixed"   : (ω(A - K))^+
    strike_type="floating": (ω(S_T - A))^+
    """

    strike: float
    fixing_times: np.ndarray
    option_type: str = "call"
    average: str = "arithmetic"
    strike_type: str = "fixed"

    @property
    def observation_times(self):
        return np.asarray(self.fixing_times, dtype=float)

    @property
    def maturity(self) -> float:
        return float(np.max(self.fixing_times))

    def average_of(self, paths: Paths) -> np.ndarray:
        idx = time_indices(paths, self.fixing_times)
        s = paths.spot[:, idx]
        return np.exp(np.log(s).mean(axis=1)) if self.average == "geometric" else s.mean(axis=1)

    def payoff(self, paths: Paths) -> np.ndarray:
        w = option_sign(self.option_type)
        a = self.average_of(paths)
        x = a - self.strike if self.strike_type == "fixed" else paths.spot[:, -1] - a
        return np.maximum(w * x, 0.0) * paths.df(-1)


@dataclass
class BarrierOption(Product):
    """Option barrière knock-in/knock-out.

    monitoring="discrete"  : barrière observée uniquement aux dates données ;
    monitoring="continuous": correction de pont brownien entre les dates.
    """

    strike: float
    barrier: float
    maturity: float
    option_type: str = "call"
    barrier_type: str = "down-and-out"
    n_monitoring: int = 50
    monitoring: str = "continuous"
    rebate: float = 0.0

    @property
    def observation_times(self):
        return self.maturity * np.arange(1, self.n_monitoring + 1) / self.n_monitoring

    def survival(self, paths: Paths) -> np.ndarray:
        """Probabilité (conditionnelle) de ne pas avoir touché la barrière."""
        s = paths.spot
        down = self.barrier_type.startswith("down")
        x = np.log(s / self.barrier) * (1.0 if down else -1.0)  # distance > 0 côté vivant
        alive = np.all(x > 0, axis=1).astype(float)
        if self.monitoring == "discrete":
            return alive
        if paths.int_var is None:
            raise ValueError("surveillance continue : le modèle doit fournir la variance intégrée (int_var)")
        iv = np.maximum(paths.int_var, 1e-300)
        prod_x = np.maximum(x[:, :-1], 0.0) * np.maximum(x[:, 1:], 0.0)
        p_cross = np.exp(-2.0 * prod_x / iv)
        return alive * np.prod(1.0 - p_cross, axis=1)

    def payoff(self, paths: Paths) -> np.ndarray:
        w = option_sign(self.option_type)
        vanilla = np.maximum(w * (paths.spot[:, -1] - self.strike), 0.0)
        surv = self.survival(paths)
        knocked = 1.0 - surv
        if self.barrier_type.endswith("out"):
            val = vanilla * surv + self.rebate * knocked
        else:
            val = vanilla * knocked + self.rebate * surv
        return val * paths.df(-1)


@dataclass
class LookbackOption(Product):
    """Lookback à strike flottant (call : S_T - min S, put : max S - S_T),
    surveillance discrète. ``bgk_correction`` applique le décalage de
    Broadie-Glasserman-Kou pour approcher la surveillance continue :
    min_c ≈ min_d · e^{-β σ sqrt(Δt)}, max_c ≈ max_d · e^{+β σ sqrt(Δt)}."""

    maturity: float
    option_type: str = "call"
    n_monitoring: int = 252
    bgk_sigma: float | None = None

    @property
    def observation_times(self):
        return self.maturity * np.arange(1, self.n_monitoring + 1) / self.n_monitoring

    def payoff(self, paths: Paths) -> np.ndarray:
        from ..analytics.exotics import BGK_BETA

        s = paths.spot
        shift = 1.0
        if self.bgk_sigma is not None:
            shift = np.exp(BGK_BETA * self.bgk_sigma * np.sqrt(self.maturity / self.n_monitoring))
        if option_sign(self.option_type) > 0:
            val = s[:, -1] - s.min(axis=1) / shift
        else:
            val = s.max(axis=1) * shift - s[:, -1]
        return np.maximum(val, 0.0) * paths.df(-1)


@dataclass
class Cliquet(Product):
    """Cliquet « locally capped, globally floored » (Napoléon/ratchet...) :

        N · min(max(Σ_i min(max(S_{t_i}/S_{t_{i-1}} - 1, f_loc), c_loc), F_glob), C_glob)

    Très sensible au smile *forward* : LV et LSV calibrés aux mêmes vanilles
    donnent des prix différents (test de la dynamique du smile).
    """

    reset_times: np.ndarray
    local_floor: float = -0.05
    local_cap: float = 0.05
    global_floor: float = 0.0
    global_cap: float = np.inf
    notional: float = 1.0

    @property
    def observation_times(self):
        return np.asarray(self.reset_times, dtype=float)

    def payoff(self, paths: Paths) -> np.ndarray:
        idx = np.concatenate(([0], time_indices(paths, self.reset_times)))
        s = paths.spot[:, idx]
        ret = s[:, 1:] / s[:, :-1] - 1.0
        total = np.clip(ret, self.local_floor, self.local_cap).sum(axis=1)
        return self.notional * np.clip(total, self.global_floor, self.global_cap) * paths.df(-1)


@dataclass
class VarianceSwap(Product):
    """Swap de variance : N_var · (σ²_réalisée - K_var), fixings quotidiens.

    σ²_réalisée = (A/n) Σ ln²(S_i/S_{i-1}), A = 252 (annualisation).
    Avec strike nul, le prix actualisé / D(0,T) donne le strike équitable.
    """

    maturity: float
    strike_var: float = 0.0
    notional: float = 1.0
    n_fixings: int | None = None

    @property
    def observation_times(self):
        n = self.n_fixings or int(np.rint(252 * self.maturity))
        return self.maturity * np.arange(1, n + 1) / n

    def payoff(self, paths: Paths) -> np.ndarray:
        s = paths.spot
        lr = np.diff(np.log(s), axis=1)
        n = lr.shape[1]
        rv = (n / self.maturity) * np.mean(lr * lr, axis=1)
        return self.notional * (rv - self.strike_var) * paths.df(-1)
