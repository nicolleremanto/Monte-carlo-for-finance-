"""Options européennes mono et multi sous-jacents."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analytics.black_scholes import option_sign
from ..core.results import Paths
from .base import Product

__all__ = ["EuropeanOption", "DigitalOption", "BasketOption", "RainbowOption"]


@dataclass
class EuropeanOption(Product):
    strike: float
    maturity: float
    option_type: str = "call"

    @property
    def observation_times(self):
        return np.array([self.maturity])

    def payoff(self, paths: Paths) -> np.ndarray:
        w = option_sign(self.option_type)
        return np.maximum(w * (paths.spot[:, -1] - self.strike), 0.0) * paths.df(-1)


@dataclass
class DigitalOption(Product):
    """Digitale cash-or-nothing (payoff discontinu : cas d'école des Greeks
    pathwise en échec, cf. mcfin.greeks)."""
    strike: float
    maturity: float
    option_type: str = "call"
    cash: float = 1.0

    @property
    def observation_times(self):
        return np.array([self.maturity])

    def payoff(self, paths: Paths) -> np.ndarray:
        w = option_sign(self.option_type)
        return self.cash * (w * (paths.spot[:, -1] - self.strike) > 0) * paths.df(-1)


@dataclass
class BasketOption(Product):
    """Option sur panier Σ w_i S_i(T)/S_i(0) (performances pondérées)."""
    strike: float
    maturity: float
    weights: np.ndarray
    option_type: str = "call"
    initial_levels: np.ndarray | None = None  # niveaux de référence figés (sinon spot en 0)

    @property
    def observation_times(self):
        return np.array([self.maturity])

    def payoff(self, paths: Paths) -> np.ndarray:
        ref = paths.spot[:, 0, :] if self.initial_levels is None else np.asarray(self.initial_levels)
        perf = paths.spot[:, -1, :] / ref
        basket = perf @ np.asarray(self.weights, dtype=float)
        w = option_sign(self.option_type)
        return np.maximum(w * (basket - self.strike), 0.0) * paths.df(-1)


@dataclass
class RainbowOption(Product):
    """Worst-of / best-of sur performances : payoff (ω(min_i ou max_i S_i(T)/S_i(0) - K))^+."""
    strike: float
    maturity: float
    kind: str = "worst"
    option_type: str = "put"
    initial_levels: np.ndarray | None = None

    @property
    def observation_times(self):
        return np.array([self.maturity])

    def payoff(self, paths: Paths) -> np.ndarray:
        ref = paths.spot[:, 0, :] if self.initial_levels is None else np.asarray(self.initial_levels)
        perf = paths.spot[:, -1, :] / ref
        x = perf.min(axis=1) if self.kind == "worst" else perf.max(axis=1)
        w = option_sign(self.option_type)
        return np.maximum(w * (x - self.strike), 0.0) * paths.df(-1)
