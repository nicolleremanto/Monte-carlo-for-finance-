"""Autocall « Phoenix » worst-of — le produit structuré action le plus vendu.

Mécanique (sur la performance worst-of P_t = min_i S_i(t)/S_i(0)) :

* à chaque date d'observation t_k :
    - coupon c_k si P_{t_k} >= barrière coupon (avec effet mémoire : on paie
      aussi les coupons manqués) ;
    - remboursement anticipé (autocall) du nominal si P_{t_k} >= barrière de
      rappel (souvent dégressive : « step-down ») ;
* à maturité, si non rappelé :
    - nominal intégral si aucune activation de la barrière de protection ;
    - sinon nominal × min(P_T, 1) : l'investisseur est vendeur d'un put
      down-and-in worst-of.

La banque émettrice est acheteuse de ce put et vendeuse des digitales de
coupon/rappel : elle est longue corrélation et longue volatilité en bas du
smile (d'où l'importance du modèle de smile : LV/LSV plutôt que BS), et
porte un fort gamma digital près des barrières.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.results import Paths
from .base import Product, time_index

__all__ = ["PhoenixAutocall"]


@dataclass
class PhoenixAutocall(Product):
    observation_dates: np.ndarray
    autocall_barrier: float | np.ndarray = 1.0
    coupon_barrier: float | np.ndarray = 0.7
    coupon: float | np.ndarray = 0.02
    protection_barrier: float = 0.6
    memory: bool = True
    ki_monitoring: str = "european"      # "european" (à maturité) ou "daily"
    non_call_periods: int = 0
    notional: float = 1.0
    initial_levels: float | np.ndarray | None = None   # niveaux de strike figés à l'émission
    _daily: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.observation_dates = np.asarray(self.observation_dates, dtype=float)
        if self.ki_monitoring == "daily":
            T = self.observation_dates[-1]
            n = int(round(252 * T))
            self._daily = T * np.arange(1, n + 1) / n

    @property
    def observation_times(self):
        if self._daily is None:
            return self.observation_dates
        return np.union1d(np.round(self.observation_dates, 12), np.round(self._daily, 12))

    def _perf(self, paths: Paths) -> np.ndarray:
        """Performance worst-of. Les niveaux initiaux sont ceux fixés à
        l'émission (``initial_levels``) ; par défaut le spot en t = 0 (pricing
        à l'émission). Pour les Greeks d'un produit vivant, il faut figer les
        niveaux : sinon un choc de spot est neutralisé par la normalisation."""
        s = paths.spot
        ref = s[:, :1] if self.initial_levels is None else np.asarray(self.initial_levels, float)
        if s.ndim == 2:
            return s / ref
        ref = s[:, :1, :] if self.initial_levels is None else ref[None, None, :]
        return (s / ref).min(axis=2)

    def cashflows(self, paths: Paths):
        """Renvoie (valeur actualisée par trajectoire, proba de rappel par date,
        indicatrice de perte en capital)."""
        perf = self._perf(paths)
        idx = time_index(paths, self.observation_dates)
        n_obs = idx.size
        ac = np.broadcast_to(np.asarray(self.autocall_barrier, float), (n_obs,))
        cb = np.broadcast_to(np.asarray(self.coupon_barrier, float), (n_obs,))
        cpn = np.broadcast_to(np.asarray(self.coupon, float), (n_obs,))
        n = perf.shape[0]
        alive = np.ones(n, dtype=bool)
        missed = np.zeros(n)
        pv = np.zeros(n)
        call_prob = np.zeros(n_obs)
        for k, j in enumerate(idx):
            p = perf[:, j]
            df = paths.df(j)
            pay = alive & (p >= cb[k])
            n_cpn = 1.0 + missed if self.memory else 1.0
            pv += np.where(pay, self.notional * cpn[k] * n_cpn * df, 0.0)
            missed = np.where(alive & ~pay, missed + 1.0, np.where(pay, 0.0, missed))
            if k < n_obs - 1 and k >= self.non_call_periods:
                called = alive & (p >= ac[k])
                pv += np.where(called, self.notional * df, 0.0)
                call_prob[k] = called.mean()
                alive &= ~called
        p_T = perf[:, idx[-1]]
        if self.ki_monitoring == "daily":
            knocked = perf[:, 1: idx[-1] + 1].min(axis=1) < self.protection_barrier
        else:
            knocked = p_T < self.protection_barrier
        redemption = np.where(knocked, self.notional * np.minimum(p_T, 1.0), self.notional)
        pv += np.where(alive, redemption * paths.df(idx[-1]), 0.0)
        call_prob[-1] = alive.mean()  # remboursement à maturité
        loss = alive & knocked & (p_T < 1.0)
        return pv, call_prob, loss

    def payoff(self, paths: Paths) -> np.ndarray:
        return self.cashflows(paths)[0]

    def analytics(self, paths: Paths) -> dict:
        """Statistiques de structuration : probas de rappel, durée de vie
        espérée, probabilité de perte en capital, perte moyenne sachant perte."""
        pv, call_prob, loss = self.cashflows(paths)
        perf_T = self._perf(paths)[:, time_index(paths, self.observation_dates[-1])]
        return {
            "price": float(pv.mean()),
            "redemption_prob_by_date": call_prob,
            "expected_life": float(np.dot(call_prob, self.observation_dates)),
            "prob_capital_loss": float(loss.mean()),
            "expected_loss_given_loss": float((1 - perf_T[loss]).mean()) if loss.any() else 0.0,
        }
