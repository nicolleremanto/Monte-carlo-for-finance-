"""Options bermudéennes/américaines : Longstaff-Schwartz et borne duale.

Longstaff & Schwartz (2001) — borne inférieure
----------------------------------------------
Rétro-induction sur les dates d'exercice t_1 < ... < t_K. La valeur de
continuation C_k(x) = E[flux futurs actualisés | X_{t_k} = x] est approchée
par régression linéaire (MCO) des flux réalisés sur une base de fonctions
de l'état, restreinte aux trajectoires dans la monnaie. On exerce si
h_k > Ĉ_k. Pour éviter le biais de « prévoyance » (les coefficients voient
les trajectoires qu'ils valorisent), on estime la règle sur un jeu de
trajectoires et on valorise sur un jeu indépendant : le prix obtenu est
alors une borne inférieure (règle sous-optimale) sans biais.

Andersen & Broadie (2004) — borne supérieure duale
--------------------------------------------------
Dualité de Rogers (2002) / Haugh & Kogan (2004) : pour toute martingale M
nulle en 0,
    V_0 <= E[ max_k (h̃_k - M_k) ]      (h̃ = payoff actualisé).
L'égalité est atteinte pour la partie martingale de l'enveloppe de Snell.
A-B construisent M à partir de la règle d'exercice L-S :
    L̃_k = valeur (actualisée) de la règle démarrée en t_k,
    M_k = M_{k-1} + L̃_k - E_{k-1}[L̃_k],
les espérances conditionnelles étant estimées par simulations imbriquées.
L'écart de dualité mesure la sous-optimalité de la règle d'exercice.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ..analytics.black_scholes import option_sign
from ..core.results import MCResult, Paths
from ..core.timegrid import TimeGrid
from ..market.curves import FlatCurve, as_curve
from .base import Product, time_indices

__all__ = ["BermudanOption", "LongstaffSchwartz", "andersen_broadie_upper_bound"]


@dataclass
class BermudanOption(Product):
    """Option bermudéenne sur un ou plusieurs actifs.

    kind : "vanilla" (1 actif), "max" / "min" (call/put sur max/min des
    spots), "basket" (moyenne pondérée des spots).
    """

    strike: float
    exercise_times: np.ndarray
    option_type: str = "put"
    kind: str = "vanilla"
    weights: np.ndarray | None = None

    @property
    def observation_times(self):
        return np.asarray(self.exercise_times, dtype=float)

    def underlying(self, s: np.ndarray) -> np.ndarray:
        if s.ndim == 1 or self.kind == "vanilla":
            return s if s.ndim == 1 else s[..., 0]
        if self.kind == "max":
            return s.max(axis=-1)
        if self.kind == "min":
            return s.min(axis=-1)
        w = np.ones(s.shape[-1]) / s.shape[-1] if self.weights is None else self.weights
        return s @ w

    def exercise_value(self, s: np.ndarray) -> np.ndarray:
        return np.maximum(option_sign(self.option_type) * (self.underlying(s) - self.strike), 0.0)

    def features(self, s: np.ndarray) -> np.ndarray:
        """Variables explicatives normalisées par le strike (spots triés si
        multi-actifs : la valeur d'un max-call est symétrique)."""
        x = s / self.strike
        if x.ndim == 1:
            return x[:, None]
        return np.sort(x, axis=-1)[..., ::-1]

    def payoff(self, paths: Paths) -> np.ndarray:  # exercice à la dernière date seulement
        return self.exercise_value(paths.spot[:, -1]) * paths.df(-1)


def _basis(x: np.ndarray, h: np.ndarray, degree: int, strike: float) -> np.ndarray:
    """Monômes de degré total <= degree des variables + payoff normalisé."""
    cols = [np.ones(x.shape[0])]
    m = x.shape[1]
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(m), d):
            cols.append(np.prod(x[:, combo], axis=1))
    cols.append(h / strike)
    return np.column_stack(cols)


class LongstaffSchwartz:
    def __init__(self, degree: int = 3, itm_only: bool = True):
        self.degree = degree
        self.itm_only = itm_only
        self.coeffs: list[np.ndarray | None] = []

    def _spot_at(self, paths: Paths, j: int) -> np.ndarray:
        return paths.spot[:, j]

    def fit(self, paths: Paths, product: BermudanOption) -> LongstaffSchwartz:
        idx = time_indices(paths, product.exercise_times)
        K = idx.size
        h = [product.exercise_value(self._spot_at(paths, j)) for j in idx]
        df = [np.broadcast_to(paths.df(j), (paths.n_paths,)) for j in idx]
        cf = h[-1] * df[-1]
        self.coeffs = [None] * K
        for k in range(K - 2, -1, -1):
            s = self._spot_at(paths, idx[k])
            itm = h[k] > 0 if self.itm_only else np.ones_like(h[k], dtype=bool)
            if itm.sum() < 10:
                continue
            X = _basis(product.features(s), h[k], self.degree, product.strike)
            y = cf / df[k]  # flux futurs exprimés en valeur t_k
            beta, *_ = np.linalg.lstsq(X[itm], y[itm], rcond=None)
            self.coeffs[k] = beta
            ex = itm & (h[k] > X @ beta)
            cf = np.where(ex, h[k] * df[k], cf)
        self.in_sample_price = float(cf.mean())
        return self

    def exercise(
        self, k: int, s: np.ndarray, h: np.ndarray, product: BermudanOption, last: bool
    ) -> np.ndarray:
        """Décision d'exercice de la règle estimée à la date d'indice k."""
        if last:
            return h > 0
        beta = self.coeffs[k]
        if beta is None:
            return np.zeros_like(h, dtype=bool)
        X = _basis(product.features(s), h, self.degree, product.strike)
        return (h > 0) & (h > X @ beta)

    def price(self, paths: Paths, product: BermudanOption) -> MCResult:
        """Valorisation hors échantillon (borne inférieure)."""
        idx = time_indices(paths, product.exercise_times)
        K = idx.size
        n = paths.n_paths
        alive = np.ones(n, dtype=bool)
        cf = np.zeros(n)
        tau = np.full(n, np.nan)
        for k, j in enumerate(idx):
            s = self._spot_at(paths, j)
            h = product.exercise_value(s)
            ex = alive & self.exercise(k, s, h, product, k == K - 1)
            cf = np.where(ex, h * np.broadcast_to(paths.df(j), (n,)), cf)
            tau = np.where(ex, paths.times[j], tau)
            alive &= ~ex
        res = MCResult(
            float(cf.mean()), float(cf.std(ddof=1) / np.sqrt(n)), n, method="LSM (hors échantillon)"
        )
        res.extra["mean_exercise_time"] = float(np.nanmean(tau))
        res.extra["prob_exercise"] = float(np.isfinite(tau).mean())
        return res


def andersen_broadie_upper_bound(
    model,
    product: BermudanOption,
    lsm: LongstaffSchwartz,
    lower_bound: float,
    n_outer: int = 1000,
    n_inner: int = 500,
    seed: int = 1234,
    chunk: int = 200,
    lower_bound_stderr: float = 0.0,
) -> MCResult:
    """Borne supérieure duale d'Andersen-Broadie (modèle Black-Scholes, courbes plates).

    ``lower_bound`` (et son erreur standard ``lower_bound_stderr``) est le prix
    LSM hors échantillon ; il sert de E_0[L̃_1] dans la martingale et son bruit
    est propagé à l'erreur standard de la borne supérieure.

    Pour le GBM, S_{t_j} = S_{t_k} · R_{k,j} où R est indépendant de S_{t_k} :
    les simulations imbriquées se font sur des trajectoires « unitaires ».
    """
    from ..models.equity import BlackScholes

    if not isinstance(model, BlackScholes):
        raise TypeError("implémenté pour BlackScholes (multi-actifs)")
    r_curve = as_curve(model.rate)
    divs = model.div if isinstance(model.div, (list, tuple, np.ndarray)) else [model.div]
    # les trajectoires imbriquées « unitaires » partent de t = 0 : elles ne sont
    # correctes que si la dérive est invariante par translation en temps
    if not isinstance(r_curve, FlatCurve) or not all(isinstance(as_curve(q), FlatCurve) for q in divs):
        raise TypeError("courbes de taux et de dividendes plates requises")
    rng = np.random.default_rng(seed)
    t_ex = np.asarray(product.exercise_times, dtype=float)
    K = t_ex.size
    d = model.n_assets
    grid = TimeGrid.build(t_ex)
    z = rng.standard_normal((n_outer, grid.n_steps, model.n_factors))
    outer = model.simulate(grid, z)
    idx = time_indices(outer, t_ex)
    disc = r_curve.df(t_ex)
    unit = BlackScholes(
        spot=np.ones(d) if d > 1 else 1.0, vol=model.vol, rate=model.rate, div=model.div, corr=model.corr
    )

    h_disc = np.empty((n_outer, K))
    L = np.empty((n_outer, K))
    Q = np.zeros((n_outer, K))
    for k in range(K):
        s_k = outer.spot[:, idx[k]]
        h_k = product.exercise_value(s_k)
        h_disc[:, k] = h_k * disc[k]
        ex_k = lsm.exercise(k, s_k, h_k, product, k == K - 1)
        if k < K - 1:
            rel_times = t_ex[k + 1 :] - t_ex[k]
            g = TimeGrid.build(rel_times)
            for c in range(0, n_outer, chunk):
                sk = s_k[c : c + chunk]
                m = sk.shape[0]
                zi = rng.standard_normal((m * n_inner, g.n_steps, unit.n_factors))
                rel = unit.simulate(g, zi).spot[:, 1:]  # (m*n_in, K-k-1[, d])
                if d == 1:
                    s_in = np.repeat(sk, n_inner)[:, None] * rel
                else:
                    s_in = np.repeat(sk, n_inner, axis=0)[:, None, :] * rel
                alive = np.ones(m * n_inner, dtype=bool)
                val = np.zeros(m * n_inner)
                for jj in range(K - k - 1):
                    kk = k + 1 + jj
                    s_j = s_in[:, jj]
                    h_j = product.exercise_value(s_j)
                    ex = alive & lsm.exercise(kk, s_j, h_j, product, kk == K - 1)
                    val = np.where(ex, h_j * disc[kk], val)
                    alive &= ~ex
                Q[c : c + chunk, k] = val.reshape(m, n_inner).mean(axis=1)
        L[:, k] = np.where(ex_k, h_disc[:, k], Q[:, k])
    M = np.empty((n_outer, K))
    M[:, 0] = L[:, 0] - lower_bound
    for k in range(1, K):
        M[:, k] = M[:, k - 1] + L[:, k] - Q[:, k - 1]
    dual = np.max(h_disc - M, axis=1)
    upper = float(dual.mean())
    # la borne inférieure (constante estimée) entre dans chaque échantillon dual :
    # son bruit s'ajoute à celui de la dispersion du dual (échantillons indépendants)
    se_dual = float(dual.std(ddof=1) / np.sqrt(n_outer))
    res = MCResult(
        upper, float(np.hypot(se_dual, lower_bound_stderr)), n_outer, method="Andersen-Broadie (dual)"
    )
    res.extra["duality_gap"] = upper - lower_bound
    res.extra["point_estimate"] = 0.5 * (upper + lower_bound)
    return res
