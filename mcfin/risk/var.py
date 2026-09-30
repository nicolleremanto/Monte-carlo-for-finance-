"""VaR et Expected Shortfall Monte Carlo d'un book d'options actions.

Méthodologie (risque de marché, FRTB)
-------------------------------------
* Facteurs de risque : log-rendements des spots sur l'horizon h
  (gaussiens ou Student-t multivariés à queues épaisses, même matrice de
  covariance) et chocs de volatilité implicite (lognormaux, corrélés
  négativement au spot : « leverage effect »).
* Revalorisation complète (full revaluation) de chaque option en t+h avec
  la maturité résiduelle, ou approximation delta-gamma-vega (Taylor).
* VaR_α = -q_{1-α}(P&L) ;  ES_α = -E[P&L | P&L <= -VaR_α].
  FRTB : ES à 97.5 %, horizon 10 jours (liquidité), cohérente (sous-additive)
  contrairement à la VaR.
* Intervalles de confiance par bootstrap ; contributions d'Euler à l'ES
  (ES = Σ_i -E[P&L_i | queue], allocation additive exacte par homogénéité).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..analytics.black_scholes import bs_greeks, bs_price

__all__ = ["OptionBook", "OptionPosition", "simulate_pnl", "var_es"]


@dataclass
class OptionPosition:
    asset: int
    quantity: float
    strike: float | None = None  # None => position en sous-jacent (action)
    maturity: float = 1.0
    option_type: str = "call"
    implied_vol: float = 0.2


@dataclass
class OptionBook:
    spots: np.ndarray
    vols: np.ndarray  # volatilités historiques des spots (annualisées)
    corr: np.ndarray
    rate: float = 0.0
    positions: list[OptionPosition] = field(default_factory=list)

    def value(self, spots: np.ndarray, vol_mult: np.ndarray | float = 1.0, dt: float = 0.0):
        """Valeur de chaque position pour des spots (n, d) : renvoie (n, n_pos)."""
        spots = np.atleast_2d(spots)
        vm = np.broadcast_to(np.asarray(vol_mult, float), spots.shape)
        out = []
        for p in self.positions:
            s = spots[:, p.asset]
            if p.strike is None:
                out.append(p.quantity * s)
            else:
                out.append(
                    p.quantity
                    * bs_price(
                        s,
                        p.strike,
                        max(p.maturity - dt, 1e-8),
                        self.rate,
                        p.implied_vol * vm[:, p.asset],
                        0.0,
                        p.option_type,
                    )
                )
        return np.column_stack(out)

    def greeks(self):
        """Sensibilités agrégées par actif : delta, gamma, vega (par unité de vol
        relative), theta."""
        d = self.spots.size
        delta, gamma, vega, theta = np.zeros(d), np.zeros(d), np.zeros(d), 0.0
        for p in self.positions:
            if p.strike is None:
                delta[p.asset] += p.quantity
                continue
            g = bs_greeks(
                self.spots[p.asset], p.strike, p.maturity, self.rate, p.implied_vol, 0.0, p.option_type
            )
            delta[p.asset] += p.quantity * g["delta"]
            gamma[p.asset] += p.quantity * g["gamma"]
            vega[p.asset] += p.quantity * g["vega"] * p.implied_vol
            theta += p.quantity * g["theta"]
        return delta, gamma, vega, theta


def simulate_pnl(
    book: OptionBook,
    horizon: float = 10 / 252,
    n_scenarios: int = 100_000,
    dist: str = "student",
    dof: float = 4.0,
    vol_of_vol: float = 0.0,
    spot_vol_corr: float = -0.5,
    method: str = "full",
    seed: int = 0,
):
    """Scénarios de P&L par position (n, n_pos) ; method ∈ {full, delta, delta-gamma}."""
    rng = np.random.default_rng(seed)
    d = book.spots.size
    chol = np.linalg.cholesky(book.corr)
    z = rng.standard_normal((n_scenarios, d)) @ chol.T
    if dist == "student":
        # t multivariée de variance unitaire : Z / sqrt(W/ν) · sqrt((ν-2)/ν)
        w = rng.chisquare(dof, size=(n_scenarios, 1))
        z = z / np.sqrt(w / dof) * np.sqrt((dof - 2) / dof)
    sig = book.vols * np.sqrt(horizon)
    ret = z * sig - 0.5 * sig**2
    new_spots = book.spots * np.exp(ret)
    zv = spot_vol_corr * z + np.sqrt(1 - spot_vol_corr**2) * rng.standard_normal((n_scenarios, d))
    vol_mult = np.exp(vol_of_vol * np.sqrt(horizon) * zv - 0.5 * vol_of_vol**2 * horizon)
    v0 = book.value(book.spots[None, :])[0]
    if method == "full":
        return book.value(new_spots, vol_mult, horizon) - v0
    # approximations de Taylor position par position
    ds = new_spots - book.spots
    dvol = vol_mult - 1.0
    cols = []
    for p in book.positions:
        if p.strike is None:
            cols.append(p.quantity * ds[:, p.asset])
            continue
        g = bs_greeks(book.spots[p.asset], p.strike, p.maturity, book.rate, p.implied_vol, 0.0, p.option_type)
        pnl = (
            g["delta"] * ds[:, p.asset] + g["theta"] * horizon + g["vega"] * p.implied_vol * dvol[:, p.asset]
        )
        if method == "delta-gamma":
            pnl = pnl + 0.5 * g["gamma"] * ds[:, p.asset] ** 2
        cols.append(p.quantity * pnl)
    return np.column_stack(cols)


def var_es(
    pnl: np.ndarray, alpha_var: float = 0.99, alpha_es: float = 0.975, n_bootstrap: int = 200, seed: int = 0
) -> dict:
    """VaR, ES, IC bootstrap à 95 % et contributions d'Euler à l'ES.

    ``pnl`` : (n,) P&L total ou (n, n_pos) P&L par position.
    """
    pnl = np.asarray(pnl, dtype=float)
    per_pos = pnl if pnl.ndim == 2 else pnl[:, None]
    total = per_pos.sum(axis=1)

    def risk(x):
        var = -np.quantile(x, 1 - alpha_var)
        q = np.quantile(x, 1 - alpha_es)
        es = -x[x <= q].mean()
        return var, es

    var, es = risk(total)
    rng = np.random.default_rng(seed)
    boot = np.array([risk(total[rng.integers(0, total.size, total.size)]) for _ in range(n_bootstrap)])
    q = np.quantile(total, 1 - alpha_es)
    tail = total <= q
    contrib = -per_pos[tail].mean(axis=0)
    return {
        "VaR": float(var),
        "ES": float(es),
        "VaR_CI95": tuple(np.quantile(boot[:, 0], [0.025, 0.975])),
        "ES_CI95": tuple(np.quantile(boot[:, 1], [0.025, 0.975])),
        "ES_contributions": contrib,
    }
