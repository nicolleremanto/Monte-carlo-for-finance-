"""Inférence statistique sur le modèle de Black-Scholes.

Maximum de vraisemblance
------------------------
Pour des cours observés à pas Δ, les log-rendements r_i = ln(S_{i}/S_{i-1})
sont i.i.d. N(m Δ, σ² Δ) avec m = μ - σ²/2. L'EMV est explicite :

    σ̂² = (1/(nΔ)) Σ (r_i - r̄)²,     μ̂ = r̄/Δ + σ̂²/2,

et l'information de Fisher donne les variances asymptotiques

    Var(σ̂) ≈ σ² / (2n),      Var(μ̂) ≈ σ²/(nΔ) = σ²/T   (+ O(1/n)).

Résultat clé (et contre-intuitif) : la précision sur σ croît avec le
nombre d'observations n (échantillonner plus finement aide), mais celle sur
μ ne dépend que de la durée totale T = nΔ. Avec σ = 20 % et 10 ans de
données, l'IC à 95 % de μ a une demi-largeur de ≈ 12 % : la tendance est
statistiquement inobservable — heureusement, le prix d'une option n'en
dépend pas (changement de mesure de Girsanov).

Estimateurs de volatilité par l'amplitude (données OHLC)
--------------------------------------------------------
Exploiter les plus haut/bas d'une séance améliore l'efficacité :
* Parkinson (1980) : σ̂² = (ln H/L)² / (4 ln 2 · Δ)          (efficacité ≈ 5,2)
* Garman-Klass (1980) : σ̂²Δ = ½(ln H/L)² - (2 ln 2 - 1)(ln C/O)²   (≈ 7,4)
* Rogers-Satchell (1991) : σ̂²Δ = ln(H/C) ln(H/O) + ln(L/C) ln(L/O)
  (sans biais même en présence de tendance).
Efficacité = Var(close-to-close) / Var(estimateur), pour une séance.

Biais de discrétisation : si les plus haut/bas ne sont relevés que sur une
grille de m points (données réelles : ticks, bougies minute), l'extremum
continu est manqué et l'amplitude est sous-estimée d'environ
2β sqrt(1/m) écarts-types journaliers (β ≈ 0,5826, cf. Broadie-Glasserman-Kou) :
les estimateurs de range sont biaisés vers le bas, d'autant plus que m est
petit. Le test ``test_range_estimators_efficiency_and_discretisation_bias``
le vérifie.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = ["GBMFit", "fit_gbm_mle", "simulate_ohlc", "range_volatility"]


@dataclass
class GBMFit:
    mu: float
    sigma: float
    se_mu: float
    se_sigma: float
    n: int
    log_likelihood: float

    def ci(self, level: float = 0.95) -> dict:
        z = norm.ppf(0.5 + level / 2)
        return {"mu": (self.mu - z * self.se_mu, self.mu + z * self.se_mu),
                "sigma": (self.sigma - z * self.se_sigma, self.sigma + z * self.se_sigma)}


def fit_gbm_mle(prices: np.ndarray, dt: float) -> GBMFit:
    """EMV de (μ, σ) à partir d'une série de prix équi-espacés."""
    r = np.diff(np.log(np.asarray(prices, dtype=float)))
    n = r.size
    m = r.mean()
    s2 = np.mean((r - m) ** 2) / dt
    sigma = float(np.sqrt(s2))
    mu = float(m / dt + 0.5 * s2)
    ll = float(np.sum(norm.logpdf(r, loc=m, scale=np.sqrt(s2 * dt))))
    # méthode delta : μ̂ = r̄/Δ + σ̂²/2, avec r̄ et σ̂² asymptotiquement indépendants
    var_mu = s2 / (n * dt) + (sigma**2) ** 2 / (2 * n)
    return GBMFit(mu, sigma, float(np.sqrt(var_mu)), sigma / np.sqrt(2 * n), n, ll)


def simulate_ohlc(S0: float, mu: float, sigma: float, n_days: int, steps_per_day: int = 390,
                  dt_day: float = 1 / 252, seed: int | None = 0) -> dict:
    """Séances OHLC d'un GBM (plus haut/bas relevés sur une grille intra-journalière)."""
    rng = np.random.default_rng(seed)
    h = dt_day / steps_per_day
    z = rng.standard_normal((n_days, steps_per_day))
    log_incr = (mu - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * z
    intraday = np.cumsum(log_incr, axis=1)
    day_close = np.concatenate([[0.0], np.cumsum(intraday[:, -1])])
    opens = day_close[:-1]
    path = opens[:, None] + intraday
    high = np.maximum(path.max(axis=1), opens)
    low = np.minimum(path.min(axis=1), opens)
    return {k: S0 * np.exp(v) for k, v in
            {"open": opens, "high": high, "low": low, "close": day_close[1:]}.items()}


def range_volatility(ohlc: dict, dt: float = 1 / 252, method: str = "garman_klass") -> float:
    o, h, lo, c = (np.log(ohlc[k]) for k in ("open", "high", "low", "close"))
    hl, co = h - lo, c - o
    if method == "close_to_close":
        prev = np.concatenate([[o[0]], c[:-1]])
        var = np.mean((c - prev) ** 2)
    elif method == "parkinson":
        var = np.mean(hl**2) / (4 * np.log(2))
    elif method == "garman_klass":
        var = np.mean(0.5 * hl**2 - (2 * np.log(2) - 1) * co**2)
    elif method == "rogers_satchell":
        var = np.mean((h - c) * (h - o) + (lo - c) * (lo - o))
    else:
        raise ValueError(f"méthode inconnue : {method}")
    return float(np.sqrt(var / dt))
