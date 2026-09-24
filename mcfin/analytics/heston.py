"""Heston (1993) et Bates (1996) semi-analytiques + calibration.

Fonction caractéristique de x_T = ln(S_T / F_T) sous la mesure risque-neutre,
dans la formulation « little Heston trap » d'Albrecher, Mayer, Schoutens &
Tistaert (2007), continue en u (pas de saut de branche du logarithme
complexe, contrairement à la formulation originale de Heston) :

    β = κ - ρ ξ i u,   d = sqrt(β² + ξ² (i u + u²)),   g = (β - d)/(β + d)
    C = κθ/ξ² [ (β - d) T - 2 ln((1 - g e^{-dT}) / (1 - g)) ]
    D = (β - d)/ξ² · (1 - e^{-dT}) / (1 - g e^{-dT})
    φ(u) = exp(C + D v0)

Prix par la formule de Lewis (2000), intégrale unique à décroissance en
1/u² (plus stable que Gil-Pelaez) :

    C(K) = S0 e^{-qT} - sqrt(F K) e^{-rT}/π ∫_0^∞ Re[e^{i u k} φ(u - i/2)] / (u² + 1/4) du,
    k = ln(F/K).

La quadrature est de Gauss-Legendre par panneaux (vectorisée sur les
strikes) : suffisamment rapide pour la calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .black_scholes import black_price, bs_greeks, implied_vol, option_sign

__all__ = ["HestonParams", "heston_cf", "heston_price", "heston_implied_vol",
           "calibrate_heston", "heston_expected_variance"]


@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    # sauts de Merton (Bates) : intensité, moyenne et écart-type du log-saut
    lam: float = 0.0
    mu_j: float = 0.0
    sigma_j: float = 0.0

    @property
    def feller(self) -> float:
        """2κθ/ξ² : > 1 => la variance n'atteint pas 0."""
        return 2 * self.kappa * self.theta / self.xi**2


def heston_cf(u, T: float, p: HestonParams):
    """φ(u) = E[exp(i u ln(S_T/F_T))] (u complexe accepté)."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    beta = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(beta * beta + p.xi**2 * (iu + u * u))
    g = (beta - d) / (beta + d)
    edt = np.exp(-d * T)
    C = p.kappa * p.theta / p.xi**2 * ((beta - d) * T
                                        - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
    D = (beta - d) / p.xi**2 * (1.0 - edt) / (1.0 - g * edt)
    out = C + D * p.v0
    if p.lam > 0:
        kbar = np.exp(p.mu_j + 0.5 * p.sigma_j**2) - 1.0
        out = out + p.lam * T * (np.exp(iu * p.mu_j - 0.5 * u * u * p.sigma_j**2) - 1.0
                                 - iu * kbar)
    return np.exp(out)


def _quadrature(u_max: float):
    """Nœuds/poids Gauss-Legendre par panneaux : fins près de 0, puis de largeur 16."""
    x, w = np.polynomial.legendre.leggauss(32)
    edges = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    while edges[-1] < u_max:
        edges.append(edges[-1] + 16.0)
    edges = np.asarray(edges)
    a, b = edges[:-1, None], edges[1:, None]
    nodes = (0.5 * (b - a) * x[None, :] + 0.5 * (a + b)).ravel()
    weights = (0.5 * (b - a) * w[None, :]).ravel()
    return nodes, weights


def _truncation(T: float, p: HestonParams, tol: float = 1e-14) -> float:
    u = np.geomspace(1.0, 2e4, 400)
    mag = np.abs(heston_cf(u - 0.5j, T, p)) / (u * u + 0.25)
    above = np.nonzero(mag > tol)[0]
    return float(u[above[-1]]) * 1.1 if above.size else 1.0


def heston_price(S0, K, T, r, q, params: HestonParams, option_type="call"):
    """Prix Heston/Bates d'options européennes (vectorisé sur K)."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    F = S0 * np.exp((r - q) * T)
    nodes, weights = _quadrature(_truncation(T, params))
    phi = heston_cf(nodes - 0.5j, T, params)
    k = np.log(F / K)
    integrand = np.real(np.exp(1j * np.outer(k, nodes)) * phi[None, :]) / (nodes**2 + 0.25)
    integral = integrand @ weights
    call = S0 * np.exp(-q * T) - np.sqrt(F * K) * np.exp(-r * T) / np.pi * integral
    w = option_sign(option_type)
    price = np.where(np.asarray(w) > 0, call, call - S0 * np.exp(-q * T) + K * np.exp(-r * T))
    # le prix ne peut être inférieur à la valeur intrinsèque actualisée
    intrinsic = np.maximum(w * (F - K), 0.0) * np.exp(-r * T)
    return np.maximum(price, intrinsic)


def heston_implied_vol(S0, K, T, r, q, params: HestonParams):
    """Smile implicite Black-Scholes du modèle de Heston (calls OTM/puts OTM)."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    F = S0 * np.exp((r - q) * T)
    otype = np.where(K >= F, 1.0, -1.0)
    prices = heston_price(S0, K, T, r, q, params, "call")
    prices = np.where(otype > 0, prices, prices - S0 * np.exp(-q * T) + K * np.exp(-r * T))
    return implied_vol(prices, S0, K, T, r, q, otype)


def heston_expected_variance(T: float, p: HestonParams) -> float:
    """E[(1/T)∫_0^T v_t dt] = θ + (v0 - θ)(1 - e^{-κT})/(κT) : strike du swap de variance
    (hors sauts)."""
    return p.theta + (p.v0 - p.theta) * (1 - np.exp(-p.kappa * T)) / (p.kappa * T)


def calibrate_heston(S0: float, r: float, q: float, maturities, strikes, market_vols,
                     x0=(0.04, 1.5, 0.04, 0.5, -0.6), feller_penalty: float = 0.0,
                     verbose: int = 0) -> tuple[HestonParams, dict]:
    """Calibration de Heston à une nappe de volatilités implicites.

    Minimise Σ ((P_model - P_mkt)/Vega_mkt)² ≈ Σ (σ_model - σ_mkt)² (erreurs en
    volatilité au premier ordre, sans inversion coûteuse), par
    Levenberg-Marquardt / trust-region réflectif sous contraintes de bornes.

    maturities, strikes, market_vols : tableaux 1D de même longueur (une
    ligne par option).
    """
    T = np.asarray(maturities, dtype=float)
    K = np.asarray(strikes, dtype=float)
    vol = np.asarray(market_vols, dtype=float)
    F = S0 * np.exp((r - q) * T)
    otype = np.where(K >= F, 1.0, -1.0)
    mkt = black_price(F, K, T, np.exp(-r * T), vol, otype)
    vega = bs_greeks(S0, K, T, r, vol, q, otype)["vega"]
    vega = np.maximum(vega, 1e-4 * S0)
    uniq = np.unique(T)

    def model_prices(x):
        p = HestonParams(*x)
        out = np.empty_like(K)
        for t in uniq:
            m = T == t
            c = heston_price(S0, K[m], t, r, q, p, "call")
            out[m] = np.where(otype[m] > 0, c,
                              c - S0 * np.exp(-q * t) + K[m] * np.exp(-r * t))
        return out

    def residuals(x):
        res = (model_prices(x) - mkt) / vega
        if feller_penalty > 0:
            v0, kappa, theta, xi, rho = x
            res = np.append(res, feller_penalty * max(0.0, xi**2 - 2 * kappa * theta))
        return res

    lb = [1e-4, 1e-2, 1e-4, 1e-2, -0.999]
    ub = [1.0, 20.0, 1.0, 5.0, 0.999]
    sol = least_squares(residuals, x0=np.asarray(x0, dtype=float), bounds=(lb, ub),
                        method="trf", x_scale="jac", verbose=verbose,
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
    params = HestonParams(*sol.x)
    fitted = implied_vol(model_prices(sol.x), S0, K, T, r, q, otype)
    info = {"rmse_vol": float(np.sqrt(np.mean((fitted - vol) ** 2))),
            "model_vols": fitted, "success": sol.success, "nfev": sol.nfev}
    return params, info
