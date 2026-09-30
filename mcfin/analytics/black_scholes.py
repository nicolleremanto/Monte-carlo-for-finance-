"""Formules fermées Black-Scholes / Black-76 / Bachelier et volatilité implicite.

Toutes les fonctions sont vectorisées (broadcasting numpy). Le type d'option
est "call"/"put" (ou +1/-1).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

__all__ = [
    "bachelier_implied_vol",
    "bachelier_price",
    "black_implied_vol",
    "black_price",
    "bs_digital_price",
    "bs_greeks",
    "bs_price",
    "implied_vol",
    "option_sign",
]

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def option_sign(option_type) -> np.ndarray | float:
    if isinstance(option_type, str):
        t = option_type.lower()
        if t in ("call", "c"):
            return 1.0
        if t in ("put", "p"):
            return -1.0
        raise ValueError(f"type d'option inconnu : {option_type}")
    return np.asarray(option_type, dtype=float)


def _pdf(x):
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def black_price(F, K, T, df, sigma, option_type="call"):
    """Black-76 : df · E[(ω(F_T - K))^+] avec F_T lognormal de volatilité σ."""
    F, K, T, df, sigma = map(lambda a: np.asarray(a, dtype=float), (F, K, T, df, sigma))
    w = option_sign(option_type)
    s = sigma * np.sqrt(T)
    intrinsic = np.maximum(w * (F - K), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * s * s) / s
        d2 = d1 - s
        val = w * (F * ndtr(w * d1) - K * ndtr(w * d2))
    return df * np.where(s > 0, val, intrinsic)


def bs_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Black-Scholes-Merton avec taux r et rendement du dividende q continus."""
    S, K, T, r, q = map(lambda a: np.asarray(a, dtype=float), (S, K, T, r, q))
    F = S * np.exp((r - q) * T)
    return black_price(F, K, T, np.exp(-r * T), sigma, option_type)


def bs_greeks(S, K, T, r, sigma, q=0.0, option_type="call") -> dict:
    """Grecques analytiques jusqu'à l'ordre 3.

    Conventions : dérivées « brutes » (vega, rho, epsilon pour une variation
    de 1 et non de 1 %) ; les sensibilités au temps (theta, charm, color) sont
    des dérivées par rapport au temps calendaire t, soit -∂/∂T.

    ============  ==============================  ===========================
    clé           définition                      usage desk
    ============  ==============================  ===========================
    delta         ∂V/∂S                           couverture en sous-jacent
    gamma         ∂²V/∂S²                         convexité, P&L de gamma
    vega          ∂V/∂σ                           risque de vol
    theta         ∂V/∂t                           portage (Θ ≈ -½Γσ²S²)
    rho           ∂V/∂r                           risque de taux
    epsilon       ∂V/∂q                           risque dividende/repo
    vanna         ∂²V/∂S∂σ                        skew / delta-vega croisé
    volga         ∂²V/∂σ²  (vomma)                convexité en vol (smile)
    charm         ∂Δ/∂t                           dérive du delta (« delta bleed »)
    speed         ∂Γ/∂S                           stabilité du gamma
    zomma         ∂Γ/∂σ                           gamma vs vol
    color         ∂Γ/∂t                           dérive du gamma
    ultima        ∂³V/∂σ³                         ordre 3 en vol
    dual_delta    ∂V/∂K                           densité cumulée risque-neutre
    dual_gamma    ∂²V/∂K²                         densité risque-neutre (Breeden-Litzenberger)
    ============  ==============================  ===========================
    """
    S, K, T, r, q, sigma = map(lambda a: np.asarray(a, dtype=float), (S, K, T, r, q, sigma))
    w = option_sign(option_type)
    sqT = np.sqrt(T)
    sT = sigma * sqT
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sT
    d2 = d1 - sT
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    pdf1, pdf2 = _pdf(d1), _pdf(d2)
    delta = w * dq * ndtr(w * d1)
    gamma = dq * pdf1 / (S * sT)
    vega = S * dq * pdf1 * sqT
    theta = -S * dq * pdf1 * sigma / (2 * sqT) + w * q * S * dq * ndtr(w * d1) - w * r * K * dr * ndtr(w * d2)
    rho = w * K * T * dr * ndtr(w * d2)
    epsilon = -w * S * T * dq * ndtr(w * d1)
    vanna = -dq * pdf1 * d2 / sigma
    volga = vega * d1 * d2 / sigma
    # ∂d1/∂T et ∂d2/∂T interviennent dans charm et color
    dd1_dT = (2 * (r - q) * T - d2 * sT) / (2 * T * sT)
    charm = w * q * dq * ndtr(w * d1) - dq * pdf1 * dd1_dT
    speed = -gamma / S * (d1 / sT + 1.0)
    zomma = gamma * (d1 * d2 - 1.0) / sigma
    color = gamma * (q + 1.0 / (2 * T) + d1 * dd1_dT)
    ultima = -vega / sigma**2 * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2)
    dual_delta = -w * dr * ndtr(w * d2)
    dual_gamma = dr * pdf2 / (K * sT)
    return {
        "price": bs_price(S, K, T, r, sigma, q, option_type),
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "epsilon": epsilon,
        "vanna": vanna,
        "volga": volga,
        "charm": charm,
        "speed": speed,
        "zomma": zomma,
        "color": color,
        "ultima": ultima,
        "dual_delta": dual_delta,
        "dual_gamma": dual_gamma,
    }


def bs_digital_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Digitale cash-or-nothing payant 1 si ω(S_T - K) > 0."""
    S, K, T, r, q, sigma = map(lambda a: np.asarray(a, dtype=float), (S, K, T, r, q, sigma))
    w = option_sign(option_type)
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-r * T) * ndtr(w * d2)


def bachelier_price(F, K, T, df, sigma_n, option_type="call"):
    """Modèle normal (Bachelier) — standard des swaptions depuis les taux négatifs."""
    F, K, T, df, sigma_n = map(lambda a: np.asarray(a, dtype=float), (F, K, T, df, sigma_n))
    w = option_sign(option_type)
    s = sigma_n * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (F - K) / s
        val = w * (F - K) * ndtr(w * d) + s * _pdf(d)
    return df * np.where(s > 0, val, np.maximum(w * (F - K), 0.0))


def black_implied_vol(price, F, K, T, df=1.0, option_type="call", tol: float = 1e-12, max_iter: int = 100):
    """Volatilité implicite Black-76, vectorisée.

    Newton sur s = σ√T, sécurisé par un encadrement [lo, hi] (bissection si le
    pas de Newton sort de l'encadrement). Point de départ au point
    d'inflection s* = sqrt(2|ln(F/K)|) où la convergence de Newton est
    monotone (Jäckel, « By implication », 2006). Renvoie NaN si le prix viole
    les bornes d'arbitrage.
    """
    price, F, K, T, df = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (price, F, K, T, df)))
    w = np.broadcast_to(option_sign(option_type), price.shape)
    # On inverse toujours l'option hors de la monnaie (valeur temps pure) :
    # parité call-put pour convertir, puis tolérance relative sur ce prix.
    m = np.where(K >= F, 1.0, -1.0)
    c = price / df - np.where(w == m, 0.0, m * (K - F))
    valid = (c > 1e-300) & (c < np.where(m > 0, F, K))
    x = np.log(F / K)
    s = np.sqrt(2.0 * np.abs(x))
    s = np.where(s < 1e-8, _SQRT_2PI * np.abs(c) / F, s)
    s = np.clip(s, 1e-6, 5.0)
    lo = np.zeros_like(s)
    hi = np.full_like(s, 10.0)
    done = ~valid
    for _ in range(max_iter):
        d1 = x / s + 0.5 * s
        f = m * (F * ndtr(m * d1) - K * ndtr(m * (d1 - s))) - c
        done = done | (np.abs(f) <= tol * np.abs(c)) | (hi - lo < 1e-15)
        if np.all(done):
            break
        vega = F * _pdf(d1)
        hi = np.where(f > 0, s, hi)
        lo = np.where(f < 0, s, lo)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_new = s - f / vega
        bad = ~np.isfinite(s_new) | (s_new <= lo) | (s_new >= hi)
        s_new = np.where(bad, 0.5 * (lo + hi), s_new)
        s = np.where(done, s, s_new)
    vol = s / np.sqrt(T)
    out = np.where(valid, vol, np.nan)
    return out if out.ndim else float(out)


def implied_vol(price, S, K, T, r, q=0.0, option_type="call", **kw):
    """Volatilité implicite Black-Scholes (spot, taux et dividende continus)."""
    S, T, r, q = map(lambda a: np.asarray(a, dtype=float), (S, T, r, q))
    return black_implied_vol(price, S * np.exp((r - q) * T), K, T, np.exp(-r * T), option_type, **kw)


def bachelier_implied_vol(price, F, K, T, df=1.0, option_type="call"):
    """Volatilité normale implicite (inversion par Brent, scalaire)."""
    from scipy.optimize import brentq

    f = lambda s: float(bachelier_price(F, K, T, df, s, option_type)) - price
    return brentq(f, 1e-10, 1.0)
