"""Formules fermées d'options exotiques sous Black-Scholes.

Servent à valider les moteurs Monte Carlo et comme variables de contrôle :

* Merton (1976) : série de Black-Scholes conditionnée au nombre de sauts ;
* barrières continues (Reiner & Rubinstein 1991, notations de Haug 2007) et
  correction de continuité de Broadie-Glasserman-Kou (1997) ;
* asiatique géométrique discrète (Kemna & Vorst 1990) ;
* lookback à strike flottant (Goldman, Sosin & Gatto 1979).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, ndtr, zeta

from .black_scholes import bs_price, option_sign

__all__ = [
    "BGK_BETA",
    "barrier_price",
    "bgk_shift",
    "geometric_asian_price",
    "lookback_floating_price",
    "merton_price",
]

#: β = -ζ(1/2)/sqrt(2π) ≈ 0.5826 (Broadie, Glasserman & Kou 1997)
BGK_BETA = float(-zeta(0.5) / np.sqrt(2 * np.pi))


def merton_price(S, K, T, r, sigma, lam, mu_j, sigma_j, q=0.0, option_type="call", n_terms: int = 120):
    """Merton jump-diffusion : ln(1+J) ~ N(μ_J, δ²), sauts Poisson(λ).

    C = Σ_n e^{-λ'T} (λ'T)^n / n! · BS(S, K, T, r_n, σ_n)
    avec k̄ = e^{μ_J+δ²/2} - 1, λ' = λ(1+k̄), σ_n² = σ² + nδ²/T,
    r_n = r - λk̄ + n ln(1+k̄)/T.
    """
    kbar = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    lam_p = lam * (1.0 + kbar)
    total = 0.0
    for n in range(n_terms):
        w = np.exp(-lam_p * T + n * np.log(lam_p * T) - gammaln(n + 1)) if lam_p > 0 else float(n == 0)
        sig_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)
        r_n = r - lam * kbar + n * np.log1p(kbar) / T
        # BS actualisé au taux r_n : les poids de Poisson en λ' absorbent e^{(r_n - r)T}
        total = total + w * bs_price(S, K, T, r_n, sig_n, q, option_type)
    return total


def bgk_shift(barrier, sigma, dt, direction: str):
    """Barrière continue équivalente à une barrière discrète de pas dt :
    H·exp(+β σ sqrt(dt)) si barrière haute, H·exp(-β σ sqrt(dt)) si basse."""
    s = 1.0 if direction == "up" else -1.0
    return barrier * np.exp(s * BGK_BETA * sigma * np.sqrt(dt))


def barrier_price(S, K, H, T, r, sigma, q=0.0, option_type="call", barrier_type="down-and-out"):
    """Options barrières à surveillance continue, sans rebate.

    barrier_type ∈ {down-and-out, down-and-in, up-and-out, up-and-in}.
    """
    phi = option_sign(option_type)
    b = r - q
    sT = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma**2) / sigma**2
    x1 = np.log(S / K) / sT + (1 + mu) * sT
    x2 = np.log(S / H) / sT + (1 + mu) * sT
    y1 = np.log(H**2 / (S * K)) / sT + (1 + mu) * sT
    y2 = np.log(H / S) / sT + (1 + mu) * sT
    down = barrier_type.startswith("down")
    eta = 1.0 if down else -1.0
    dq, dr = np.exp((b - r) * T), np.exp(-r * T)
    A = phi * S * dq * ndtr(phi * x1) - phi * K * dr * ndtr(phi * x1 - phi * sT)
    B = phi * S * dq * ndtr(phi * x2) - phi * K * dr * ndtr(phi * x2 - phi * sT)
    C = phi * S * dq * (H / S) ** (2 * (mu + 1)) * ndtr(eta * y1) - phi * K * dr * (H / S) ** (2 * mu) * ndtr(
        eta * y1 - eta * sT
    )
    D = phi * S * dq * (H / S) ** (2 * (mu + 1)) * ndtr(eta * y2) - phi * K * dr * (H / S) ** (2 * mu) * ndtr(
        eta * y2 - eta * sT
    )
    call = phi > 0
    kh = K > H
    table = {
        # (call?, K>H?) -> formule
        ("down-and-in", True, True): C,
        ("down-and-in", True, False): A - B + D,
        ("down-and-out", True, True): A - C,
        ("down-and-out", True, False): B - D,
        ("up-and-in", True, True): A,
        ("up-and-in", True, False): B - C + D,
        ("up-and-out", True, True): 0.0 * A,
        ("up-and-out", True, False): A - B + C - D,
        ("down-and-in", False, True): B - C + D,
        ("down-and-in", False, False): A,
        ("down-and-out", False, True): A - B + C - D,
        ("down-and-out", False, False): 0.0 * A,
        ("up-and-in", False, True): A - B + D,
        ("up-and-in", False, False): C,
        ("up-and-out", False, True): B - D,
        ("up-and-out", False, False): A - C,
    }
    price = table[(barrier_type, bool(call), bool(kh))]
    # barrière déjà franchie à l'origine
    breached = (S <= H) if down else (S >= H)
    if breached:
        return float(bs_price(S, K, T, r, sigma, q, option_type)) if barrier_type.endswith("in") else 0.0
    return float(price)


def geometric_asian_price(S, K, T, r, sigma, fixing_times, q=0.0, option_type="call"):
    """Asiatique à moyenne géométrique discrète (fixings t_1..t_n), paiement en T.

    ln G ~ N(μ_G, σ_G²) avec μ_G = ln S + (r - q - σ²/2) t̄ et
    σ_G² = σ²/n² Σ_i Σ_j min(t_i, t_j).
    """
    t = np.asarray(fixing_times, dtype=float)
    n = t.size
    mu_g = np.log(S) + (r - q - 0.5 * sigma**2) * t.mean()
    var_g = sigma**2 * np.minimum.outer(t, t).sum() / n**2
    sd = np.sqrt(var_g)
    w = option_sign(option_type)
    d2 = (mu_g - np.log(K)) / sd
    d1 = d2 + sd
    fwd_g = np.exp(mu_g + 0.5 * var_g)
    return float(np.exp(-r * T) * w * (fwd_g * ndtr(w * d1) - K * ndtr(w * d2)))


def lookback_floating_price(S, T, r, sigma, q=0.0, option_type="call"):
    """Lookback à strike flottant, surveillance continue, émis à la date 0
    (min = max = S) : call paie S_T - min S, put paie max S - S_T (Hull)."""
    b = r - q
    sT = sigma * np.sqrt(T)
    a1 = (b + 0.5 * sigma**2) * T / sT
    a2 = a1 - sT
    k = sigma**2 / (2 * b)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    if option_type == "call":
        return float(
            S * dq * ndtr(a1)
            - S * dq * k * ndtr(-a1)
            - S * dr * (ndtr(a2) - k * ndtr(-a1 + 2 * b * np.sqrt(T) / sigma))
        )
    # put : max(S) - S_T, avec b1 = (b + σ²/2)√T/σ
    return float(
        S * dr * (ndtr(-a2) - k * ndtr(a1 - 2 * b * np.sqrt(T) / sigma))
        + S * dq * k * ndtr(a1)
        - S * dq * ndtr(-a1)
    )
