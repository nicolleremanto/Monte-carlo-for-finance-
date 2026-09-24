"""Arbre binomial de Leisen-Reimer (1996) pour options américaines.

Référence « exacte » pour valider Longstaff-Schwartz. L'arbre LR (inversion
de Peizer-Pratt, méthode 2) converge en O(1/n²) pour les européennes,
contre O(1/n) pour Cox-Ross-Rubinstein, et reste très précis pour les
américaines.
"""
from __future__ import annotations

import numpy as np

from .black_scholes import option_sign

__all__ = ["leisen_reimer_price"]


def _peizer_pratt(x: float, n: int) -> float:
    return 0.5 + np.sign(x) * np.sqrt(
        0.25 - 0.25 * np.exp(-(x / (n + 1.0 / 3.0 + 0.1 / (n + 1))) ** 2 * (n + 1.0 / 6.0)))


def leisen_reimer_price(S, K, T, r, sigma, q=0.0, option_type="put",
                        american: bool = True, n_steps: int = 1001) -> float:
    n = n_steps if n_steps % 2 else n_steps + 1
    dt = T / n
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    p = _peizer_pratt(d2, n)
    p_bar = _peizer_pratt(d1, n)
    growth = np.exp((r - q) * dt)
    u = growth * p_bar / p
    d = (growth - p * u) / (1 - p)
    disc = np.exp(-r * dt)
    w = option_sign(option_type)
    j = np.arange(n + 1)
    spot = S * u ** (n - j) * d**j
    value = np.maximum(w * (spot - K), 0.0)
    for step in range(n - 1, -1, -1):
        value = disc * (p * value[:-1] + (1 - p) * value[1:])
        if american:
            spot = S * u ** (step - np.arange(step + 1)) * d ** np.arange(step + 1)
            value = np.maximum(value, w * (spot - K))
    return float(value[0])
