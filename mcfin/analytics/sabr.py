"""Formule asymptotique de Hagan et al. (2002), « Managing Smile Risk ».

    dF = α F^β dW1,   dα = ν α dW2,   d<W1, W2> = ρ dt   (mesure forward)

Volatilité implicite lognormale (Black) :

    σ_B(K) = α / [(FK)^{(1-β)/2} (1 + (1-β)²/24 ln²(F/K) + (1-β)⁴/1920 ln⁴(F/K))]
             · z / x(z)
             · [1 + ((1-β)²/24 · α²/(FK)^{1-β} + ρβνα/(4 (FK)^{(1-β)/2})
                    + (2-3ρ²)/24 ν²) T]
    z = ν/α (FK)^{(1-β)/2} ln(F/K),  x(z) = ln[(sqrt(1-2ρz+z²) + z - ρ)/(1-ρ)]
"""
from __future__ import annotations

import numpy as np

__all__ = ["sabr_hagan_vol"]


def sabr_hagan_vol(F, K, T, alpha, beta, rho, nu):
    F = float(F)
    K = np.asarray(K, dtype=float)
    lfk = np.log(F / K)
    fk = (F * K) ** ((1 - beta) / 2)
    denom = fk * (1 + (1 - beta) ** 2 / 24 * lfk**2 + (1 - beta) ** 4 / 1920 * lfk**4)
    z = nu / alpha * fk * lfk
    with np.errstate(divide="ignore", invalid="ignore"):
        xz = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
        zx = np.where(np.abs(z) < 1e-8, 1.0 - 0.5 * rho * z, z / xz)
    corr = 1 + ((1 - beta) ** 2 / 24 * alpha**2 / fk**2
                + rho * beta * nu * alpha / (4 * fk)
                + (2 - 3 * rho**2) / 24 * nu**2) * T
    return alpha / denom * zx * corr
