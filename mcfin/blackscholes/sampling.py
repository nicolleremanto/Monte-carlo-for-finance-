"""Simulation de la loi normale « from scratch ».

Tout Monte Carlo financier repose sur des gaussiennes. Ce module
implémente les quatre méthodes classiques d'un cours de simulation, à
partir d'uniformes seules, pour pouvoir les comparer et les tester :

1. **Inversion** : X = Φ⁻¹(U). Approximation rationnelle de Beasley-Springer
   (1977) au centre et développement de Moro (1995) dans les queues
   (Glasserman 2003, fig. 2.13). Erreur absolue ~ 3·10⁻⁹. Seule méthode
   *monotone* en U : indispensable en QMC (Sobol) et pour la stratification.

2. **Box-Muller (1958)** : pour U1, U2 i.i.d. uniformes,
   R = sqrt(-2 ln U1), Θ = 2π U2  =>  (R cos Θ, R sin Θ) i.i.d. N(0, 1).
   Preuve : changement de variables polaire, R² ~ Exp(1/2) indépendant de Θ.

3. **Méthode polaire de Marsaglia (1964)** : on évite sin/cos en tirant
   (V1, V2) uniforme dans le disque unité par rejet (acceptation π/4) :
   S = V1² + V2², X = V1 sqrt(-2 ln S / S).

4. **Rejet (von Neumann)** depuis une loi de Laplace g(x) = ½ e^{-|x|} :
   f/g <= M = sqrt(2e/π) ≈ 1.3155 ; taux d'acceptation 1/M ≈ 0.760.
   Illustration du principe acceptation-rejet : on accepte X ~ g avec
   probabilité f(X)/(M g(X)) = exp(-(|X| - 1)²/2).
"""
from __future__ import annotations

import numpy as np

__all__ = ["inverse_normal_bsm", "box_muller", "marsaglia_polar", "rejection_laplace",
           "REJECTION_CONSTANT"]

_A = (2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637)
_B = (-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833)
_C = (0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863,
      0.0038405729373609, 0.0003951896511919, 0.0000321767881768, 0.0000002888167364,
      0.0000003960315187)

#: constante M = sup f/g du rejet gaussien depuis Laplace
REJECTION_CONSTANT = float(np.sqrt(2 * np.e / np.pi))


def inverse_normal_bsm(u) -> np.ndarray:
    """Φ⁻¹(u) par Beasley-Springer-Moro, vectorisé, u ∈ (0, 1)."""
    u = np.asarray(u, dtype=float)
    y = u - 0.5
    out = np.empty_like(u)
    central = np.abs(y) < 0.42
    yc = y[central]
    r = yc * yc
    num = ((_A[3] * r + _A[2]) * r + _A[1]) * r + _A[0]
    den = (((_B[3] * r + _B[2]) * r + _B[1]) * r + _B[0]) * r + 1.0
    out[central] = yc * num / den
    tail = ~central
    ut = u[tail]
    rt = np.where(ut < 0.5, ut, 1.0 - ut)
    s = np.log(-np.log(rt))
    x = np.zeros_like(s)
    for c in reversed(_C):
        x = x * s + c
    out[tail] = np.where(ut < 0.5, -x, x)
    return out


def box_muller(n: int, rng: np.random.Generator) -> np.ndarray:
    m = (n + 1) // 2
    u1 = 1.0 - rng.random(m)          # dans (0, 1] : log fini
    u2 = rng.random(m)
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    return np.concatenate([r * np.cos(theta), r * np.sin(theta)])[:n]


def marsaglia_polar(n: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Renvoie (échantillon, taux d'acceptation empirique)."""
    out, drawn, got = [], 0, 0
    while got < n:
        m = int(1.3 * (n - got) / 2) + 16
        v = 2.0 * rng.random((m, 2)) - 1.0
        s = (v * v).sum(axis=1)
        ok = (s > 0) & (s < 1)
        drawn += m
        v, s = v[ok], s[ok]
        f = np.sqrt(-2.0 * np.log(s) / s)
        out.append((v * f[:, None]).ravel())
        got += 2 * ok.sum()
    return np.concatenate(out)[:n], got / (2 * drawn)


def rejection_laplace(n: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Acceptation-rejet depuis Laplace ; renvoie (échantillon, taux d'acceptation)."""
    out, drawn, got = [], 0, 0
    while got < n:
        m = int(1.4 * (n - got)) + 16
        e = -np.log(1.0 - rng.random(m))                 # Exp(1)
        x = np.where(rng.random(m) < 0.5, -e, e)         # Laplace(0, 1)
        accept = rng.random(m) <= np.exp(-0.5 * (np.abs(x) - 1.0) ** 2)
        drawn += m
        out.append(x[accept])
        got += accept.sum()
    return np.concatenate(out)[:n], got / drawn
