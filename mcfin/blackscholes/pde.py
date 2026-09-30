"""Équation de Black-Scholes par différences finies (Crank-Nicolson).

Par Feynman-Kac, V(t, S) = E^Q[e^{-r(T-t)} Φ(S_T) | S_t = S] résout

    ∂_t V + ½σ²S² ∂_SS V + (r - q) S ∂_S V - r V = 0,   V(T, S) = Φ(S).

En log-spot x = ln S et temps restant τ = T - t, les coefficients deviennent
constants (équation de la chaleur avec transport) :

    ∂_τ V = ½σ² ∂_xx V + ν ∂_x V - r V,   ν = r - q - ½σ².

Discrétisation
--------------
* grille uniforme en x, centrée sur ln S0, de demi-largeur n_std · σ sqrt(T) ;
* θ-schéma : θ = ½ (Crank-Nicolson, ordre 2 en temps et en espace,
  inconditionnellement stable en norme L² mais non monotone) ;
* payoff moyenné sur chaque cellule (erreur indépendante de la position du
  strike dans la grille) ;
* lissage de Rannacher (1984) : les premiers pas sont faits en Euler
  implicite (demi-pas) pour amortir les oscillations créées par le coin du
  payoff, sans perdre l'ordre 2 ;
* conditions de Dirichlet asymptotiques aux bords ;
* exercice américain : problème de complémentarité linéaire
  min(-∂_τV + LV, V - Φ) = 0 résolu à chaque pas par l'algorithme de
  Brennan & Schwartz (1977) — une élimination de Gauss tridiagonale
  « projetée », exacte lorsque la région d'exercice est connexe et du bon
  côté de la grille (put : S petit ; call avec dividende : S grand).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from ..analytics.black_scholes import option_sign

__all__ = ["PDEResult", "bs_pde_price"]


@dataclass
class PDEResult:
    price: float
    delta: float
    gamma: float
    theta: float
    spots: np.ndarray
    values: np.ndarray
    exercise_boundary: np.ndarray | None = None  # frontière d'exercice S*(τ) (américain)


def _thomas(a, b, c, d):
    """Système tridiagonal a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i."""
    n = d.size
    cp, dp = np.empty(n), np.empty(n)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _brennan_schwartz(a, b, c, d, g, exercise_low: bool):
    """Gauss projeté : la substitution arrière part de la région d'exercice
    et impose x_i >= g_i au fil de l'eau."""
    if exercise_low:  # on renverse le système pour substituer de i = 0 vers n-1
        return _brennan_schwartz(c[::-1], b[::-1], a[::-1], d[::-1], g[::-1], False)[::-1]
    n = d.size
    cp, dp = np.empty(n), np.empty(n)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = max(dp[-1], g[-1])
    for i in range(n - 2, -1, -1):
        x[i] = max(dp[i] - cp[i] * x[i + 1], g[i])
    return x


def _cell_average_payoff(x: np.ndarray, dx: float, K: float, w: float) -> np.ndarray:
    """Moyenne exacte de (ω(e^y - K))^+ sur [x_i - dx/2, x_i + dx/2].

    Lisser la donnée initiale (Kreiss, Thomée & Widlund 1970) rend l'erreur
    indépendante de la position du strike dans la grille et restaure la
    convergence d'ordre 2 de Crank-Nicolson malgré le coin du payoff.
    """
    lo, hi = x - dx / 2, x + dx / 2
    k = np.log(K)
    if w > 0:
        a = np.maximum(lo, k)
        integral = np.where(hi > k, (np.exp(hi) - np.exp(a)) - K * (hi - a), 0.0)
    else:
        b = np.minimum(hi, k)
        integral = np.where(lo < k, K * (b - lo) - (np.exp(b) - np.exp(lo)), 0.0)
    return integral / dx


def bs_pde_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    american: bool = False,
    n_space: int = 400,
    n_time: int = 200,
    n_std: float = 6.0,
    theta: float = 0.5,
    rannacher_steps: int = 4,
    smooth_payoff: bool = True,
) -> PDEResult:
    """Prix, delta, gamma et theta par Crank-Nicolson (+ Rannacher)."""
    w = option_sign(option_type)
    half = n_std * sigma * np.sqrt(T)
    x0 = np.log(S0)
    x = np.linspace(x0 - half, x0 + half, n_space + 1)
    dx = x[1] - x[0]
    S = np.exp(x)
    payoff = np.maximum(w * (S - K), 0.0)
    V0 = _cell_average_payoff(x, dx, K, w) if smooth_payoff else payoff
    nu = r - q - 0.5 * sigma**2
    s2 = 0.5 * sigma**2
    # opérateur L sur les nœuds intérieurs : l V_{i-1} + m V_i + u V_{i+1}
    lo = s2 / dx**2 - nu / (2 * dx)
    mid = -2 * s2 / dx**2 - r
    up = s2 / dx**2 + nu / (2 * dx)

    def boundary(tau):
        if w > 0:
            low = 0.0
            high = S[-1] * np.exp(-q * tau) - K * np.exp(-r * tau)
            if american:
                high = max(high, S[-1] - K)
        else:
            low = K * np.exp(-r * tau) - S[0] * np.exp(-q * tau)
            if american:
                low = max(low, K - S[0])
            high = 0.0
        return low, high

    # pas de temps : Rannacher (demi-pas implicites) puis Crank-Nicolson
    dt_full = T / n_time
    steps = [(dt_full / 2, 1.0)] * (2 * rannacher_steps) + [(dt_full, theta)] * (n_time - rannacher_steps)
    V = V0.copy()
    tau = 0.0
    boundary_path = []
    n_in = n_space - 1
    for dt, th in steps:
        tau_new = tau + dt
        # (I - θ dt L) V^{n+1} = (I + (1-θ) dt L) V^n
        rhs = V[1:-1] + (1 - th) * dt * (lo * V[:-2] + mid * V[1:-1] + up * V[2:])
        b_low, b_high = boundary(tau_new)
        a = np.full(n_in, -th * dt * lo)
        b = np.full(n_in, 1 - th * dt * mid)
        c = np.full(n_in, -th * dt * up)
        rhs[0] -= a[0] * b_low
        rhs[-1] -= c[-1] * b_high
        a[0] = c[-1] = 0.0
        if american:
            inner = _brennan_schwartz(a, b, c, rhs, payoff[1:-1], exercise_low=w < 0)
            ex = np.nonzero((inner <= payoff[1:-1] + 1e-12) & (payoff[1:-1] > 0))[0]
            if ex.size:
                idx = ex.max() if w < 0 else ex.min()
                boundary_path.append((tau_new, S[1 + idx]))
        else:
            inner = _thomas(a, b, c, rhs)
        V = np.concatenate(([b_low], inner, [b_high]))
        tau = tau_new
    # dérivées : dans la variable x puis retour en S (dV/dS = V_x / S, ...)
    spline = CubicSpline(x, V)
    price = float(spline(x0))
    vx, vxx = float(spline(x0, 1)), float(spline(x0, 2))
    delta = vx / S0
    gamma = (vxx - vx) / S0**2
    # theta via l'EDP elle-même : ∂_t V = -(½σ²S²Γ + (r-q)SΔ - rV)
    theta_t = -(0.5 * sigma**2 * S0**2 * gamma + (r - q) * S0 * delta - r * price)
    if american:  # l'EDP n'est vraie que dans la région de continuation
        theta_t = float("nan") if price <= max(w * (S0 - K), 0) + 1e-10 else theta_t
    eb = np.array(boundary_path) if boundary_path else None
    return PDEResult(price, delta, gamma, theta_t, S, V, eb)
