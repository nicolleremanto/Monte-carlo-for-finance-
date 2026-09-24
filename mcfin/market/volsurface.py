"""Nappes de volatilité implicite et volatilité locale de Dupire.

Notations : y = ln(K / F_T) (log-moneyness forward), w(y, T) = σ_imp²(y, T)·T
(variance totale implicite).

Volatilité locale de Dupire exprimée en variance totale (Gatheral 2006, éq. 1.10) :

    σ_loc²(y, T) = ∂_T w / [ 1 - (y/w) ∂_y w + ¼(-¼ - 1/w + y²/w²)(∂_y w)² + ½ ∂²_y w ]

Le dénominateur est la densité (à une normalisation près) : il est positif
si et seulement si la nappe est sans arbitrage papillon ; le numérateur est
positif ssi elle est sans arbitrage calendaire.

SSVI (Gatheral & Jacquier 2014, « Arbitrage-free SVI volatility surfaces ») :

    w(y, θ_T) = θ_T/2 · [1 + ρ φ(θ_T) y + sqrt((φ(θ_T) y + ρ)² + 1 - ρ²)]

θ_T = variance totale ATM ; φ en loi puissance φ(θ) = η / (θ^γ (1+θ)^{1-γ}).
Conditions suffisantes d'absence d'arbitrage statique : θ_T croissante,
0 < γ <= 1/2 et η(1 + |ρ|) <= 2.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["ImpliedVolSurface", "FlatVolSurface", "SSVISurface"]


class ImpliedVolSurface(ABC):
    @abstractmethod
    def total_variance(self, y, T):
        ...

    def implied_vol(self, y, T):
        T = np.asarray(T, dtype=float)
        return np.sqrt(self.total_variance(y, T) / T)

    # dérivées par différences finies (surchargées analytiquement si possible)
    def w_y(self, y, T, h=1e-4):
        return (self.total_variance(y + h, T) - self.total_variance(y - h, T)) / (2 * h)

    def w_yy(self, y, T, h=1e-4):
        return (self.total_variance(y + h, T) - 2 * self.total_variance(y, T)
                + self.total_variance(y - h, T)) / h**2

    def w_T(self, y, T, h=1e-5):
        T = np.asarray(T, dtype=float)
        return (self.total_variance(y, T + h) - self.total_variance(y, np.maximum(T - h, 1e-8))) \
            / (T + h - np.maximum(T - h, 1e-8))

    def local_variance(self, y, T, floor: float = 1e-8):
        """Variance locale de Dupire au point (y = ln(K/F_T), T)."""
        y = np.asarray(y, dtype=float)
        w = np.maximum(self.total_variance(y, T), 1e-12)
        wy, wyy, wt = self.w_y(y, T), self.w_yy(y, T), self.w_T(y, T)
        den = (1 - y / w * wy + 0.25 * (-0.25 - 1 / w + y * y / (w * w)) * wy * wy + 0.5 * wyy)
        return np.maximum(wt, floor) / np.maximum(den, 1e-4)


class FlatVolSurface(ImpliedVolSurface):
    def __init__(self, vol: float):
        self.vol = float(vol)

    def total_variance(self, y, T):
        return self.vol**2 * np.asarray(T, dtype=float) + 0.0 * np.asarray(y, dtype=float)

    def local_variance(self, y, T, floor=1e-8):
        return np.full(np.broadcast(np.asarray(y), np.asarray(T)).shape, self.vol**2)


class SSVISurface(ImpliedVolSurface):
    """SSVI à terme de variance ATM « mean-reverting » :

        θ_T = σ_∞² T + (σ_0² - σ_∞²)(1 - e^{-λT})/λ    (croissante si σ_0, σ_∞ > 0)
    """

    def __init__(self, sigma0: float = 0.2, sigma_inf: float = 0.22, lam: float = 1.0,
                 rho: float = -0.6, eta: float = 1.0, gamma: float = 0.4):
        if eta * (1 + abs(rho)) > 2 + 1e-12 or not 0 < gamma <= 0.5:
            raise ValueError("paramètres SSVI hors du domaine sans arbitrage")
        self.sigma0, self.sigma_inf, self.lam = sigma0, sigma_inf, lam
        self.rho, self.eta, self.gamma = rho, eta, gamma

    # terme ATM
    def theta(self, T):
        T = np.asarray(T, dtype=float)
        return self.sigma_inf**2 * T + (self.sigma0**2 - self.sigma_inf**2) \
            * (1 - np.exp(-self.lam * T)) / self.lam

    def theta_prime(self, T):
        T = np.asarray(T, dtype=float)
        return self.sigma_inf**2 + (self.sigma0**2 - self.sigma_inf**2) * np.exp(-self.lam * T)

    def phi(self, th):
        return self.eta / (th**self.gamma * (1 + th) ** (1 - self.gamma))

    def phi_prime(self, th):
        g = self.gamma
        return -self.phi(th) * (g / th + (1 - g) / (1 + th))

    # variance totale et dérivées analytiques
    def _parts(self, y, T):
        y = np.asarray(y, dtype=float)
        th = np.maximum(self.theta(T), 1e-12)
        ph = self.phi(th)
        a = ph * y + self.rho
        R = np.sqrt(a * a + 1 - self.rho**2)
        return y, th, ph, a, R

    def total_variance(self, y, T):
        y, th, ph, a, R = self._parts(y, T)
        return 0.5 * th * (1 + self.rho * ph * y + R)

    def w_y(self, y, T, h=None):
        y, th, ph, a, R = self._parts(y, T)
        return 0.5 * th * ph * (self.rho + a / R)

    def w_yy(self, y, T, h=None):
        y, th, ph, a, R = self._parts(y, T)
        return 0.5 * th * ph * ph * (1 - self.rho**2) / R**3

    def w_T(self, y, T, h=None):
        y, th, ph, a, R = self._parts(y, T)
        w = 0.5 * th * (1 + self.rho * ph * y + R)
        dphi = self.phi_prime(th)
        dw_dth = w / th + 0.5 * th * y * dphi * (self.rho + a / R)
        return dw_dth * self.theta_prime(T)
