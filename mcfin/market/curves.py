"""Courbes de taux (actualisation) et de dividendes.

Conventions : temps en années (ACT/365 implicite), taux continus.
Les courbes servent à la fois de courbe d'actualisation (r) et de courbe de
rendement du dividende / repo (q) pour les modèles actions : le forward
vaut F(t) = S0 · D_q(t) / D_r(t).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Curve", "FlatCurve", "InterpolatedCurve", "NelsonSiegelSvensson",
           "ShiftedCurve", "as_curve"]


class Curve(ABC):
    """Courbe de facteurs d'actualisation t -> D(0, t)."""

    @abstractmethod
    def df(self, t):
        ...

    def zero_rate(self, t):
        t = np.asarray(t, dtype=float)
        safe = np.where(t > 1e-12, t, 1e-12)
        return -np.log(self.df(safe)) / safe

    def inst_forward(self, t, h: float = 1e-5):
        """Taux forward instantané f(0, t) = -d ln D / dt (différences finies)."""
        t = np.asarray(t, dtype=float)
        lo = np.maximum(t - h, 0.0)
        hi = t + h
        return -(np.log(self.df(hi)) - np.log(self.df(lo))) / (hi - lo)

    def forward_rate(self, t1, t2):
        """Taux forward continu entre t1 et t2."""
        return np.log(self.df(t1) / self.df(t2)) / (np.asarray(t2) - np.asarray(t1))

    def simple_forward(self, t1, t2):
        """Taux forward simple (type Euribor, mono-courbe) entre t1 et t2."""
        return (self.df(t1) / self.df(t2) - 1.0) / (np.asarray(t2) - np.asarray(t1))

    def shift(self, h: float) -> "Curve":
        """Choc parallèle de ``h`` sur les taux zéro continus."""
        return ShiftedCurve(self, h)


class FlatCurve(Curve):
    def __init__(self, rate: float):
        self.rate = float(rate)

    def df(self, t):
        return np.exp(-self.rate * np.asarray(t, dtype=float))

    def zero_rate(self, t):
        return np.full_like(np.asarray(t, dtype=float), self.rate)

    def inst_forward(self, t, h: float = 1e-5):
        return np.full_like(np.asarray(t, dtype=float), self.rate)

    def shift(self, h: float) -> "FlatCurve":
        return FlatCurve(self.rate + h)

    def __repr__(self) -> str:
        return f"FlatCurve({self.rate})"


class InterpolatedCurve(Curve):
    """Interpolation log-linéaire des facteurs d'actualisation (forwards
    instantanés constants par morceaux : « raw interpolation »)."""

    def __init__(self, times, zero_rates):
        t = np.asarray(times, dtype=float)
        z = np.asarray(zero_rates, dtype=float)
        if np.any(np.diff(t) <= 0) or t[0] <= 0:
            raise ValueError("piliers strictement croissants et > 0")
        self.times = np.concatenate(([0.0], t))
        self.log_df = np.concatenate(([0.0], -z * t))

    def df(self, t):
        t = np.asarray(t, dtype=float)
        # extrapolation plate du dernier forward
        slope = (self.log_df[-1] - self.log_df[-2]) / (self.times[-1] - self.times[-2])
        inside = np.interp(t, self.times, self.log_df)
        beyond = self.log_df[-1] + slope * (t - self.times[-1])
        return np.exp(np.where(t > self.times[-1], beyond, inside))


class NelsonSiegelSvensson(Curve):
    """Courbe Nelson-Siegel-Svensson (utilisée par les banques centrales).

    y(t) = β0 + β1 (1-e^{-t/τ1})/(t/τ1) + β2 [(1-e^{-t/τ1})/(t/τ1) - e^{-t/τ1}]
         + β3 [(1-e^{-t/τ2})/(t/τ2) - e^{-t/τ2}]
    Le forward instantané est analytique, ce qui donne un θ(t) lisse en
    Hull-White.
    """

    def __init__(self, beta0, beta1, beta2, beta3=0.0, tau1=2.0, tau2=5.0):
        self.b = (beta0, beta1, beta2, beta3)
        self.tau1, self.tau2 = tau1, tau2

    @staticmethod
    def _g(x):
        x = np.asarray(x, dtype=float)
        small = np.abs(x) < 1e-8
        xs = np.where(small, 1.0, x)
        return np.where(small, 1.0 - x / 2, (1.0 - np.exp(-xs)) / xs)

    def zero_rate(self, t):
        t = np.asarray(t, dtype=float)
        b0, b1, b2, b3 = self.b
        x1, x2 = t / self.tau1, t / self.tau2
        g1, g2 = self._g(x1), self._g(x2)
        return b0 + b1 * g1 + b2 * (g1 - np.exp(-x1)) + b3 * (g2 - np.exp(-x2))

    def df(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero_rate(t) * t)

    def inst_forward(self, t, h: float = 1e-5):
        t = np.asarray(t, dtype=float)
        b0, b1, b2, b3 = self.b
        x1, x2 = t / self.tau1, t / self.tau2
        return b0 + b1 * np.exp(-x1) + b2 * x1 * np.exp(-x1) + b3 * x2 * np.exp(-x2)


class ShiftedCurve(Curve):
    def __init__(self, base: Curve, h: float):
        self.base, self.h = base, float(h)

    def df(self, t):
        t = np.asarray(t, dtype=float)
        return self.base.df(t) * np.exp(-self.h * t)

    def inst_forward(self, t, h: float = 1e-5):
        return self.base.inst_forward(t, h) + self.h


def as_curve(x) -> Curve:
    """float -> FlatCurve ; Curve -> inchangée."""
    if isinstance(x, Curve):
        return x
    return FlatCurve(float(x))
