"""Modèle de Hull-White à un facteur (Vasicek étendu).

    dr_t = (θ(t) - a r_t) dt + σ dW_t        (mesure risque-neutre)

Décomposition r_t = x_t + φ(t) avec dx = -a x dt + σ dW, x_0 = 0 et
φ(t) = f(0,t) + σ²/(2a²)(1 - e^{-at})² : θ(t) est implicite et la courbe
initiale P(0, T) est reproduite exactement.

Prix zéro-coupon (affine en x) :
    P(t,T) = P(0,T)/P(0,t) · exp(-B x_t - σ²/(4a)(1-e^{-2at}) B² - B σ²/(2a²)(1-e^{-at})²),
    B = B(t,T) = (1 - e^{-a(T-t)})/a.

Simulation exacte : (x_t, I_t = ∫_0^t x_s ds) est un vecteur gaussien
dont on connaît la loi conditionnelle sur chaque pas (Glasserman §3.3) :
    x_{t+Δ} = e^{-aΔ} x_t + ε1,          Var ε1 = σ²/(2a)(1 - e^{-2aΔ})
    I_{t+Δ} = I_t + x_t (1-e^{-aΔ})/a + ε2,
    Var ε2 = σ²/a² [Δ - 2(1-e^{-aΔ})/a + (1-e^{-2aΔ})/(2a)],
    Cov(ε1, ε2) = σ²/(2a²)(1 - e^{-aΔ})²
Aucun biais de discrétisation, même avec des pas d'un an.

Swaptions européennes : décomposition de Jamshidian (1989) en somme
d'options sur zéro-coupons (le prix de l'obligation à coupons est monotone
en x).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from ..market.curves import Curve

__all__ = ["HullWhite", "HWPaths"]


@dataclass
class HWPaths:
    times: np.ndarray       # (nt,)
    x: np.ndarray           # (n, nt) facteur OU
    discount: np.ndarray    # (n, nt) déflateur D(0, t) = 1/B(t)

    def index(self, t: float) -> int:
        i = int(np.searchsorted(self.times, t - 1e-10))
        if i >= self.times.size or abs(self.times[i] - t) > 1e-9:
            raise ValueError(f"date {t} absente de la grille")
        return i


@dataclass
class HullWhite:
    a: float
    sigma: float
    curve: Curve

    # --- fonctions analytiques ------------------------------------------------
    def B(self, t, T):
        return (1.0 - np.exp(-self.a * (np.asarray(T) - np.asarray(t)))) / self.a

    def bond(self, t, T, x):
        """P(t, T) sachant x_t (vectorisé sur x et/ou T)."""
        a, s = self.a, self.sigma
        t = np.asarray(t, dtype=float)
        T = np.asarray(T, dtype=float)
        B = self.B(t, T)
        ratio = self.curve.df(T) / self.curve.df(t)
        conv = s**2 / (4 * a) * (1 - np.exp(-2 * a * t)) * B**2 \
            + B * s**2 / (2 * a**2) * (1 - np.exp(-a * t)) ** 2
        return ratio * np.exp(-B * x - conv)

    def _zcb_vol(self, T, S):
        """Volatilité de ln P(T,S) vue de 0 : σ_p."""
        a, s = self.a, self.sigma
        return s * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a)) * self.B(T, S)

    def zcb_option(self, T, S, K, option_type="put"):
        """Option européenne d'échéance T sur le zéro-coupon P(T, S), strike K."""
        P_T, P_S = self.curve.df(T), self.curve.df(S)
        sp = self._zcb_vol(T, S)
        h = np.log(P_S / (P_T * K)) / sp + 0.5 * sp
        if option_type == "call":
            return P_S * ndtr(h) - K * P_T * ndtr(h - sp)
        return K * P_T * ndtr(-h + sp) - P_S * ndtr(-h)

    def caplet(self, T, S, K):
        """Caplet sur le taux simple [T, S] = (1+τK) · put(T, S, 1/(1+τK))."""
        tau = S - T
        return (1 + tau * K) * self.zcb_option(T, S, 1 / (1 + tau * K), "put")

    def swaption(self, expiry: float, pay_times, strike: float, payer: bool = True) -> float:
        """Swaption européenne (Jamshidian). ``pay_times`` : dates de paiement
        fixe T_1..T_n (le swap démarre à ``expiry``)."""
        pay = np.asarray(pay_times, dtype=float)
        tau = np.diff(np.concatenate(([expiry], pay)))
        c = strike * tau
        c[-1] += 1.0

        def f(x):
            return float(np.sum(c * self.bond(expiry, pay, x)) - 1.0)
        x_star = brentq(f, -5.0, 5.0)
        K_i = self.bond(expiry, pay, x_star)
        # payer = put sur l'obligation à coupons de strike 1
        typ = "put" if payer else "call"
        return float(np.sum(c * self.zcb_option(expiry, pay, K_i, typ)))

    def swap_rate(self, t, pay_times, x, start=None):
        """Taux swap forward et annuité en t (start par défaut = t)."""
        pay = np.asarray(pay_times, dtype=float)
        start = t if start is None else start
        tau = np.diff(np.concatenate(([start], pay)))
        P = self.bond(t, pay[None, :], np.asarray(x)[:, None])
        annuity = P @ tau
        p0 = 1.0 if start == t else self.bond(t, start, x)
        return (p0 - P[:, -1]) / annuity, annuity

    # --- simulation --------------------------------------------------------------
    def simulate(self, times, n_paths: int, seed: int | None = 0,
                 antithetic: bool = False) -> HWPaths:
        a, s = self.a, self.sigma
        t = np.unique(np.concatenate(([0.0], np.asarray(times, dtype=float))))
        rng = np.random.default_rng(seed)
        n = n_paths
        if antithetic:
            half = rng.standard_normal((n // 2, t.size - 1, 2))
            z = np.concatenate([half, -half])
            n = z.shape[0]
        else:
            z = rng.standard_normal((n, t.size - 1, 2))
        x = np.zeros((n, t.size))
        I = np.zeros(n)
        disc = np.ones((n, t.size))
        for i, dt in enumerate(np.diff(t)):
            e = np.exp(-a * dt)
            v1 = s**2 / (2 * a) * (1 - e * e)
            v2 = s**2 / a**2 * (dt - 2 * (1 - e) / a + (1 - e * e) / (2 * a))
            c12 = s**2 / (2 * a**2) * (1 - e) ** 2
            l11 = np.sqrt(v1)
            l21 = c12 / l11
            l22 = np.sqrt(max(v2 - l21**2, 0.0))
            e1 = l11 * z[:, i, 0]
            e2 = l21 * z[:, i, 0] + l22 * z[:, i, 1]
            I = I + x[:, i] * (1 - e) / a + e2
            x[:, i + 1] = x[:, i] * e + e1
            T = t[i + 1]
            # ∫_0^T φ = -ln P(0,T) + σ²/(2a²)[T - 2(1-e^{-aT})/a + (1-e^{-2aT})/(2a)]
            conv = s**2 / (2 * a**2) * (T - 2 * (1 - np.exp(-a * T)) / a
                                        + (1 - np.exp(-2 * a * T)) / (2 * a))
            disc[:, i + 1] = self.curve.df(T) * np.exp(-I - conv)
        return HWPaths(times=t, x=x, discount=disc)
