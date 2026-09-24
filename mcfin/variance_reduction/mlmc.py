"""Multilevel Monte Carlo (Giles 2008, Operations Research).

Idée : décomposer l'espérance au pas le plus fin en somme télescopique

    E[P_L] = E[P_0] + Σ_{l=1}^{L} E[P_l - P_{l-1}],

où P_l est le payoff discrétisé au pas h_l = T/M^l. Les corrections
P_l - P_{l-1} sont simulées avec le MÊME brownien (incréments fins sommés
par paquets de M pour le niveau grossier) : leur variance V_l décroît en
h_l^β. L'allocation optimale N_l ∝ sqrt(V_l / C_l) donne, pour une erreur
quadratique ε² :

    coût = O(ε^{-2})            si β > γ   (Milstein, payoffs lipschitziens)
    coût = O(ε^{-2} (ln ε)²)    si β = γ   (Euler)
contre O(ε^{-3}) pour un Monte Carlo standard avec schéma d'Euler.

L'algorithme adaptatif ajoute des niveaux jusqu'à ce que le biais estimé
|E[P_L - P_{L-1}]|/(M^α - 1) soit inférieur à ε/sqrt(2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..analytics.black_scholes import option_sign

__all__ = ["MLMCResult", "mlmc", "gbm_level_sampler", "heston_level_sampler"]


@dataclass
class MLMCResult:
    price: float
    n_levels: int
    n_samples: list = field(default_factory=list)
    means: list = field(default_factory=list)
    variances: list = field(default_factory=list)
    cost: float = 0.0
    std_mc_cost: float = 0.0  # coût estimé d'un MC standard au même ε

    def __repr__(self):
        return (f"MLMCResult(price={self.price:.6f}, L={self.n_levels}, N_l={self.n_samples}, "
                f"cost={self.cost:.3g}, gain vs MC={self.std_mc_cost / max(self.cost, 1):.1f}x)")


def mlmc(level_sampler: Callable, eps: float, M: int = 2, L_min: int = 2, L_max: int = 12,
         n_initial: int = 10_000, seed: int = 0, alpha: float | None = None) -> MLMCResult:
    """Algorithme adaptatif de Giles.

    ``level_sampler(l, n, rng) -> (Σ(P_l - P_{l-1}), Σ(P_l - P_{l-1})², coût)``
    (au niveau 0 : P_0 seul).
    """
    rng = np.random.default_rng(seed)
    sums = [np.zeros(3) for _ in range(L_max + 1)]
    N = np.zeros(L_max + 1, dtype=int)
    dN = np.zeros(L_max + 1, dtype=int)
    L = L_min
    dN[: L + 1] = n_initial
    cost_per = np.zeros(L_max + 1)
    while dN[: L + 1].sum() > 0:
        for l in range(L + 1):
            if dN[l] > 0:
                s1, s2, c = level_sampler(l, int(dN[l]), rng)
                sums[l] += (s1, s2, c)
                N[l] += dN[l]
        mean = np.array([sums[l][0] / max(N[l], 1) for l in range(L + 1)])
        var = np.array([max(sums[l][1] / max(N[l], 1) - mean[l] ** 2, 1e-300)
                        for l in range(L + 1)])
        cost_per[: L + 1] = [sums[l][2] / max(N[l], 1) for l in range(L + 1)]
        # allocation optimale (Lagrangien) : N_l = 2 ε^-2 sqrt(V_l/C_l) Σ sqrt(V_k C_k)
        Ns = np.ceil(2 * eps**-2 * np.sqrt(var / cost_per[: L + 1])
                     * np.sum(np.sqrt(var * cost_per[: L + 1]))).astype(int)
        dN[: L + 1] = np.maximum(0, Ns - N[: L + 1])
        if dN[: L + 1].sum() == 0:
            # test de convergence du biais sur les derniers niveaux
            a = alpha
            if a is None:
                x = np.arange(1, L + 1)
                y = np.log2(np.abs(mean[1:]) + 1e-300)
                a = max(0.5, -np.polyfit(x, y, 1)[0]) if L >= 2 else 1.0
            rem = max(abs(mean[L]), abs(mean[L - 1]) / M**a) / (M**a - 1)
            if rem > eps / np.sqrt(2):
                if L == L_max:
                    break
                L += 1
                dN[L] = n_initial
    price = float(sum(sums[l][0] / N[l] for l in range(L + 1)))
    cost = float(sum(N[l] * cost_per[l] for l in range(L + 1)))
    v_pl = float(sums[0][1] / N[0] - (sums[0][0] / N[0]) ** 2)
    std_cost = 2 * eps**-2 * v_pl * cost_per[L]
    return MLMCResult(price, L, N[: L + 1].tolist(), mean.tolist(), var.tolist(), cost, std_cost)


def gbm_level_sampler(s0, K, T, r, sigma, option_type="call", scheme="milstein",
                      M: int = 2, payoff: str = "european", base_steps: int = 1):
    """Échantillonneur de niveaux pour un GBM discrétisé (Euler ou Milstein).

    payoff : "european" ou "asian" (moyenne arithmétique continue approchée
    par la règle des trapèzes). Le niveau l utilise base_steps · M^l pas.
    """
    w = option_sign(option_type)

    def step(s, dw, h):
        s_new = s + r * s * h + sigma * s * dw
        if scheme == "milstein":
            s_new += 0.5 * sigma**2 * s * (dw * dw - h)
        return s_new

    def pay(s_T, avg):
        x = s_T if payoff == "european" else avg
        return np.exp(-r * T) * np.maximum(w * (x - K), 0.0)

    def sampler(l, n, rng):
        s1 = s2 = 0.0
        chunk = 50_000
        for start in range(0, n, chunk):
            m = min(chunk, n - start)
            nf = base_steps * M**l
            hf = T / nf
            dw = np.sqrt(hf) * rng.standard_normal((m, nf))
            sf = np.full(m, float(s0))
            af = 0.5 * sf * hf
            for i in range(nf):
                sf = step(sf, dw[:, i], hf)
                af += sf * hf
            af -= 0.5 * sf * hf
            pf = pay(sf, af / T)
            if l == 0:
                d = pf
            else:
                nc = nf // M
                hc = T / nc
                dwc = dw.reshape(m, nc, M).sum(axis=2)
                sc = np.full(m, float(s0))
                ac = 0.5 * sc * hc
                for i in range(nc):
                    sc = step(sc, dwc[:, i], hc)
                    ac += sc * hc
                ac -= 0.5 * sc * hc
                d = pf - pay(sc, ac / T)
            s1 += d.sum()
            s2 += (d * d).sum()
        return s1, s2, float(n * base_steps * M**l)
    return sampler


def heston_level_sampler(s0, K, T, r, v0, kappa, theta, xi, rho, option_type="call",
                         M: int = 2, base_steps: int = 8):
    """Niveaux Heston (Euler full truncation, log-spot) — illustre le MLMC
    avec un schéma d'ordre faible 1 / fort 1/2 (β ≈ 1). Niveau de base à
    ``base_steps`` pas : un Euler à 1-2 pas est trop grossier pour Heston
    (variance des premières corrections énorme, aucun gain)."""
    w = option_sign(option_type)
    rc = np.sqrt(1 - rho**2)

    def run(z1, z2, h):
        x = np.zeros(z1.shape[0])
        v = np.full(z1.shape[0], float(v0))
        for i in range(z1.shape[1]):
            vp = np.maximum(v, 0.0)
            sq = np.sqrt(vp)
            x += (r - 0.5 * vp) * h + sq * (rho * z1[:, i] + rc * z2[:, i])
            v += kappa * (theta - vp) * h + xi * sq * z1[:, i]
        return np.exp(-r * T) * np.maximum(w * (s0 * np.exp(x) - K), 0.0)

    def sampler(l, n, rng):
        s1 = s2 = 0.0
        for start in range(0, n, 50_000):
            m = min(50_000, n - start)
            nf = base_steps * M**l
            hf = T / nf
            dw1 = np.sqrt(hf) * rng.standard_normal((m, nf))
            dw2 = np.sqrt(hf) * rng.standard_normal((m, nf))
            pf = run(dw1, dw2, hf)
            if l == 0:
                d = pf
            else:
                nc = nf // M
                d = pf - run(dw1.reshape(m, nc, M).sum(2), dw2.reshape(m, nc, M).sum(2), T / nc)
            s1 += d.sum()
            s2 += (d * d).sum()
        return s1, s2, float(n * base_steps * M**l)
    return sampler
