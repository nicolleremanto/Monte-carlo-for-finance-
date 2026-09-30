"""LIBOR/Euribor Market Model (Brace-Gatarek-Musiela 1997, Jamshidian 1997).

Forwards L_i(t) sur [T_i, T_{i+1}] (i = 0..N-1, T_0 = 0), lognormaux sous
leur mesure forward T_{i+1} (=> formule de Black pour les caplets). Sous la
mesure spot (numéraire = compte courant discret
B(t) = P(t, T_{η(t)}) Π_{j<η(t)} (1 + τ_j L_j(T_j))) :

    dL_i / L_i = μ_i(t) dt + σ_i(t) dW_i,
    μ_i(t) = σ_i(t) Σ_{j=η(t)}^{i} τ_j ρ_ij σ_j(t) L_j / (1 + τ_j L_j)

où η(t) est l'indice de la première date de tenor strictement après t.

Discrétisation : log-Euler avec prédicteur-correcteur sur la dérive
(Hunter, Jäckel & Joshi 2001) ; covariance intégrée exacte sur chaque pas
C_ij = ρ_ij ∫ σ_i σ_j dt, réduite à k facteurs par ACP (renormalisée pour
conserver les variances des caplets).

Volatilité « abcd » (Rebonato) : σ_i(t) = [a + b(T_i - t)] e^{-c(T_i - t)} + d,
corrélation exponentielle ρ_ij = exp(-β |T_i - T_j|).

Contexte post-LIBOR : l'Euribor subsiste (et donc le LMM classique) ; pour
les taux RFR composés (€STR, SOFR), l'extension naturelle est le « Forward
Market Model » de Lyashenko & Mercurio (2019).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.integrate import quad

from ..analytics.black_scholes import black_price
from ..market.curves import Curve

__all__ = ["LMMPaths", "LiborMarketModel"]


@dataclass
class LMMPaths:
    tenor: np.ndarray  # T_0..T_N
    forwards: np.ndarray  # (n, N+1, N) : forwards[:, k, j] = L_j(T_k) (NaN si j < k)
    numeraire: np.ndarray  # (n, N+1) : B(T_k)

    def bond(self, k: int, m: int) -> np.ndarray:
        """P(T_k, T_m) pour m >= k (produit des facteurs 1/(1+τL))."""
        tau = np.diff(self.tenor)
        out = np.ones(self.forwards.shape[0])
        for j in range(k, m):
            out = out / (1 + tau[j] * self.forwards[:, k, j])
        return out

    def swap(self, k: int, end: int):
        """(taux swap, annuité) en T_k du swap T_k -> T_end."""
        tau = np.diff(self.tenor)
        dfs = []
        p = np.ones(self.forwards.shape[0])
        for j in range(k, end):
            p = p / (1 + tau[j] * self.forwards[:, k, j])
            dfs.append(p)
        dfs = np.column_stack(dfs)
        annuity = dfs @ tau[k:end]
        return (1 - dfs[:, -1]) / annuity, annuity


@dataclass
class LiborMarketModel:
    tenor: np.ndarray
    curve: Curve
    a: float = 0.05
    b: float = 0.09
    c: float = 0.44
    d: float = 0.11
    beta: float = 0.1
    n_factors: int = 3
    steps_per_period: int = 4

    def __post_init__(self):
        self.tenor = np.asarray(self.tenor, dtype=float)
        if self.tenor[0] != 0.0:
            raise ValueError("tenor[0] doit valoir 0")
        self.tau = np.diff(self.tenor)
        self.N = self.tau.size
        self.L0 = self.curve.simple_forward(self.tenor[:-1], self.tenor[1:])
        T = self.tenor[:-1]
        self.rho = np.exp(-self.beta * np.abs(T[:, None] - T[None, :]))

    # --- volatilités -------------------------------------------------------------
    def vol(self, i, t):
        """σ_i(t) (abcd), vectorisé en i et t."""
        x = np.maximum(self.tenor[np.asarray(i)] - t, 0.0)
        return (self.a + self.b * x) * np.exp(-self.c * x) + self.d

    def caplet_black_vol(self, i: int) -> float:
        """σ_Black² T_i = ∫_0^{T_i} σ_i(t)² dt."""
        Ti = self.tenor[i]
        if Ti <= 0:
            return 0.0
        return float(np.sqrt(quad(lambda t: self.vol(i, t) ** 2, 0, Ti)[0] / Ti))

    def caplet_price(self, i: int, K: float) -> float:
        """Caplet de Black sur L_i (fixing T_i, paiement T_{i+1})."""
        return float(
            self.tau[i]
            * black_price(
                self.L0[i], K, self.tenor[i], self.curve.df(self.tenor[i + 1]), self.caplet_black_vol(i)
            )
        )

    def swaption_rebonato(self, s: int, e: int, K: float) -> float:
        """Approximation de Rebonato (poids gelés) pour la swaption payeuse
        d'expiry T_s sur le swap T_s -> T_e."""
        idx = np.arange(s, e)
        P = self.curve.df(self.tenor[idx + 1])
        A = np.sum(self.tau[idx] * P)
        S = (self.curve.df(self.tenor[s]) - self.curve.df(self.tenor[e])) / A
        # poids w_i = τ_i P(0,T_{i+1}) / A (approx. usuelle, dérivée de S par rapport à L_i ~ w_i)
        w = self.tau[idx] * P / A
        Ts = self.tenor[s]
        cov = np.zeros((idx.size, idx.size))
        for a_, i in enumerate(idx):
            for b_, j in enumerate(idx):
                if b_ < a_:
                    cov[a_, b_] = cov[b_, a_]
                    continue
                cov[a_, b_] = self.rho[i, j] * quad(lambda t: self.vol(i, t) * self.vol(j, t), 0, Ts)[0]
        L = self.L0[idx]
        var = (w * L) @ cov @ (w * L) / S**2
        return float(A * black_price(S, K, Ts, 1.0, np.sqrt(var / Ts)))

    # --- simulation ---------------------------------------------------------------
    def _step_cov(self, t0: float, t1: float, alive: np.ndarray) -> np.ndarray:
        x, wts = np.polynomial.legendre.leggauss(5)
        ts = 0.5 * (t1 - t0) * x + 0.5 * (t0 + t1)
        sig = np.array([self.vol(alive, t) for t in ts])  # (5, m)
        integ = np.einsum("q,qi,qj->ij", 0.5 * (t1 - t0) * wts, sig, sig)
        return self.rho[np.ix_(alive, alive)] * integ

    def _loadings(self, C: np.ndarray) -> np.ndarray:
        lam, vec = np.linalg.eigh(C)
        order = np.argsort(lam)[::-1][: self.n_factors]
        A = vec[:, order] * np.sqrt(np.clip(lam[order], 0, None))
        norm = np.sqrt(np.maximum((A * A).sum(axis=1), 1e-300))
        return A * (np.sqrt(np.diag(C)) / norm)[:, None]

    def _drift(self, logL: np.ndarray, alive: np.ndarray, C: np.ndarray) -> np.ndarray:
        L = np.exp(logL)
        tl = self.tau[alive] * L
        g = tl / (1 + tl)  # (n, m)
        # μ_i Δ = Σ_{j<=i, j vivant} C_ij g_j  (produit triangulaire inférieur)
        return g @ np.tril(C).T - 0.5 * np.diag(C)[None, :]

    def simulate(self, n_paths: int, seed: int | None = 0, antithetic: bool = False) -> LMMPaths:
        rng = np.random.default_rng(seed)
        N = self.N
        n = n_paths
        fwd = np.full((n, N + 1, N), np.nan)
        logL = np.tile(np.log(self.L0), (n, 1))
        fwd[:, 0, :] = self.L0
        numer = np.ones((n, N + 1))
        for k in range(N):
            # sur [T_k, T_{k+1}) : L_k est fixé en T_k ; vivants j >= k+1
            numer[:, k + 1] = numer[:, k] * (1 + self.tau[k] * np.exp(logL[:, k]))
            alive = np.arange(k + 1, N)
            if alive.size:
                ts = np.linspace(self.tenor[k], self.tenor[k + 1], self.steps_per_period + 1)
                for t0, t1 in pairwise(ts):
                    C = self._step_cov(t0, t1, alive)
                    A = self._loadings(C)
                    if antithetic:
                        h = rng.standard_normal((n // 2, A.shape[1]))
                        z = np.concatenate([h, -h])
                    else:
                        z = rng.standard_normal((n, A.shape[1]))
                    dW = z @ A.T
                    x = logL[:, alive]
                    mu0 = self._drift(x, alive, C)
                    pred = x + mu0 + dW
                    mu1 = self._drift(pred, alive, C)
                    logL[:, alive] = x + 0.5 * (mu0 + mu1) + dW
            fwd[:, k + 1, k + 1 :] = np.exp(logL[:, k + 1 :])
        return LMMPaths(tenor=self.tenor, forwards=fwd, numeraire=numer)
