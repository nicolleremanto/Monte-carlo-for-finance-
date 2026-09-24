"""Swaptions bermudéennes par Longstaff-Schwartz (Hull-White et LMM).

Le détenteur d'une bermudéenne payeuse peut, à chaque date T_k, entrer
dans le swap co-terminal T_k -> T_end au taux K. Valeur d'exercice :
    h_k = A_k (S_k - K)^+ ,  A_k = Σ τ_j P(T_k, T_j) (annuité), S_k taux swap.
Les flux sont déflatés par le numéraire du modèle (compte courant continu
en HW, discret en LMM) ; les régressions se font en valeur « t_k ».

Contrôles de cohérence : prix bermudéen >= max des européennes
co-terminales, et égal à l'européenne quand il n'y a qu'une date.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..core.results import MCResult
from .hull_white import HullWhite
from .lmm import LiborMarketModel

__all__ = ["lsm_backward", "bermudan_swaption_hw", "bermudan_swaption_lmm"]


def _poly(x: np.ndarray, degree: int, h: np.ndarray) -> np.ndarray:
    cols = [np.ones(x.shape[0])]
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(x.shape[1]), d):
            cols.append(np.prod(x[:, combo], axis=1))
    cols.append(h)
    return np.column_stack(cols)


def lsm_backward(features, h, df, degree=3, coeffs=None):
    """LSM générique.

    features : liste de K tableaux (n, m) ; h : (n, K) valeurs d'exercice en
    t_k ; df : (n, K) déflateurs D(0, t_k). Si ``coeffs`` est None, estime les
    coefficients (rétro-induction) ; sinon applique la règle (hors
    échantillon). Renvoie (flux déflatés par trajectoire, coefficients).
    """
    n, K = h.shape
    scale = np.maximum(np.abs(h).mean(), 1e-12)
    if coeffs is None:
        coeffs = [None] * K
        cf = h[:, -1] * df[:, -1]
        for k in range(K - 2, -1, -1):
            itm = h[:, k] > 0
            if itm.sum() < 10:
                continue
            X = _poly(features[k], degree, h[:, k] / scale)
            beta, *_ = np.linalg.lstsq(X[itm], (cf / df[:, k])[itm], rcond=None)
            coeffs[k] = beta
            ex = itm & (h[:, k] > X @ beta)
            cf = np.where(ex, h[:, k] * df[:, k], cf)
        return cf, coeffs
    alive = np.ones(n, dtype=bool)
    cf = np.zeros(n)
    for k in range(K):
        if k == K - 1:
            ex = alive & (h[:, k] > 0)
        elif coeffs[k] is None:
            ex = np.zeros(n, dtype=bool)
        else:
            X = _poly(features[k], degree, h[:, k] / scale)
            ex = alive & (h[:, k] > 0) & (h[:, k] > X @ coeffs[k])
        cf = np.where(ex, h[:, k] * df[:, k], cf)
        alive &= ~ex
    return cf, coeffs


def _hw_data(hw: HullWhite, ex_times, pay_times, strike, payer, n_paths, seed):
    p = hw.simulate(ex_times, n_paths, seed=seed, antithetic=True)
    sign = 1.0 if payer else -1.0
    feats, H, D = [], [], []
    for T in ex_times:
        i = p.index(T)
        rem = pay_times[pay_times > T + 1e-12]
        S, A = hw.swap_rate(T, rem, p.x[:, i])
        H.append(np.maximum(sign * A * (S - strike), 0.0))
        D.append(p.discount[:, i])
        feats.append(np.column_stack([p.x[:, i] / hw.sigma]))
    return feats, np.column_stack(H), np.column_stack(D)


def bermudan_swaption_hw(hw: HullWhite, exercise_times, end: float, strike: float,
                         payer: bool = True, freq: float = 1.0, n_paths: int = 50_000,
                         degree: int = 3, seed: int = 0) -> MCResult:
    """Bermudéenne co-terminale sous Hull-White : exercice aux dates données,
    swap sous-jacent jusqu'à ``end`` avec paiements fixes tous les ``freq`` ans."""
    ex = np.asarray(exercise_times, dtype=float)
    pay = np.arange(ex[0] + freq, end + 1e-9, freq)
    fit = _hw_data(hw, ex, pay, strike, payer, n_paths, seed)
    _, coeffs = lsm_backward(*fit, degree=degree)
    cf, _ = lsm_backward(*_hw_data(hw, ex, pay, strike, payer, n_paths, seed + 1),
                         degree=degree, coeffs=coeffs)
    # erreur standard sur les moyennes de paires antithétiques
    h = cf.size // 2
    pairs = 0.5 * (cf[:h] + cf[h:])
    europeans = [hw.swaption(T, pay[pay > T + 1e-12], strike, payer) for T in ex]
    return MCResult(float(cf.mean()), float(pairs.std(ddof=1) / np.sqrt(h)), cf.size,
                    method="LSM Hull-White",
                    extra={"max_european": max(europeans), "europeans": europeans})


def bermudan_swaption_lmm(lmm: LiborMarketModel, first_ex: int, end: int, strike: float,
                          payer: bool = True, n_paths: int = 50_000, degree: int = 2,
                          seed: int = 0) -> MCResult:
    """Bermudéenne sous LMM : exercice en T_k, k = first_ex..end-1, swap co-terminal
    jusqu'à T_end. Variables de régression : taux swap et premier forward."""
    sign = 1.0 if payer else -1.0

    def data(sd):
        p = lmm.simulate(n_paths, seed=sd, antithetic=True)
        feats, H, D = [], [], []
        for k in range(first_ex, end):
            S, A = p.swap(k, end)
            H.append(np.maximum(sign * A * (S - strike), 0.0))
            D.append(1.0 / p.numeraire[:, k])
            feats.append(np.column_stack([S / strike, p.forwards[:, k, k] / strike]))
        return feats, np.column_stack(H), np.column_stack(D)

    _, coeffs = lsm_backward(*data(seed), degree=degree)
    cf, _ = lsm_backward(*data(seed + 1), degree=degree, coeffs=coeffs)
    h = cf.size // 2
    pairs = 0.5 * (cf[:h] + cf[h:])
    return MCResult(float(cf.mean()), float(pairs.std(ddof=1) / np.sqrt(h)), cf.size,
                    method="LSM LMM")
