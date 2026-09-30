"""Générateurs de gaussiennes pour la simulation Monte Carlo.

Toute la librairie consomme des tableaux de gaussiennes standard de forme
``(n_paths, n_steps, n_factors)`` : ``z[p, i, f]`` est l'incrément brownien
normalisé ``ΔW_f(t_i) / sqrt(Δt_i)`` du facteur ``f`` sur le pas ``i``.

Méthodes disponibles
--------------------
* ``"pseudo"`` : PCG64 (numpy). Estimation d'erreur par l'écart-type empirique.
* ``"sobol"``  : suites de Sobol brouillées (Owen, directions Joe-Kuo 2008,
  implémentation scipy). Une réalisation de QMC randomisé (RQMC) ; l'erreur
  est estimée par répétition sur des brouillages indépendants.

Constructions du mouvement brownien
-----------------------------------
En pseudo-aléatoire, la construction ne change pas la loi. En QMC elle est
cruciale : les premières coordonnées de Sobol sont les mieux réparties, on
leur confie donc les directions qui portent le plus de variance
(pont brownien : W(T) d'abord, puis les milieux ; ACP : vecteurs propres de
la covariance min(s, t) par valeur propre décroissante). Cela réduit la
« dimension effective » du problème (Caflisch, Morokoff & Owen 1997).

Techniques de réduction de variance « à la source »
---------------------------------------------------
* variables antithétiques : (Z, -Z) ;
* moment matching : recentrage/réduction empirique pas par pas ;
* stratification de W(T) (via le pont brownien).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

__all__ = ["BrownianBridge", "GaussianGenerator", "PCAConstruction", "poisson_inverse"]


def poisson_inverse(u: np.ndarray, mean) -> np.ndarray:
    """Inverse de la fonction de répartition de Poisson(mean), vectorisée.

    Recherche séquentielle P(N <= k) = Σ_{j<=k} e^{-μ} μ^j/j! : très rapide
    pour les petites intensités par pas (λΔt << 1) des modèles à sauts, et
    compatible QMC (une uniforme par tirage). ~100x plus rapide que
    scipy.stats.poisson.ppf.
    """
    u = np.asarray(u, dtype=float)
    mu = np.broadcast_to(np.asarray(mean, dtype=float), u.shape)
    p = np.exp(-mu)
    cdf = p.copy()
    n = np.zeros(u.shape)
    active = u > cdf
    k = 0
    while np.any(active) and k < 1000:
        k += 1
        p = p * mu / k
        cdf = cdf + p
        n = np.where(active, k, n)
        active = active & (u > cdf)
    return n


class BrownianBridge:
    """Construction du mouvement brownien par pont brownien (Glasserman §3.1).

    Ordre de construction : W(t_n), puis récursivement le point milieu (en
    indice) de chaque intervalle déjà encadré. Pour ``s < u < t`` :

        W(u) | W(s), W(t) ~ N( ((t-u) W(s) + (u-s) W(t)) / (t-s),
                               (u-s)(t-u)/(t-s) )

    Implémentation vectorisée inspirée de QuantLib (Jäckel, *Monte Carlo
    Methods in Finance*, 2002).
    """

    def __init__(self, times: np.ndarray):
        t = np.asarray(times, dtype=float)
        if t.ndim != 1 or t.size == 0 or np.any(np.diff(t) <= 0) or t[0] <= 0:
            raise ValueError("times doit être strictement croissant et > 0")
        n = t.size
        self.times = t
        self.size = n
        bmap = np.zeros(n, dtype=int)
        self.bridge_index = np.zeros(n, dtype=int)
        self.left_index = np.zeros(n, dtype=int)
        self.right_index = np.zeros(n, dtype=int)
        self.left_weight = np.zeros(n)
        self.right_weight = np.zeros(n)
        self.std_dev = np.zeros(n)

        bmap[n - 1] = 1
        self.bridge_index[0] = n - 1
        self.std_dev[0] = np.sqrt(t[n - 1])
        j = 0
        for i in range(1, n):
            while bmap[j]:
                j += 1
            k = j
            while not bmap[k]:
                k += 1
            m = j + ((k - 1 - j) >> 1)
            bmap[m] = i
            self.bridge_index[i] = m
            self.left_index[i] = j
            self.right_index[i] = k
            if j != 0:
                span = t[k] - t[j - 1]
                self.left_weight[i] = (t[k] - t[m]) / span
                self.right_weight[i] = (t[m] - t[j - 1]) / span
                self.std_dev[i] = np.sqrt((t[m] - t[j - 1]) * (t[k] - t[m]) / span)
            else:
                self.left_weight[i] = (t[k] - t[m]) / t[k]
                self.right_weight[i] = t[m] / t[k]
                self.std_dev[i] = np.sqrt(t[m] * (t[k] - t[m]) / t[k])
            j = k + 1
            if j >= n:
                j = 0
        self._sqrt_dt = np.sqrt(np.diff(np.concatenate(([0.0], t))))

    def transform(self, z: np.ndarray) -> np.ndarray:
        """Transforme des gaussiennes i.i.d. (ordre « pont ») en incréments
        normalisés ΔW_i/sqrt(Δt_i). ``z`` : (n_paths, n_steps, n_factors)."""
        w = np.empty_like(z)
        n = self.size
        w[:, n - 1] = self.std_dev[0] * z[:, 0]
        for i in range(1, n):
            j, k, m = self.left_index[i], self.right_index[i], self.bridge_index[i]
            if j != 0:
                w[:, m] = (
                    self.left_weight[i] * w[:, j - 1]
                    + self.right_weight[i] * w[:, k]
                    + self.std_dev[i] * z[:, i]
                )
            else:
                w[:, m] = self.right_weight[i] * w[:, k] + self.std_dev[i] * z[:, i]
        dw = np.diff(w, axis=1, prepend=0.0)
        return dw / self._sqrt_dt[None, :, None]


class PCAConstruction:
    """Construction par analyse en composantes principales.

    W = V diag(sqrt(λ)) Z avec C_ij = min(t_i, t_j) = V Λ V^T, valeurs propres
    triées par ordre décroissant. Optimal au sens de la variance expliquée
    (Acworth, Broadie & Glasserman 1998) mais coût O(n²) par trajectoire.
    """

    def __init__(self, times: np.ndarray):
        t = np.asarray(times, dtype=float)
        cov = np.minimum.outer(t, t)
        lam, vec = np.linalg.eigh(cov)
        order = np.argsort(lam)[::-1]
        lam, vec = np.clip(lam[order], 0.0, None), vec[:, order]
        self.matrix = vec * np.sqrt(lam)[None, :]
        self.explained_variance = np.cumsum(lam) / lam.sum()
        self._sqrt_dt = np.sqrt(np.diff(np.concatenate(([0.0], t))))

    def transform(self, z: np.ndarray) -> np.ndarray:
        w = np.einsum("ij,pjf->pif", self.matrix, z)
        dw = np.diff(w, axis=1, prepend=0.0)
        return dw / self._sqrt_dt[None, :, None]


class GaussianGenerator:
    """Source de gaussiennes (n_paths, n_steps, n_factors).

    Parameters
    ----------
    method : "pseudo" | "sobol"
    seed : graine (reproductibilité, indispensable aux Greeks par bump : CRN).
    antithetic : renvoie (Z, -Z) ; l'estimateur doit apparier i et i + n/2.
    moment_matching : impose moyenne 0 / variance 1 empiriques pas par pas.
    construction : "standard" | "bridge" | "pca".
    n_strata : si fourni, stratifie la 1re coordonnée (W(T) du facteur 0 avec
        un pont brownien) : u = (j + U)/J, j = indice de trajectoire mod J.
    """

    def __init__(
        self,
        method: str = "pseudo",
        seed: int | None = None,
        antithetic: bool = False,
        moment_matching: bool = False,
        construction: str = "standard",
        n_strata: int | None = None,
    ):
        if method not in ("pseudo", "sobol"):
            raise ValueError(f"méthode inconnue : {method}")
        if construction not in ("standard", "bridge", "pca"):
            raise ValueError(f"construction inconnue : {construction}")
        self.method = method
        self.antithetic = antithetic
        self.moment_matching = moment_matching
        self.construction = construction
        self.n_strata = n_strata
        self._rng = np.random.default_rng(seed)
        self._sobol: qmc.Sobol | None = None
        self._dim: int | None = None
        self._builder: BrownianBridge | PCAConstruction | None = None
        self._builder_times: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _uniform_or_normal(self, n: int, n_steps: int, n_factors: int) -> np.ndarray:
        dim = n_steps * n_factors
        if self.method == "pseudo":
            if self.n_strata:
                z = self._rng.standard_normal((n, n_steps, n_factors))
                u = (np.arange(n) % self.n_strata + self._rng.random(n)) / self.n_strata
                z[:, 0, 0] = ndtri(np.clip(u, 1e-16, 1 - 1e-16))
                return z
            return self._rng.standard_normal((n, n_steps, n_factors))
        if self._sobol is None:
            if dim > 21201:
                raise ValueError("Sobol limité à 21201 dimensions (Joe-Kuo)")
            self._sobol = qmc.Sobol(d=dim, scramble=True, seed=self._rng)
            self._dim = dim
        elif dim != self._dim:
            raise ValueError("dimension QMC modifiée entre deux tirages")
        u = self._sobol.random(n)
        u = np.clip(u, 1e-16, 1.0 - 1e-16)
        # Coordonnée d -> (priorité temporelle d // F, facteur d % F) : les
        # premières coordonnées alimentent le 1er niveau du pont / 1re composante.
        return ndtri(u).reshape(n, n_steps, n_factors)

    def _get_builder(self, times: np.ndarray):
        if self.construction == "standard":
            return None
        cached = self._builder_times
        if self._builder is None or cached is None or not np.array_equal(cached, times):
            cls = BrownianBridge if self.construction == "bridge" else PCAConstruction
            self._builder = cls(times)
            self._builder_times = times.copy()
        return self._builder

    def normals(self, n_paths: int, times: np.ndarray, n_factors: int) -> np.ndarray:
        """Renvoie z de forme (n_paths, len(times)-1, n_factors).

        ``times`` est la grille complète incluant t0 = 0.
        """
        times = np.asarray(times, dtype=float)
        n_steps = times.size - 1
        if self.antithetic:
            if n_paths % 2:
                raise ValueError("n_paths doit être pair en antithétique")
            half = self._uniform_or_normal(n_paths // 2, n_steps, n_factors)
            z = np.concatenate([half, -half], axis=0)
        else:
            z = self._uniform_or_normal(n_paths, n_steps, n_factors)
        if self.moment_matching:
            if not self.antithetic:
                z = z - z.mean(axis=0, keepdims=True)
            z = z / z.std(axis=0, keepdims=True)
        builder = self._get_builder(times[1:])
        if builder is not None:
            z = builder.transform(z)
        return z

    def uniforms(self, shape) -> np.ndarray:
        """Uniformes pseudo-aléatoires auxiliaires (hors QMC)."""
        return self._rng.random(shape)
