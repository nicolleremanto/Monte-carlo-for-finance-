"""Moteur de pricing Monte Carlo.

Chaîne de traitement
--------------------
1. le produit déclare ses dates d'observation ;
2. le modèle construit la grille de simulation (pas maximal éventuel) ;
3. le générateur fournit les gaussiennes (pseudo ou Sobol, pont brownien...) ;
4. le modèle simule, le produit calcule le flux actualisé par trajectoire ;
5. l'estimateur agrège : moyenne, erreur standard, variables de contrôle.

Estimation de l'erreur
----------------------
* pseudo-aléatoire : écart-type empirique / sqrt(n) ; en antithétique, sur
  les moyennes de paires (les tirages d'une paire ne sont pas indépendants) ;
* stratifié : Var = Σ_j p_j² s_j² / n_j ;
* QMC randomisé : R brouillages indépendants de Sobol, erreur = écart-type
  des R estimateurs / sqrt(R) (Owen 1997, L'Ecuyer & Lemieux 2002).

Variables de contrôle
---------------------
Pour Y (payoff) et X (contrôles d'espérance connue μ_X) :
    Ŷ_cv = Ȳ - β̂ᵀ (X̄ - μ_X),   β̂ = Cov(X)⁻¹ Cov(X, Y)   (MCO)
La réduction de variance vaut 1 - R² de la régression de Y sur X.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .core.results import MCResult, Paths
from .core.rng import GaussianGenerator
from .models.base import Model, bump_model

__all__ = ["ControlVariate", "MonteCarloEngine"]


def _concat_paths(parts: list[Paths]) -> Paths:
    """Concatène des lots de trajectoires (axe 0 = trajectoires)."""
    if len(parts) == 1:
        return parts[0]
    first = parts[0]

    def cat(name):
        vals = [getattr(p, name) for p in parts]
        if vals[0] is None:
            return None
        if name == "discount" and vals[0].ndim == 1:
            return vals[0]
        return np.concatenate(vals, axis=0)

    extra = {k: np.concatenate([p.extra[k] for p in parts], axis=0) for k in first.extra}
    return Paths(
        times=first.times,
        spot=cat("spot"),
        discount=cat("discount"),
        int_var=cat("int_var"),
        variance=cat("variance"),
        extra=extra,
    )


@dataclass
class ControlVariate:
    """Variable de contrôle : fonction des trajectoires d'espérance connue."""

    fn: Callable[[Paths], np.ndarray]
    expectation: float
    name: str = "cv"


class MonteCarloEngine:
    """Moteur générique.

    Parameters
    ----------
    n_paths : nombre total de trajectoires (en QMC : par brouillage × R).
    method : "pseudo" | "sobol".
    antithetic, moment_matching, construction, n_strata : voir GaussianGenerator.
    n_randomizations : nombre de brouillages indépendants en QMC.
    batch_size : taille des lots (mémoire bornée quel que soit n_paths).
    max_dt : pas de temps maximal (sinon celui du modèle).
    """

    def __init__(
        self,
        n_paths: int = 100_000,
        seed: int | None = 42,
        method: str = "pseudo",
        antithetic: bool = False,
        moment_matching: bool = False,
        construction: str = "standard",
        n_randomizations: int = 16,
        batch_size: int = 50_000,
        max_dt: float | None = None,
        n_strata: int | None = None,
        max_batch_elements: int = 10_000_000,
    ):
        self.n_paths = int(n_paths)
        self.seed = seed
        self.method = method
        self.antithetic = antithetic
        self.moment_matching = moment_matching
        self.construction = construction
        self.n_randomizations = n_randomizations if method == "sobol" else 1
        self.batch_size = int(batch_size)
        self.max_dt = max_dt
        self.n_strata = n_strata
        self.max_batch_elements = int(max_batch_elements)
        if n_strata and (antithetic or method != "pseudo"):
            raise ValueError("stratification : pseudo-aléatoire sans antithétique")
        if n_strata and self.batch_size % n_strata:
            raise ValueError("batch_size doit être un multiple de n_strata")

    # ------------------------------------------------------------------
    def _generator(self, rep: int) -> GaussianGenerator:
        seed = None if self.seed is None else self.seed + 7919 * rep
        return GaussianGenerator(
            self.method, seed, self.antithetic, self.moment_matching, self.construction, self.n_strata
        )

    def _batch_size(self, grid, n_factors: int) -> int:
        """Taille de lot bornant la mémoire des gaussiennes (~80 Mo par défaut)."""
        cap = max(1024, self.max_batch_elements // max(1, grid.n_steps * n_factors))
        bs = min(self.batch_size, cap)
        if self.n_strata:
            bs = max(self.n_strata, bs - bs % self.n_strata)
        return bs

    def _batches(self, n: int, bs: int):
        if self.antithetic and bs % 2:
            bs += 1
        while n > 0:
            b = min(bs, n)
            if self.antithetic and b % 2:
                b += 1
            yield b
            n -= b

    def simulate(
        self,
        model: Model,
        obs_times,
        n_paths: int | None = None,
        rep: int = 0,
        drift_shift: np.ndarray | None = None,
    ) -> Paths:
        """Simule toutes les trajectoires d'un coup (pour LSM, XVA, analyses)."""
        n = n_paths or self.n_paths
        grid = model.build_grid(obs_times, self.max_dt)
        gen = self._generator(rep)
        parts = []
        for b in self._batches(n, self._batch_size(grid, model.n_factors)):
            z = gen.normals(b, grid.times, model.n_factors)
            if drift_shift is not None:
                z = z + drift_shift[None]
            parts.append(model.simulate(grid, z, gen))
        return _concat_paths(parts)

    # ------------------------------------------------------------------
    def sample(
        self,
        model: Model,
        product,
        control_variates: Sequence[ControlVariate] = (),
        drift_shift: np.ndarray | None = None,
    ):
        """Renvoie la liste (par brouillage) des tableaux (payoffs, contrôles)."""
        grid = model.build_grid(product.observation_times, self.max_dt)
        n_rep = self.n_randomizations
        n_per_rep = self.n_paths // n_rep
        out = []
        for rep in range(n_rep):
            gen = self._generator(rep)
            ys, xs = [], []
            for b in self._batches(n_per_rep, self._batch_size(grid, model.n_factors)):
                z = gen.normals(b, grid.times, model.n_factors)
                weight = None
                if drift_shift is not None:
                    # échantillonnage préférentiel : Z = Z0 + μ, rapport de vraisemblance
                    # dP/dQ = exp(-μ·Z0 - |μ|²/2) = exp(-μ·Z + |μ|²/2)
                    mu = drift_shift
                    z = z + mu[None]
                    weight = np.exp(-(z * mu[None]).sum(axis=(1, 2)) + 0.5 * np.sum(mu * mu))
                paths = model.simulate(grid, z, gen)
                y = np.asarray(product.payoff(paths), dtype=float)
                if weight is not None:
                    y = y * weight
                x = (
                    np.column_stack([cv.fn(paths) for cv in control_variates])
                    if control_variates
                    else np.empty((b, 0))
                )
                if weight is not None and control_variates:
                    x = x * weight[:, None]
                if self.antithetic:
                    h = b // 2
                    y = 0.5 * (y[:h] + y[h:])
                    x = 0.5 * (x[:h] + x[h:])
                ys.append(y)
                xs.append(x)
            out.append((np.concatenate(ys), np.concatenate(xs)))
        return out

    def price(
        self,
        model: Model,
        product,
        control_variates: Sequence[ControlVariate] = (),
        drift_shift: np.ndarray | None = None,
    ) -> MCResult:
        t0 = time.perf_counter()
        samples = self.sample(model, product, control_variates, drift_shift)
        mu_x = np.array([cv.expectation for cv in control_variates])
        extra = {}
        beta = None
        if control_variates:
            y_all = np.concatenate([s[0] for s in samples])
            x_all = np.concatenate([s[1] for s in samples])
            xc = x_all - x_all.mean(axis=0)
            beta = np.linalg.lstsq(xc, y_all - y_all.mean(), rcond=None)[0]
            extra["beta"] = beta
            resid_var = np.var(y_all - xc @ beta, ddof=1)
            extra["variance_reduction"] = float(np.var(y_all, ddof=1) / max(resid_var, 1e-300))

        def adjust(y, x):
            return y if beta is None else y - (x - mu_x) @ beta

        if self.method == "sobol":
            ests = np.array([adjust(y, x).mean() for y, x in samples])
            price = float(ests.mean())
            stderr = float(ests.std(ddof=1) / np.sqrt(ests.size)) if ests.size > 1 else np.nan
        else:
            y, x = samples[0]
            v = adjust(y, x)
            price = float(v.mean())
            if self.n_strata:
                labels = np.arange(v.size) % self.n_strata
                var = (
                    sum(np.var(v[labels == j], ddof=1) / np.sum(labels == j) for j in range(self.n_strata))
                    / self.n_strata**2
                )
                stderr = float(np.sqrt(var))
            else:
                stderr = float(v.std(ddof=1) / np.sqrt(v.size))
        desc = (
            self.method
            + ("+anti" if self.antithetic else "")
            + ("+MM" if self.moment_matching else "")
            + (f"+{self.construction}" if self.construction != "standard" else "")
            + (f"+strat{self.n_strata}" if self.n_strata else "")
            + ("+CV" if control_variates else "")
            + ("+IS" if drift_shift is not None else "")
        )
        dof = self.n_randomizations - 1 if self.method == "sobol" else None
        return MCResult(price, stderr, self.n_paths, time.perf_counter() - t0, desc, extra, dof)

    # ------------------------------------------------------------------
    def greeks(
        self,
        model: Model,
        product,
        params: dict[str, float] | None = None,
        spot_bump: float = 0.01,
        second_order: bool = True,
    ) -> dict:
        """Greeks par différences finies centrées avec nombres aléatoires communs.

        Avec la même graine, V(θ+h) - V(θ-h) a une variance O(1) au lieu de
        O(1/h²) : c'est ce qui rend les Greeks par bump utilisables.
        ``spot_bump`` est relatif (1 % par défaut) ; ``params`` = {nom: choc absolu}.
        """
        base = self.price(model, product)
        out = {"price": base.price, "price_stderr": base.stderr}
        s0 = np.asarray(model.spot, dtype=float)
        h = spot_bump * s0
        up = self.price(bump_model(model, "spot", spot_bump, relative=True), product).price
        dn = self.price(bump_model(model, "spot", -spot_bump, relative=True), product).price
        hs = float(np.mean(h))
        out["delta"] = (up - dn) / (2 * hs)
        if second_order:
            out["gamma"] = (up - 2 * base.price + dn) / hs**2
        for name, bump in (params or {}).items():
            up = self.price(bump_model(model, name, bump), product).price
            dn = self.price(bump_model(model, name, -bump), product).price
            out[name] = (up - dn) / (2 * bump)
        return out
