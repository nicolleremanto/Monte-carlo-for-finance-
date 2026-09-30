"""Résultats Monte Carlo et estimateurs statistiques."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

__all__ = ["MCResult", "Paths", "mean_and_stderr"]


@dataclass
class MCResult:
    """Prix Monte Carlo avec son erreur standard.

    ``stderr`` est l'écart-type de l'estimateur (et non des tirages) :
    l'intervalle de confiance asymptotique (TCL) est price ± z_α stderr.
    ``dof`` : degrés de liberté quand l'erreur est estimée sur peu de
    répétitions indépendantes (QMC randomisé : R - 1) ; l'IC utilise alors le
    quantile de Student au lieu du quantile gaussien.
    """

    price: float
    stderr: float
    n_paths: int
    elapsed: float = 0.0
    method: str = ""
    extra: dict = field(default_factory=dict)
    dof: int | None = None

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        p = 0.5 + level / 2
        z = norm.ppf(p) if self.dof is None else student_t.ppf(p, self.dof)
        return self.price - z * self.stderr, self.price + z * self.stderr

    @property
    def rel_error(self) -> float:
        return self.stderr / abs(self.price) if self.price else np.inf

    def __repr__(self) -> str:
        lo, hi = self.ci()
        return (
            f"MCResult(price={self.price:.6f}, stderr={self.stderr:.6f}, "
            f"IC95=[{lo:.6f}, {hi:.6f}], n={self.n_paths}, "
            f"{self.method}, {self.elapsed:.2f}s)"
        )


@dataclass
class Paths:
    """Trajectoires simulées, restreintes aux dates d'observation.

    Attributes
    ----------
    times : (n_obs+1,) dates, times[0] = 0.
    spot : (n_paths, n_obs+1) ou (n_paths, n_obs+1, n_assets).
    discount : facteurs d'actualisation D(0, t) = 1/B(t), de forme (n_obs+1,)
        (taux déterministes) ou (n_paths, n_obs+1) (taux stochastiques).
    int_var : variance intégrée ∫σ²dt entre deux dates d'observation
        (n_paths, n_obs[, n_assets]) — utilisée par la correction de pont
        brownien des barrières et par les swaps de variance.
    variance : variance instantanée (modèles à volatilité stochastique).
    """

    times: np.ndarray
    spot: np.ndarray
    discount: np.ndarray
    int_var: np.ndarray | None = None
    variance: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n_paths(self) -> int:
        return self.spot.shape[0]

    def df(self, idx) -> np.ndarray:
        """Facteur d'actualisation vers l'indice d'observation ``idx``."""
        d = self.discount
        return d[idx] if d.ndim == 1 else d[:, idx]


def mean_and_stderr(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))
