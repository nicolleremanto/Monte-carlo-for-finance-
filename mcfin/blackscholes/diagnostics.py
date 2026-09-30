"""Diagnostics statistiques d'un estimateur Monte Carlo.

Un prix Monte Carlo est un *estimateur* : on doit pouvoir vérifier ses
propriétés comme en statistique inférentielle.

* ``coverage_study`` : on répète R fois l'estimation (graines
  indépendantes) avec un prix de référence connu, et l'on teste
  - la couverture empirique de l'IC à 95 % (IC binomial de Wilson sur la
    proportion : la couverture nominale doit y appartenir) ;
  - la loi des erreurs réduites Z_r = (V̂_r - V)/σ̂_r par Kolmogorov-Smirnov :
    N(0, 1) (TCL) en Monte Carlo, Student(R-1) en QMC randomisé à R
    brouillages ;
  - l'absence de biais : test de Student sur les erreurs *brutes* V̂_r - V.
    (Tester la moyenne des Z_r serait faux : pour un payoff asymétrique,
    V̂ et σ̂ sont corrélés, donc E[Z] ≠ 0 même pour un estimateur sans biais.)
* ``convergence_rate`` : régression log-log de l'erreur standard sur N ;
  la pente vaut -1/2 en Monte Carlo et s'approche de -1 en QMC.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..core.results import MCResult

__all__ = ["CoverageReport", "convergence_rate", "coverage_study"]


@dataclass
class CoverageReport:
    coverage: float
    coverage_ci: tuple[float, float]
    ks_pvalue: float
    bias_t_stat: float
    bias_pvalue: float
    z_scores: np.ndarray

    @property
    def passed(self) -> bool:
        lo, hi = self.coverage_ci
        return lo <= 0.95 <= hi and self.ks_pvalue > 0.01 and self.bias_pvalue > 0.01

    def __repr__(self) -> str:
        lo, hi = self.coverage_ci
        return (
            f"CoverageReport(couverture IC95 = {self.coverage:.3f} [{lo:.3f}, {hi:.3f}], "
            f"KS p = {self.ks_pvalue:.3f}, biais t = {self.bias_t_stat:+.2f} "
            f"(p = {self.bias_pvalue:.3f}), {'OK' if self.passed else 'ÉCHEC'})"
        )


def _wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(centre - half), float(centre + half)


def coverage_study(
    estimator: Callable[[int], MCResult], true_value: float, n_replications: int = 500
) -> CoverageReport:
    """``estimator(seed) -> MCResult`` ; ``true_value`` : prix exact."""
    z = np.empty(n_replications)
    err = np.empty(n_replications)
    inside = 0
    dof = None
    for i in range(n_replications):
        res = estimator(i)
        err[i] = res.price - true_value
        z[i] = err[i] / res.stderr
        lo, hi = res.ci(0.95)
        inside += int(lo <= true_value <= hi)
        dof = res.dof
    ref = "norm" if dof is None else stats.t(dof).cdf
    t = stats.ttest_1samp(err, 0.0)
    return CoverageReport(
        inside / n_replications,
        _wilson(inside, n_replications),
        float(stats.kstest(z, ref).pvalue),
        float(t.statistic),
        float(t.pvalue),
        z,
    )


def convergence_rate(estimator: Callable[[int], MCResult], n_values) -> dict:
    """Pente de log(erreur standard) contre log(N) avec son IC à 95 %."""
    n = np.asarray(n_values, dtype=float)
    se = np.array([estimator(int(k)).stderr for k in n])
    fit = stats.linregress(np.log(n), np.log(se))
    half = stats.t.ppf(0.975, n.size - 2) * fit.stderr
    return {"slope": float(fit.slope), "slope_ci": (fit.slope - half, fit.slope + half), "n": n, "stderr": se}
