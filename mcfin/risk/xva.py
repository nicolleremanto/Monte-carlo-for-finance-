"""XVA d'un portefeuille de swaps de taux (Hull-White 1F).

Chaîne de calcul d'un desk XVA
------------------------------
1. Diffusion des facteurs de risque (ici x_t de Hull-White, exact) sur une
   grille de dates d'exposition, enrichie des dates de fixing (coupon
   variable en cours) et des dates décalées de la MPOR (collatéral).
2. Revalorisation analytique de chaque trade à chaque date (pas de MC
   imbriqué : prix affines en x) puis agrégation par ensemble de netting.
3. Collatéral (CSA bilatéral) : C(t) = (V(t-δ) - H_c)^+ - (-V(t-δ) - H_b)^+,
   δ = marge de risque (MPOR, ~10 jours ouvrés) ; exposition = V(t) - C(t).
4. Profils : EE(t) = E[D(t) E(t)^+] / P(0,t) (espérance sous Q),
   ENE(t), PFE_q(t) = quantile q de E(t) (sous Q ici ; sous P en pratique).
5. Ajustements (formules unilatérales discrétisées, Gregory 2020) :
     CVA = (1-R_c) Σ_i E[D(t_i) E⁺(t_i)] · (S_c(t_{i-1}) - S_c(t_i)) · S_b(t_{i-1})
     DVA = (1-R_b) Σ_i E[D(t_i) E⁻(t_i)] · (S_b(t_{i-1}) - S_b(t_i)) · S_c(t_{i-1})
     FVA = FCA + FBA,  FCA = -Σ s_f E[D E⁺] Δt S_c S_b,  FBA = +Σ s_f E[D E⁻] Δt S_c S_b
   (convention : ajustements en signe de valeur pour la banque ; CVA > 0
   est un coût). S(t) = exp(-λ t) avec λ ≈ spread CDS / (1 - R) (triangle
   du crédit). Le facteur S_b/S_c optionnel donne la version « first-to-
   default » bilatérale.
Hypothèse : indépendance crédit/marché (pas de wrong-way risk).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from ..rates.hull_white import HullWhite, HWPaths

__all__ = [
    "CSA",
    "Counterparty",
    "ExposureResult",
    "InterestRateSwap",
    "compute_xva",
    "simulate_exposure",
]


@dataclass
class InterestRateSwap:
    """Swap vanille mono-courbe : jambe fixe et variable de même échéancier.

    payer=True : la banque paie le fixe et reçoit le variable."""

    notional: float
    fixed_rate: float
    start: float
    end: float
    freq: float = 1.0
    payer: bool = True

    @property
    def schedule(self) -> np.ndarray:
        return np.round(np.arange(self.start, self.end + 1e-9, self.freq), 10)

    def value(self, hw: HullWhite, t: float, x: np.ndarray, fixings: dict) -> np.ndarray:
        """Valeur en t (vectorisée sur x). ``fixings[T_{m-1}]`` = taux variable
        fixé en T_{m-1} pour la période en cours."""
        sched = self.schedule
        if t >= sched[-1] - 1e-12:
            return np.zeros_like(x)
        tau = np.diff(sched)
        m = int(np.searchsorted(sched, t, side="right"))  # prochaine date > t
        if m == 0:  # swap non démarré
            float_leg = hw.bond(t, sched[0], x) - hw.bond(t, sched[-1], x)
            pays, taus = sched[1:], tau
        else:
            T_m, T_prev = sched[m], sched[m - 1]
            L_fix = fixings[round(T_prev, 10)]
            P_m = hw.bond(t, T_m, x)
            float_leg = (tau[m - 1] * L_fix) * P_m + P_m - hw.bond(t, sched[-1], x)
            pays, taus = sched[m:], tau[m - 1 :]
        P = hw.bond(t, pays[None, :], x[:, None])
        fixed_leg = self.fixed_rate * (P @ taus)
        v = float_leg - fixed_leg
        return self.notional * (v if self.payer else -v)


@dataclass
class CSA:
    threshold_cpty: float = 0.0  # seuil au-delà duquel la contrepartie poste
    threshold_bank: float = 0.0  # seuil au-delà duquel la banque poste
    mpor: float = 10.0 / 252  # marge de risque (Margin Period of Risk)


@dataclass
class Counterparty:
    cds_spread: float
    recovery: float = 0.4

    @property
    def hazard(self) -> float:
        return self.cds_spread / (1 - self.recovery)

    def survival(self, t):
        return np.exp(-self.hazard * np.asarray(t, dtype=float))


@dataclass
class ExposureResult:
    times: np.ndarray
    mtm: np.ndarray  # (n, nt) valeur du netting set
    exposure: np.ndarray  # (n, nt) après collatéral
    discount: np.ndarray  # (n, nt)
    ee: np.ndarray = field(default=None)
    ene: np.ndarray = field(default=None)
    pfe: np.ndarray = field(default=None)
    epe: float = 0.0
    eepe: float = 0.0


def simulate_exposure(
    hw: HullWhite,
    trades: list[InterestRateSwap],
    exposure_times,
    n_paths: int = 20_000,
    seed: int = 0,
    csa: CSA | None = None,
    pfe_quantile: float = 0.95,
) -> ExposureResult:
    t_exp = np.asarray(exposure_times, dtype=float)
    fix_dates = np.unique(np.concatenate([tr.schedule[:-1] for tr in trades]))
    grid = [t_exp, fix_dates]
    if csa is not None:
        grid.append(np.maximum(t_exp - csa.mpor, 0.0))
    grid = np.unique(np.round(np.concatenate(grid), 10))
    grid = grid[grid > 0]
    paths: HWPaths = hw.simulate(grid, n_paths, seed=seed, antithetic=True)
    n = paths.x.shape[0]

    # fixings variables : L(T_{m-1}, T_m) = (1/P(T_{m-1}, T_m) - 1)/τ
    fixings = []
    for tr in trades:
        f = {}
        sched = tr.schedule
        for T0, T1 in pairwise(sched):
            x0 = paths.x[:, paths.index(T0)] if T0 > 0 else np.zeros(n)
            f[round(T0, 10)] = (1 / hw.bond(T0, T1, x0) - 1) / (T1 - T0)
        fixings.append(f)

    def netting_value(t):
        x = paths.x[:, paths.index(t)] if t > 0 else np.zeros(n)
        return sum(tr.value(hw, t, x, fx) for tr, fx in zip(trades, fixings, strict=True))

    mtm = np.column_stack([netting_value(t) for t in t_exp])
    if csa is not None:
        lagged = np.column_stack([netting_value(max(t - csa.mpor, 0.0)) for t in t_exp])
        coll = np.maximum(lagged - csa.threshold_cpty, 0.0) - np.maximum(-lagged - csa.threshold_bank, 0.0)
        expo = mtm - coll
    else:
        expo = mtm
    disc = np.column_stack([paths.discount[:, paths.index(t)] for t in t_exp])
    p0 = hw.curve.df(t_exp)
    ee = (disc * np.maximum(expo, 0)).mean(axis=0) / p0
    ene = (disc * np.minimum(expo, 0)).mean(axis=0) / p0
    pfe = np.quantile(expo, pfe_quantile, axis=0)
    # EPE et EEPE (Bâle) sur la première année
    dt = np.diff(np.concatenate(([0.0], t_exp)))
    one_y = t_exp <= 1.0 + 1e-9
    epe = float(np.sum(ee[one_y] * dt[one_y]) / max(dt[one_y].sum(), 1e-12))
    eepe = float(np.sum(np.maximum.accumulate(ee)[one_y] * dt[one_y]) / max(dt[one_y].sum(), 1e-12))
    return ExposureResult(t_exp, mtm, expo, disc, ee, ene, pfe, epe, eepe)


def compute_xva(
    res: ExposureResult,
    cpty: Counterparty,
    bank: Counterparty | None = None,
    funding_spread: float = 0.0,
    first_to_default: bool = True,
) -> dict:
    t = res.times
    t_prev = np.concatenate(([0.0], t[:-1]))
    dee = (res.discount * np.maximum(res.exposure, 0)).mean(axis=0)  # E[D E⁺]
    dne = (res.discount * np.minimum(res.exposure, 0)).mean(axis=0)  # E[D E⁻] <= 0
    pd_c = cpty.survival(t_prev) - cpty.survival(t)
    s_b = bank.survival(t_prev) if (bank is not None and first_to_default) else 1.0
    cva = (1 - cpty.recovery) * np.sum(dee * pd_c * s_b)
    out = {"CVA": float(cva)}
    if bank is not None:
        pd_b = bank.survival(t_prev) - bank.survival(t)
        s_c = cpty.survival(t_prev) if first_to_default else 1.0
        out["DVA"] = float(-(1 - bank.recovery) * np.sum(dne * pd_b * s_c))
    if funding_spread:
        dt = t - t_prev
        surv = cpty.survival(t_prev) * (bank.survival(t_prev) if bank is not None else 1.0)
        out["FCA"] = float(funding_spread * np.sum(dee * dt * surv))
        out["FBA"] = float(-funding_spread * np.sum(dne * dt * surv))
    # ajustement total de valeur (vue banque) : -CVA + DVA - FCA + FBA
    out["total"] = -out["CVA"] + out.get("DVA", 0.0) - out.get("FCA", 0.0) + out.get("FBA", 0.0)
    return out
