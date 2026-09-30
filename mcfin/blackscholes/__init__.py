"""Black-Scholes de A à Z : simulation, EDP, couverture, inférence, diagnostics.

Modules
-------
pricer       façade : formule fermée, Monte Carlo, EDP, arbre sur une même option
sampling     loi normale « from scratch » (inversion, Box-Muller, polaire, rejet)
pde          Crank-Nicolson + Rannacher, américaines par Brennan-Schwartz
hedging      couverture delta discrète, vol mal spécifiée, coûts de Leland
estimation   maximum de vraisemblance, estimateurs de volatilité par l'amplitude
diagnostics  couverture des IC, normalité (TCL), vitesse de convergence
"""

from .diagnostics import CoverageReport, convergence_rate, coverage_study
from .estimation import GBMFit, fit_gbm_mle, range_volatility, simulate_ohlc
from .hedging import HedgeResult, kamal_derman_std, leland_volatility, simulate_delta_hedge
from .pde import PDEResult, bs_pde_price
from .pricer import BlackScholesPricer
from .sampling import REJECTION_CONSTANT, box_muller, inverse_normal_bsm, marsaglia_polar, rejection_laplace

__all__ = [
    "REJECTION_CONSTANT",
    "BlackScholesPricer",
    "CoverageReport",
    "GBMFit",
    "HedgeResult",
    "PDEResult",
    "box_muller",
    "bs_pde_price",
    "convergence_rate",
    "coverage_study",
    "fit_gbm_mle",
    "inverse_normal_bsm",
    "kamal_derman_std",
    "leland_volatility",
    "marsaglia_polar",
    "range_volatility",
    "rejection_laplace",
    "simulate_delta_hedge",
    "simulate_ohlc",
]
