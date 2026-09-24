"""Estimateurs de sensibilités Monte Carlo (Glasserman 2003, ch. 7).

1. Différences finies avec nombres aléatoires communs (MonteCarloEngine.greeks).
   Biais O(h²) (centré), variance O(1) avec CRN (O(1/h²) sans).

2. Pathwise (IPA) : dV/dθ = E[ dY/dθ ] en dérivant la trajectoire.
   Sans biais si le payoff est lipschitzien ; variance minimale.
   Pour un GBM : ∂S_t/∂S_0 = S_t/S_0 et ∂S_t/∂σ = S_t (W_t - σ t).
   Échoue pour les payoffs discontinus (digitale : dérivée nulle p.s.).
   Implémenté ici par AAD (mcfin.greeks.aad) : un seul balayage arrière.

3. Ratio de vraisemblance (LRM, Broadie & Glasserman 1996) :
   dV/dθ = E[ Y · ∂ ln p_θ(X) / ∂θ ]  — on dérive la densité, pas le payoff :
   aucune régularité requise (digitales, barrières), mais variance élevée
   (et qui explose quand Δt_1 -> 0). Scores pour un GBM (z_i gaussiennes des
   pas de simulation) :
       delta : z_1 / (S_0 σ sqrt(Δt_1))
       gamma : (z_1² - 1)/(S_0² σ² Δt_1) - z_1/(S_0² σ sqrt(Δt_1))
       vega  : Σ_i [ (z_i² - 1)/σ - z_i sqrt(Δt_i) ]
"""
from __future__ import annotations

import numpy as np

from ..analytics.black_scholes import option_sign
from ..core.rng import GaussianGenerator
from ..models.equity import BlackScholes
from . import aad

__all__ = ["likelihood_ratio_greeks", "gbm_path_pricer", "heston_pricer"]


def likelihood_ratio_greeks(model: BlackScholes, product, n_paths: int = 200_000,
                            seed: int = 0, max_dt: float | None = None) -> dict:
    """Delta, gamma, vega par LRM pour tout produit mono-sous-jacent sous GBM."""
    if model.n_assets != 1:
        raise ValueError("LRM implémenté pour un seul sous-jacent")
    grid = model.build_grid(product.observation_times, max_dt)
    z = GaussianGenerator("pseudo", seed).normals(n_paths, grid.times, 1)
    y = product.payoff(model.simulate(grid, z))
    s0, sig = float(model.spot), float(model.vol)
    dt = grid.dt
    z1 = z[:, 0, 0]
    scores = {
        "delta": z1 / (s0 * sig * np.sqrt(dt[0])),
        "gamma": (z1**2 - 1) / (s0**2 * sig**2 * dt[0]) - z1 / (s0**2 * sig * np.sqrt(dt[0])),
        "vega": np.sum((z[:, :, 0] ** 2 - 1) / sig - z[:, :, 0] * np.sqrt(dt)[None], axis=1),
    }
    out = {"price": float(y.mean())}
    for k, sc in scores.items():
        v = y * sc
        out[k] = float(v.mean())
        out[k + "_stderr"] = float(v.std(ddof=1) / np.sqrt(n_paths))
    return out


def gbm_path_pricer(product_payoff, fixing_times, maturity: float):
    """Fabrique un pricer différentiable (AAD) sous GBM.

    ``product_payoff(spots: list[Var], xp) -> Var par trajectoire`` où
    ``spots[i]`` est le spot au fixing i et ``xp`` le module aad (fonctions
    exp/maximum/... compatibles AAD). Paramètres : spot, vol, rate, div.
    """
    times = np.asarray(fixing_times, dtype=float)
    dts = np.diff(np.concatenate(([0.0], times)))

    def pricer(p, z):
        s0, sig, r = p["spot"], p["vol"], p["rate"]
        q = p.get("div", 0.0)
        drift = r - q - 0.5 * sig * sig
        x = 0.0
        spots = []
        for i, dt in enumerate(dts):
            x = drift * dt + (sig * np.sqrt(dt)) * z[:, i] + x
            spots.append(s0 * aad.exp(x))
        pay = product_payoff(spots, aad)
        return aad.mean(pay) * aad.exp(-r * maturity)
    return pricer


def heston_pricer(strike: float, maturity: float, n_steps: int, option_type="call"):
    """Pricer européen Heston (Euler full truncation) différentiable par AAD
    par rapport à spot, v0, kappa, theta, xi, rho, rate. z : (n, n_steps, 2)."""
    w = option_sign(option_type)
    dt = maturity / n_steps

    def pricer(p, z):
        s0, v0, kappa, theta = p["spot"], p["v0"], p["kappa"], p["theta"]
        xi, rho, r = p["xi"], p["rho"], p["rate"]
        rc = aad.sqrt(1.0 - rho * rho)
        x = 0.0
        v = v0 + np.zeros(z.shape[0])
        for i in range(n_steps):
            vp = aad.maximum(v, 0.0)
            sq = aad.sqrt(vp * dt)
            zv = z[:, i, 0]
            x = x + (r - 0.5 * vp) * dt + sq * (rho * zv + rc * z[:, i, 1])
            v = v + kappa * (theta - vp) * dt + xi * sq * zv
        st = s0 * aad.exp(x)
        pay = aad.maximum(w * (st - strike), 0.0)
        return aad.mean(pay) * aad.exp(-r * maturity)
    return pricer
