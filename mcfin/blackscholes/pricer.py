"""Façade Black-Scholes : une option, quatre méthodes de valorisation.

    >>> from mcfin.blackscholes import BlackScholesPricer
    >>> opt = BlackScholesPricer(S0=100, K=105, T=1, r=0.03, sigma=0.25, q=0.01)
    >>> opt.compare()           # formule fermée, Monte Carlo, EDP, arbre binomial

Les quatre méthodes reposent sur la même mathématique, vue sous des angles
différents :

* **formule fermée** : calcul explicite de E^Q[e^{-rT}(S_T - K)^+] pour S_T lognormal ;
* **Monte Carlo** : loi des grands nombres sur la même espérance ;
* **EDP** : Feynman-Kac, l'espérance conditionnelle résout l'EDP de Black-Scholes ;
* **arbre** : marche aléatoire binomiale convergeant vers le brownien
  (Donsker), probabilité risque-neutre discrète.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..analytics.black_scholes import bs_greeks, bs_price, implied_vol
from ..analytics.lattice import leisen_reimer_price
from ..core.results import MCResult
from ..engine import ControlVariate, MonteCarloEngine
from ..models.equity import BlackScholes
from ..products.vanilla import EuropeanOption
from .pde import PDEResult, bs_pde_price

__all__ = ["BlackScholesPricer"]


@dataclass(frozen=True)
class BlackScholesPricer:
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    option_type: str = "call"

    def __post_init__(self):
        if min(self.S0, self.K, self.T, self.sigma) <= 0:
            raise ValueError("S0, K, T et sigma doivent être strictement positifs")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type ∈ {'call', 'put'}")

    # --- formule fermée ------------------------------------------------------
    def price(self) -> float:
        return float(bs_price(self.S0, self.K, self.T, self.r, self.sigma, self.q, self.option_type))

    def greeks(self) -> dict[str, float]:
        g = bs_greeks(self.S0, self.K, self.T, self.r, self.sigma, self.q, self.option_type)
        return {k: float(v) for k, v in g.items()}

    def implied_vol(self, market_price: float) -> float:
        return float(implied_vol(market_price, self.S0, self.K, self.T, self.r, self.q, self.option_type))

    # --- Monte Carlo -----------------------------------------------------------
    def model(self) -> BlackScholes:
        return BlackScholes(spot=self.S0, vol=self.sigma, rate=self.r, div=self.q)

    def monte_carlo(
        self,
        n_paths: int = 100_000,
        seed: int | None = 0,
        method: str = "pseudo",
        antithetic: bool = False,
        control_variate: bool = False,
    ) -> MCResult:
        """Simulation exacte de S_T (un seul pas). Variable de contrôle
        optionnelle : le sous-jacent actualisé, E[e^{-rT} S_T] = S0 e^{-qT}."""
        eng = MonteCarloEngine(n_paths, seed=seed, method=method, antithetic=antithetic)
        product = EuropeanOption(self.K, self.T, self.option_type)
        cvs: tuple[ControlVariate, ...] = ()
        if control_variate:
            cvs = (
                ControlVariate(
                    lambda p: p.spot[:, -1] * p.df(-1), self.S0 * np.exp(-self.q * self.T), "forward"
                ),
            )
        return eng.price(self.model(), product, cvs)

    # --- EDP et arbre ----------------------------------------------------------------
    def pde(self, american: bool = False, n_space: int = 400, n_time: int = 200) -> PDEResult:
        return bs_pde_price(
            self.S0, self.K, self.T, self.r, self.sigma, self.q, self.option_type, american, n_space, n_time
        )

    def tree(self, american: bool = False, n_steps: int = 1001) -> float:
        return leisen_reimer_price(
            self.S0, self.K, self.T, self.r, self.sigma, self.q, self.option_type, american, n_steps
        )

    # --- synthèse ----------------------------------------------------------------------
    def compare(self, n_paths: int = 200_000) -> list[dict]:
        """Tableau comparatif prix / écart à la formule fermée / temps de calcul."""
        ref = self.price()
        rows = []

        def add(name, fn):
            t0 = time.perf_counter()
            out = fn()
            dt = time.perf_counter() - t0
            price = out.price if hasattr(out, "price") else float(out)
            se = getattr(out, "stderr", None)
            rows.append({"méthode": name, "prix": price, "écart": price - ref, "err_std": se, "temps_s": dt})

        add("Formule fermée", self.price)
        add("MC pseudo-aléatoire", lambda: self.monte_carlo(n_paths))
        add(
            "MC antithétique + contrôle",
            lambda: self.monte_carlo(n_paths, antithetic=True, control_variate=True),
        )
        add("QMC Sobol (RQMC)", lambda: self.monte_carlo(2 ** int(np.log2(n_paths)), method="sobol"))
        add("EDP Crank-Nicolson", lambda: self.pde())
        add("Arbre Leisen-Reimer", lambda: self.tree())
        return rows
