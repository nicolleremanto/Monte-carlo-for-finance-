"""mcfin — Monte Carlo pour la salle des marchés.

Librairie de pricing et de risque par simulation : modèles actions (BS,
Heston/Bates, Merton, SABR, volatilité locale, LSV, rough Bergomi), taux
(Hull-White, LMM), produits exotiques (asiatiques, barrières, autocalls,
bermudéennes), Greeks (bump CRN, pathwise, LRM, AAD), réduction de variance
(antithétiques, variables de contrôle, QMC Sobol + pont brownien,
stratification, échantillonnage préférentiel, MLMC) et risques (XVA, VaR/ES).

Le sous-package ``mcfin.blackscholes`` traite le modèle de Black-Scholes de
A à Z : quatre méthodes de valorisation, EDP de Crank-Nicolson, couverture
delta discrète, inférence statistique et contrôle statistique du Monte Carlo.
"""

from . import blackscholes
from .blackscholes import BlackScholesPricer
from .core.results import MCResult, Paths
from .core.rng import BrownianBridge, GaussianGenerator, PCAConstruction
from .core.timegrid import TimeGrid
from .engine import ControlVariate, MonteCarloEngine
from .market.curves import FlatCurve, InterpolatedCurve, NelsonSiegelSvensson
from .market.volsurface import FlatVolSurface, SSVISurface
from .models import (
    SABR,
    BlackScholes,
    Heston,
    LocalStochasticVol,
    LocalVol,
    MertonJumpDiffusion,
    RoughBergomi,
    bump_model,
)
from .products import (
    AsianOption,
    BarrierOption,
    BasketOption,
    BermudanOption,
    Cliquet,
    DigitalOption,
    EuropeanOption,
    LongstaffSchwartz,
    LookbackOption,
    PhoenixAutocall,
    RainbowOption,
    VarianceSwap,
    andersen_broadie_upper_bound,
)

__version__ = "1.1.0"

__all__ = [
    "SABR",
    "AsianOption",
    "BarrierOption",
    "BasketOption",
    "BermudanOption",
    "BlackScholes",
    "BlackScholesPricer",
    "BrownianBridge",
    "Cliquet",
    "ControlVariate",
    "DigitalOption",
    "EuropeanOption",
    "FlatCurve",
    "FlatVolSurface",
    "GaussianGenerator",
    "Heston",
    "InterpolatedCurve",
    "LocalStochasticVol",
    "LocalVol",
    "LongstaffSchwartz",
    "LookbackOption",
    "MCResult",
    "MertonJumpDiffusion",
    "MonteCarloEngine",
    "NelsonSiegelSvensson",
    "PCAConstruction",
    "Paths",
    "PhoenixAutocall",
    "RainbowOption",
    "RoughBergomi",
    "SSVISurface",
    "TimeGrid",
    "VarianceSwap",
    "andersen_broadie_upper_bound",
    "blackscholes",
    "bump_model",
]
