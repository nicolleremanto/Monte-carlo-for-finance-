"""mcfin — Monte Carlo pour la salle des marchés.

Librairie de pricing et de risque par simulation : modèles actions (BS,
Heston/Bates, Merton, SABR, volatilité locale, LSV, rough Bergomi), taux
(Hull-White, LMM), produits exotiques (asiatiques, barrières, autocalls,
bermudéennes), Greeks (bump CRN, pathwise, LRM, AAD), réduction de variance
(antithétiques, variables de contrôle, QMC Sobol + pont brownien,
stratification, échantillonnage préférentiel, MLMC) et risques (XVA, VaR/ES).
"""
from .core.results import MCResult, Paths
from .core.rng import GaussianGenerator, BrownianBridge, PCAConstruction
from .core.timegrid import TimeGrid
from .engine import MonteCarloEngine, ControlVariate
from .market.curves import FlatCurve, InterpolatedCurve, NelsonSiegelSvensson
from .market.volsurface import SSVISurface, FlatVolSurface
from .models import (BlackScholes, MertonJumpDiffusion, Heston, SABR, LocalVol,
                     LocalStochasticVol, RoughBergomi, bump_model)
from .products import (EuropeanOption, DigitalOption, BasketOption, RainbowOption,
                       AsianOption, BarrierOption, LookbackOption, Cliquet, VarianceSwap,
                       PhoenixAutocall, BermudanOption, LongstaffSchwartz,
                       andersen_broadie_upper_bound)

__version__ = "1.0.0"
