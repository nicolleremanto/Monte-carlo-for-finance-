"""Produits dérivés évalués par simulation."""
from .base import Product, time_index
from .vanilla import EuropeanOption, DigitalOption, BasketOption, RainbowOption
from .path_dependent import AsianOption, BarrierOption, LookbackOption, Cliquet, VarianceSwap
from .autocall import PhoenixAutocall
from .american import BermudanOption, LongstaffSchwartz, andersen_broadie_upper_bound
