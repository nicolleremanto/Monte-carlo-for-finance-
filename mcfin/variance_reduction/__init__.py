"""Réduction de variance avancée (les techniques « à la source » sont dans core.rng)."""

from .importance_sampling import optimal_drift
from .mlmc import MLMCResult, gbm_level_sampler, heston_level_sampler, mlmc
