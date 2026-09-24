import numpy as np
import pytest

from mcfin import (BermudanOption, BlackScholes, LongstaffSchwartz, MonteCarloEngine,
                   andersen_broadie_upper_bound)
from mcfin.analytics import leisen_reimer_price


def _lsm(model, product, n=100_000):
    lsm = LongstaffSchwartz(degree=3).fit(
        MonteCarloEngine(n_paths=n, seed=1, antithetic=True).simulate(model, product.observation_times), product)
    lo = lsm.price(MonteCarloEngine(n_paths=n, seed=2).simulate(model, product.observation_times), product)
    return lsm, lo


def test_lsm_american_put_longstaff_schwartz_table():
    """Article de L-S (2001), table 1 : S=36, K=40, σ=20 %, T=1, r=6 %, 50 dates -> 4.472."""
    m = BlackScholes(spot=36, vol=0.2, rate=0.06)
    put = BermudanOption(40, np.arange(1, 51) / 50, "put")
    _, lo = _lsm(m, put)
    american = leisen_reimer_price(36, 40, 1, 0.06, 0.2)
    assert lo.price < american + 3 * lo.stderr        # borne inférieure
    assert abs(lo.price - 4.472) < 4 * lo.stderr + 0.01


@pytest.mark.slow
def test_andersen_broadie_brackets_max_call():
    """Andersen-Broadie (2004) : max-call 2 actifs, S0=100 -> intervalle publié [13.892, 13.934]."""
    m = BlackScholes(spot=np.array([100.0, 100.0]), vol=0.2, rate=0.05, div=0.1, corr=np.eye(2))
    mc = BermudanOption(100, np.arange(1, 10) / 3, "call", "max")
    lsm, lo = _lsm(m, mc, n=200_000)
    up = andersen_broadie_upper_bound(m, mc, lsm, lo.price, n_outer=500, n_inner=500)
    assert lo.price < 13.90 + 3 * lo.stderr
    assert up.price > 13.90 - 3 * up.stderr
    assert 0 < up.extra["duality_gap"] < 0.2
