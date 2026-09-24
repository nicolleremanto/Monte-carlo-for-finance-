import numpy as np
from conftest import assert_mc

from mcfin import AsianOption, BlackScholes, EuropeanOption, MonteCarloEngine
from mcfin.analytics import bs_price, heston_price, HestonParams
from mcfin.variance_reduction import gbm_level_sampler, heston_level_sampler, mlmc, optimal_drift


def test_importance_sampling_deep_otm():
    m = BlackScholes(spot=100, vol=0.2, rate=0.03)
    p = EuropeanOption(180, 1.0)
    mu = optimal_drift(m, p)
    eng = MonteCarloEngine(n_paths=50_000, seed=1)
    plain, is_ = eng.price(m, p), eng.price(m, p, drift_shift=mu)
    assert_mc(is_, bs_price(100, 180, 1.0, 0.03, 0.2))
    assert (plain.stderr / is_.stderr) ** 2 > 100


def test_importance_sampling_path_dependent():
    m = BlackScholes(spot=100, vol=0.2, rate=0.03)
    p = AsianOption(150, np.arange(1, 13) / 12)
    mu = optimal_drift(m, p)
    is_ = MonteCarloEngine(n_paths=50_000, seed=2).price(m, p, drift_shift=mu)
    ref = MonteCarloEngine(n_paths=2_000_000, seed=3, batch_size=200_000).price(m, p, drift_shift=mu)
    assert_mc(is_, ref.price, n_sigma=5)
    assert is_.stderr < 0.02 * is_.price


def test_mlmc_gbm_milstein():
    res = mlmc(gbm_level_sampler(100, 100, 1, 0.05, 0.2, scheme="milstein"), eps=0.01, seed=1)
    assert abs(res.price - bs_price(100, 100, 1, 0.05, 0.2)) < 0.03
    v = np.array(res.variances[1:])
    assert np.all(v[1:] < v[:-1])          # V_l décroissante (β ≈ 2 pour Milstein)
    assert res.std_mc_cost > res.cost      # plus efficace qu'un MC standard


def test_mlmc_heston():
    ref = heston_price(100, 100, 1, 0.02, 0, HestonParams(0.04, 2.0, 0.04, 0.3, -0.7))[0]
    res = mlmc(heston_level_sampler(100, 100, 1, 0.02, 0.04, 2.0, 0.04, 0.3, -0.7), eps=0.02, seed=2)
    assert abs(res.price - ref) < 0.06
