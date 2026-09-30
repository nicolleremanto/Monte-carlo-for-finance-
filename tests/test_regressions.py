"""Non-régression : un test par défaut relevé lors de la revue de code indépendante."""

import numpy as np
import pytest

import mcfin as mc
from mcfin.analytics import bs_price, lookback_floating_price
from mcfin.blackscholes import BlackScholesPricer, coverage_study, simulate_delta_hedge
from mcfin.market import InterpolatedCurve
from mcfin.market.curves import NelsonSiegelSvensson
from mcfin.rates import HullWhite


def test_andersen_broadie_rejects_non_flat_dividend_curve():
    model = mc.BlackScholes(spot=100, vol=0.2, rate=0.05, div=InterpolatedCurve([0.5, 1.0], [0.2, 0.1]))
    put = mc.BermudanOption(95, np.array([0.5, 1.0]), "put")
    lsm = mc.LongstaffSchwartz().fit(mc.MonteCarloEngine(20_000).simulate(model, put.observation_times), put)
    with pytest.raises(TypeError):
        mc.andersen_broadie_upper_bound(model, put, lsm, 7.5, n_outer=50, n_inner=50)


def test_andersen_broadie_stderr_includes_lower_bound_noise():
    model = mc.BlackScholes(spot=100, vol=0.2, rate=0.05, div=0.1)
    put = mc.BermudanOption(95, np.array([0.5, 1.0]), "put")
    lsm = mc.LongstaffSchwartz().fit(mc.MonteCarloEngine(50_000).simulate(model, put.observation_times), put)
    lo = lsm.price(mc.MonteCarloEngine(50_000, seed=9).simulate(model, put.observation_times), put)
    up = mc.andersen_broadie_upper_bound(
        model, put, lsm, lo.price, n_outer=200, n_inner=200, lower_bound_stderr=lo.stderr
    )
    assert up.stderr >= lo.stderr


def test_moment_matching_error_is_estimated_on_independent_groups():
    opt = BlackScholesPricer(100, 100, 1.0, 0.03, 0.2, option_type="put")

    def est(seed):
        eng = mc.MonteCarloEngine(2_048, seed=seed, moment_matching=True, n_randomizations=16)
        return eng.price(opt.model(), mc.EuropeanOption(100, 1.0, "put"))

    res = est(0)
    assert res.dof == 15
    rep = coverage_study(est, opt.price(), 200)
    assert 0.88 < rep.coverage < 0.99  # l'ancienne erreur « naïve » donnait 100 %


@pytest.mark.parametrize("label", ["c", "Call", "CALL"])
def test_option_type_aliases(label):
    ref_call = lookback_floating_price(100, 1, 0.05, 0.25, 0.02, "call")
    assert lookback_floating_price(100, 1, 0.05, 0.25, 0.02, label) == ref_call
    h = simulate_delta_hedge(100, 100, 1, 0.03, 0.2, 0.2, option_type=label, n_rebalancing=52, n_paths=2_000)
    assert abs(h.mean) < 0.2 and h.std < 2.0
    hw = HullWhite(0.05, 0.01, NelsonSiegelSvensson(0.03, -0.01, 0.01))
    assert hw.zcb_option(1.0, 2.0, 0.95, label) == hw.zcb_option(1.0, 2.0, 0.95, "call")
    paths = mc.MonteCarloEngine(20_000).simulate(mc.BlackScholes(spot=100, vol=0.2), np.arange(1, 13) / 12)
    lb = mc.LookbackOption(1.0, label, n_monitoring=12)
    assert np.allclose(lb.payoff(paths), mc.LookbackOption(1.0, "call", n_monitoring=12).payoff(paths))


def test_hedging_costs_equal_pnl_drag():
    kw = dict(S0=100, K=100, T=1.0, r=0.2, sigma_real=0.2, sigma_implied=0.2, n_rebalancing=4, n_paths=5_000)
    free = simulate_delta_hedge(**kw)
    costly = simulate_delta_hedge(cost=0.01, **kw)
    assert np.allclose(free.pnl - costly.pnl, costly.costs, atol=1e-10)


def test_lookback_is_continuous_at_r_equal_q():
    for ot in ("call", "put"):
        at = lookback_floating_price(100, 1, 0.02, 0.25, 0.02, ot)
        near = lookback_floating_price(100, 1, 0.02 + 1e-3, 0.25, 0.02, ot)
        assert np.isfinite(at) and abs(at - near) < 0.1


def test_simulate_with_drift_shift_returns_likelihood_ratio():
    model = mc.BlackScholes(spot=100, vol=0.2, rate=0.03)
    mu = np.array([[2.0]])
    paths = mc.MonteCarloEngine(200_000, seed=4).simulate(model, [1.0], drift_shift=mu)
    w = paths.extra["is_weight"]
    est = np.mean(w * np.maximum(paths.spot[:, -1] - 140, 0)) * np.exp(-0.03)
    assert abs(est / float(bs_price(100, 140, 1, 0.03, 0.2)) - 1) < 0.03
    assert abs(w.mean() - 1) < 0.02  # E_Q[dP/dQ] = 1


def test_reported_path_count_and_per_asset_deltas():
    eng = mc.MonteCarloEngine(10_001, seed=1, antithetic=True)
    res = eng.price(mc.BlackScholes(spot=100, vol=0.2), mc.EuropeanOption(100, 1.0))
    assert res.n_paths == 10_002  # réellement simulé (paires complètes)
    corr = np.array([[1, 0.5], [0.5, 1]])
    m2 = mc.BlackScholes(spot=np.array([100.0, 100.0]), vol=np.array([0.2, 0.3]), corr=corr)
    basket = mc.BasketOption(1.0, 1.0, np.array([0.7, 0.3]), initial_levels=np.array([100.0, 100.0]))
    g = mc.MonteCarloEngine(100_000, seed=2).greeks(m2, basket, second_order=False)
    d = g["delta_by_asset"]
    assert d[0] > d[1] > 0  # poids 70/30
    assert abs(d.sum() - g["delta"]) < 0.02 * g["delta"]
