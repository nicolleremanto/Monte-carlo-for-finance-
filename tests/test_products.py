import numpy as np
import pytest
from conftest import assert_mc

from mcfin import (AsianOption, BarrierOption, BasketOption, BlackScholes, Cliquet, ControlVariate,
                   EuropeanOption, Heston, LookbackOption, MonteCarloEngine, PhoenixAutocall,
                   RainbowOption, VarianceSwap)
from mcfin.analytics import barrier_price, bs_price, geometric_asian_price, lookback_floating_price
from mcfin.analytics.exotics import bgk_shift
from mcfin.analytics.heston import heston_expected_variance

BS = BlackScholes(spot=100, vol=0.25, rate=0.05, div=0.02)


@pytest.mark.parametrize("bt,K,H,ot", [
    ("down-and-out", 100, 90, "call"), ("down-and-in", 100, 90, "call"),
    ("up-and-out", 100, 120, "call"), ("up-and-in", 95, 110, "put"),
    ("down-and-out", 110, 85, "put"),
])
def test_barrier_brownian_bridge_correction(bt, K, H, ot):
    eng = MonteCarloEngine(n_paths=100_000, antithetic=True, seed=1)
    cont = eng.price(BS, BarrierOption(K, H, 1.0, ot, bt, n_monitoring=25, monitoring="continuous"))
    assert_mc(cont, barrier_price(100, K, H, 1.0, 0.05, 0.25, 0.02, ot, bt))
    # surveillance discrète ≈ continue avec barrière décalée (BGK)
    disc = eng.price(BS, BarrierOption(K, H, 1.0, ot, bt, n_monitoring=25, monitoring="discrete"))
    Hs = bgk_shift(H, 0.25, 1 / 25, "down" if bt.startswith("down") else "up")
    assert_mc(disc, barrier_price(100, K, Hs, 1.0, 0.05, 0.25, 0.02, ot, bt), abs_tol=0.03)


@pytest.mark.parametrize("ot", ["call", "put"])
def test_lookback_with_bgk(ot):
    res = MonteCarloEngine(n_paths=100_000, antithetic=True, seed=2).price(
        BS, LookbackOption(1.0, ot, 252, bgk_sigma=0.25))
    assert_mc(res, lookback_floating_price(100, 1.0, 0.05, 0.25, 0.02, ot), abs_tol=0.03)


def test_asian_geometric_and_control_variate():
    fix = np.arange(1, 13) / 12
    geo = AsianOption(100, fix, "call", "geometric")
    ref_geo = geometric_asian_price(100, 100, 1.0, 0.05, 0.25, fix, 0.02)
    eng = MonteCarloEngine(n_paths=100_000, seed=3)
    assert_mc(eng.price(BS, geo), ref_geo)
    arith = AsianOption(100, fix, "call")
    plain = eng.price(BS, arith)
    cv = eng.price(BS, arith, [ControlVariate(geo.payoff, ref_geo)])
    assert cv.extra["variance_reduction"] > 100
    assert_mc(plain, cv.price)
    assert cv.price > ref_geo  # moyenne arithmétique >= géométrique


def test_variance_swap_heston():
    h = Heston(spot=100, v0=0.06, kappa=2, theta=0.04, xi=0.5, rho=-0.7, rate=0.02, dt=1 / 252)
    res = MonteCarloEngine(n_paths=20_000, seed=4).price(h, VarianceSwap(1.0))
    fair = res.price / np.exp(-0.02)
    assert abs(fair - heston_expected_variance(1.0, h.params)) < 0.002


def test_basket_and_rainbow_degenerate_cases():
    m = BlackScholes(spot=np.array([100.0, 100.0]), vol=0.2, rate=0.03,
                     corr=np.array([[1, 0.999999], [0.999999, 1]]))
    eng = MonteCarloEngine(n_paths=100_000, seed=5)
    ref = bs_price(100, 100, 1.0, 0.03, 0.2) / 100
    assert_mc(eng.price(m, BasketOption(1.0, 1.0, np.array([0.5, 0.5]))), ref, abs_tol=1e-4)
    assert_mc(eng.price(m, RainbowOption(1.0, 1.0, "worst", "call")), ref, abs_tol=1e-4)


def test_cliquet_bounds():
    reset = np.arange(1, 13) / 12
    c = Cliquet(reset, -0.03, 0.03, 0.0, np.inf)
    res = MonteCarloEngine(n_paths=50_000, seed=6).price(BS, c)
    assert 0 < res.price < 12 * 0.03


def test_autocall_properties():
    corr = np.array([[1, .6, .5], [.6, 1, .55], [.5, .55, 1]])
    m = BlackScholes(spot=np.full(3, 100.0), vol=np.array([0.25, 0.3, 0.2]), rate=0.03,
                     div=np.array([0.02, 0.01, 0.03]), corr=corr)
    dates = np.arange(1, 21) / 4
    eng = MonteCarloEngine(n_paths=50_000, seed=7)
    base = eng.price(m, PhoenixAutocall(dates, coupon=0.02))
    more = eng.price(m, PhoenixAutocall(dates, coupon=0.03))
    safer = eng.price(m, PhoenixAutocall(dates, coupon=0.02, protection_barrier=0.4))
    assert more.price > base.price and safer.price > base.price
    ac = PhoenixAutocall(dates, coupon=0.02)
    an = ac.analytics(eng.simulate(m, ac.observation_times))
    assert np.isclose(an["redemption_prob_by_date"].sum(), 1.0)
    assert 0.25 <= an["expected_life"] <= 5.0
    # KI quotidien (américain) plus probable => prix plus bas
    daily = MonteCarloEngine(n_paths=10_000, seed=7).price(m, PhoenixAutocall(dates, coupon=0.02,
                                                                             ki_monitoring="daily"))
    assert daily.price < base.price
