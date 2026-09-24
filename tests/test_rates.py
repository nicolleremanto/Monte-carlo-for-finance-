import numpy as np

from mcfin.market import NelsonSiegelSvensson
from mcfin.rates import HullWhite, LiborMarketModel, bermudan_swaption_hw, bermudan_swaption_lmm

CURVE = NelsonSiegelSvensson(0.03, -0.01, 0.01, 0.005, 2.0, 5.0)
HW = HullWhite(a=0.05, sigma=0.01, curve=CURVE)


def _mc(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(x.size)


def test_hull_white_martingales_and_options():
    p = HW.simulate([2.0, 5.0, 10.0], 100_000, seed=0, antithetic=True)
    for T in (2.0, 5.0, 10.0):
        m, s = _mc(p.discount[:, p.index(T)])
        assert abs(m - CURVE.df(T)) < 4 * s + 1e-6
    i = p.index(5.0)
    m, s = _mc(p.discount[:, i] * HW.bond(5.0, 8.0, p.x[:, i]))
    assert abs(m - CURVE.df(8.0)) < 4 * s + 1e-6
    P = HW.bond(5.0, 6.0, p.x[:, i])
    m, s = _mc(p.discount[:, i] * P * np.maximum((1 / P - 1) - 0.035, 0))
    assert abs(m - HW.caplet(5.0, 6.0, 0.035)) < 4 * s
    pay = np.arange(6.0, 11.0)
    S, A = HW.swap_rate(5.0, pay, p.x[:, i])
    m, s = _mc(p.discount[:, i] * A * np.maximum(S - 0.035, 0))
    assert abs(m - HW.swaption(5.0, pay, 0.035)) < 4 * s


def test_bermudan_hw():
    single = bermudan_swaption_hw(HW, [5.0], 10.0, 0.033, n_paths=100_000)
    assert abs(single.price - single.extra["europeans"][0]) < 4 * single.stderr
    berm = bermudan_swaption_hw(HW, np.arange(1.0, 10.0), 10.0, 0.033, n_paths=50_000)
    assert berm.price > berm.extra["max_european"]


def test_lmm_numeraire_caplets_swaptions():
    lmm = LiborMarketModel(np.arange(0, 5.5, 0.5), CURVE, n_factors=10, steps_per_period=2)
    p = lmm.simulate(50_000, seed=1, antithetic=True)
    for k in (2, 6, 10):
        m, s = _mc(1 / p.numeraire[:, k])
        assert abs(m - CURVE.df(lmm.tenor[k])) < 4 * s + 1e-6
    for i in (2, 6, 9):
        K = lmm.L0[i]
        m, s = _mc(lmm.tau[i] * np.maximum(p.forwards[:, i, i] - K, 0) / p.numeraire[:, i + 1])
        assert abs(m - lmm.caplet_price(i, K)) < 4 * s + 2e-6
    S, A = p.swap(2, 10)
    K = S.mean()
    m, s = _mc(A * np.maximum(S - K, 0) / p.numeraire[:, 2])
    assert abs(m / lmm.swaption_rebonato(2, 10, K) - 1) < 0.02  # Rebonato ~1 % près


def test_bermudan_lmm_positive():
    lmm = LiborMarketModel(np.arange(0, 5.5, 0.5), CURVE, n_factors=3, steps_per_period=1)
    res = bermudan_swaption_lmm(lmm, 2, 10, 0.03, n_paths=20_000)
    euro = lmm.swaption_rebonato(2, 10, 0.03)
    assert res.price > 0.98 * euro
