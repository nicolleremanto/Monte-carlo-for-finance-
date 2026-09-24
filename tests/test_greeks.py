import numpy as np
import pytest

from mcfin import BlackScholes, DigitalOption, EuropeanOption, MonteCarloEngine
from mcfin.analytics import bs_digital_price, bs_greeks
from mcfin.greeks import aad, gbm_path_pricer, heston_pricer, likelihood_ratio_greeks


def test_bump_and_revalue_with_crn():
    m = BlackScholes(spot=100, vol=0.2, rate=0.03, div=0.01)
    g = MonteCarloEngine(n_paths=200_000, seed=1, antithetic=True).greeks(
        m, EuropeanOption(100, 1.0), {"vol": 0.01, "rate": 1e-3})
    ref = bs_greeks(100, 100, 1.0, 0.03, 0.2, 0.01)
    assert abs(g["delta"] - ref["delta"]) < 5e-3
    assert abs(g["gamma"] - ref["gamma"]) < 2e-3
    assert abs(g["vol"] - ref["vega"]) < 0.3
    assert abs(g["rate"] - ref["rho"]) < 0.4


def test_aad_matches_analytic_greeks():
    pricer = gbm_path_pricer(lambda s, xp: xp.maximum(s[-1] - 100.0, 0.0), [1.0], 1.0)
    z = np.random.default_rng(0).standard_normal((400_000, 1))
    g = aad.aad_greeks(pricer, {"spot": 100.0, "vol": 0.2, "rate": 0.03, "div": 0.01}, z)
    ref = bs_greeks(100, 100, 1.0, 0.03, 0.2, 0.01)
    for k, r in (("spot", ref["delta"]), ("vol", ref["vega"]), ("rate", ref["rho"])):
        assert abs(g[k] - r) < 4 * g[k + "_stderr"] + 1e-3


def test_aad_tape_elementary():
    tape = aad.Tape()
    x, y = tape.variable(1.5), tape.variable(np.array([0.5, 2.0]))
    f = aad.mean(aad.exp(x * y) / (1.0 + y) + aad.sqrt(y) * x**2 - aad.log(y))
    gx, gy = tape.gradient(f, [x, y])
    xv, yv = 1.5, np.array([0.5, 2.0])
    dfdx = np.mean(yv * np.exp(xv * yv) / (1 + yv) + np.sqrt(yv) * 2 * xv)
    dfdy = (xv * np.exp(xv * yv) / (1 + yv) - np.exp(xv * yv) / (1 + yv) ** 2
            + 0.5 * xv**2 / np.sqrt(yv) - 1 / yv) / 2
    assert np.isclose(gx, dfdx) and np.allclose(gy, dfdy)


def test_aad_heston_equals_finite_difference_limit():
    """Feller respecté : l'AAD est la limite exacte des différences finies (mêmes aléas)."""
    pricer = heston_pricer(100.0, 1.0, 25)
    z = np.random.default_rng(1).standard_normal((20_000, 25, 2))
    params = dict(spot=100.0, v0=0.04, kappa=3.0, theta=0.04, xi=0.3, rho=-0.7, rate=0.02)
    g = aad.aad_greeks(pricer, params, z, batch_size=20_000)

    def value(p):
        tape = aad.Tape()
        return float(pricer({k: tape.variable(v) for k, v in p.items()}, z).value)
    for k, h in (("spot", 1e-4), ("v0", 1e-7), ("theta", 1e-7), ("rho", 1e-6)):
        up, dn = dict(params), dict(params)
        up[k] += h
        dn[k] -= h
        fd = (value(up) - value(dn)) / (2 * h)
        assert abs(fd - g[k]) < 1e-3 * max(1.0, abs(g[k]))


def test_lrm_digital():
    m = BlackScholes(spot=100, vol=0.2, rate=0.03, div=0.01)
    g = likelihood_ratio_greeks(m, DigitalOption(105, 1.0), 400_000, seed=3)
    h = 1e-4
    ref_delta = (bs_digital_price(100 + h, 105, 1, 0.03, 0.2, 0.01)
                 - bs_digital_price(100 - h, 105, 1, 0.03, 0.2, 0.01)) / (2 * h)
    ref_vega = (bs_digital_price(100, 105, 1, 0.03, 0.2 + h, 0.01)
                - bs_digital_price(100, 105, 1, 0.03, 0.2 - h, 0.01)) / (2 * h)
    assert abs(g["delta"] - ref_delta) < 4 * g["delta_stderr"]
    assert abs(g["vega"] - ref_vega) < 4 * g["vega_stderr"]
