import numpy as np
import pytest
from scipy.integrate import quad

from mcfin.analytics import (HestonParams, barrier_price, bs_greeks, bs_price, calibrate_heston,
                             heston_cf, heston_implied_vol, heston_price, implied_vol,
                             leisen_reimer_price, merton_price, sabr_hagan_vol)


def test_put_call_parity_and_greeks():
    S, K, T, r, q, v = 100, 95, 0.8, 0.03, 0.01, 0.25
    c, p = bs_price(S, K, T, r, v, q, "call"), bs_price(S, K, T, r, v, q, "put")
    assert np.isclose(c - p, S * np.exp(-q * T) - K * np.exp(-r * T))
    g = bs_greeks(S, K, T, r, v, q, "call")
    h = 1e-4
    assert np.isclose(g["delta"], (bs_price(S + h, K, T, r, v, q) - bs_price(S - h, K, T, r, v, q)) / (2 * h), atol=1e-6)
    assert np.isclose(g["vega"], (bs_price(S, K, T, r, v + h, q) - bs_price(S, K, T, r, v - h, q)) / (2 * h), atol=1e-5)
    assert np.isclose(g["gamma"], (bs_price(S + 1e-2, K, T, r, v, q) - 2 * c + bs_price(S - 1e-2, K, T, r, v, q)) / 1e-4, atol=1e-5)


def test_implied_vol_roundtrip(rng):
    n = 5000
    K = rng.uniform(40, 250, n)
    T = rng.uniform(0.02, 8, n)
    vol = rng.uniform(0.05, 1.2, n)
    otype = np.where(K >= 100, 1.0, -1.0)  # options hors de la monnaie
    price = bs_price(100, K, T, 0.02, vol, 0.0, otype)
    ok = price > 1e-10
    iv = implied_vol(price[ok], 100, K[ok], T[ok], 0.02, 0.0, otype[ok])
    assert np.nanmax(np.abs(iv - vol[ok])) < 1e-8


def test_heston_lewis_vs_quadrature():
    p = HestonParams(0.04, 1.5, 0.04, 0.8, -0.7)
    S0, r, q = 100, 0.03, 0.01
    for T in (0.05, 1.0, 5.0):
        F = S0 * np.exp((r - q) * T)
        for K in (70, 100, 130):
            k = np.log(F / K)
            f = lambda u: np.real(np.exp(1j * u * k) * heston_cf(u - 0.5j, T, p)) / (u * u + 0.25)
            ref = S0 * np.exp(-q * T) - np.sqrt(F * K) * np.exp(-r * T) / np.pi * quad(f, 0, np.inf, limit=2000, epsabs=1e-14, epsrel=1e-12)[0]
            assert abs(heston_price(S0, K, T, r, q, p)[0] - ref) < 1e-9


def test_heston_black_scholes_limit():
    p = HestonParams(0.04, 1.0, 0.04, 1e-4, 0.0)
    K = np.array([80, 100, 120.0])
    assert np.allclose(heston_price(100, K, 1.0, 0.02, 0.0, p), bs_price(100, K, 1.0, 0.02, 0.2), atol=1e-5)


def test_heston_calibration_recovers_parameters():
    true = HestonParams(0.03, 2.0, 0.05, 0.6, -0.65)
    T = np.repeat([0.25, 0.5, 1.0, 2.0], 5)
    K = np.tile([85.0, 95, 100, 105, 115], 4)
    vols = np.concatenate([heston_implied_vol(100, K[T == t], t, 0.02, 0.01, true) for t in np.unique(T)])
    p, info = calibrate_heston(100, 0.02, 0.01, T, K, vols)
    assert info["rmse_vol"] < 1e-6
    assert np.allclose([p.v0, p.kappa, p.theta, p.xi, p.rho], [0.03, 2.0, 0.05, 0.6, -0.65], rtol=1e-3)


@pytest.mark.parametrize("otype", ["call", "put"])
@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
@pytest.mark.parametrize("H,d", [(95.0, "down"), (105.0, "up")])
def test_barrier_in_out_parity(otype, K, H, d):
    args = (100, K, H, 1.0, 0.05, 0.25, 0.02, otype)
    s = barrier_price(*args, d + "-and-out") + barrier_price(*args, d + "-and-in")
    assert np.isclose(s, bs_price(100, K, 1.0, 0.05, 0.25, 0.02, otype))


def test_merton_reduces_to_bs():
    assert np.isclose(merton_price(100, 100, 1, 0.05, 0.2, 0.0, 0.0, 0.1), bs_price(100, 100, 1, 0.05, 0.2))


def test_leisen_reimer():
    assert abs(leisen_reimer_price(36, 40, 1, 0.06, 0.2, american=False) - bs_price(36, 40, 1, 0.06, 0.2, 0, "put")) < 1e-5
    assert abs(leisen_reimer_price(36, 40, 1, 0.06, 0.2) - 4.4867) < 1e-3  # référence littérature


def test_sabr_atm_and_lognormal_limit():
    # β = 1, ν -> 0 : vol lognormale constante α
    assert np.allclose(sabr_hagan_vol(0.03, np.array([0.02, 0.03, 0.04]), 1.0, 0.2, 1.0, 0.0, 1e-8), 0.2, atol=1e-6)
