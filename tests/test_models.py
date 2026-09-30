import numpy as np
import pytest
from conftest import assert_mc

from mcfin import (
    SABR,
    BlackScholes,
    EuropeanOption,
    Heston,
    LocalVol,
    MertonJumpDiffusion,
    MonteCarloEngine,
    RoughBergomi,
    SSVISurface,
)
from mcfin.analytics import (
    black_implied_vol,
    bs_price,
    heston_price,
    implied_vol,
    merton_price,
    sabr_hagan_vol,
)


@pytest.mark.parametrize(
    "kw",
    [
        dict(method="pseudo"),
        dict(method="pseudo", antithetic=True),
        dict(method="pseudo", moment_matching=True),
        dict(method="sobol"),
        dict(method="sobol", construction="bridge"),
        dict(method="pseudo", n_strata=64, batch_size=8192),
    ],
)
def test_black_scholes_all_rng(kw):
    m = BlackScholes(spot=100, vol=0.2, rate=0.03, div=0.01)
    res = MonteCarloEngine(n_paths=2**16, seed=1, **kw).price(m, EuropeanOption(105, 1.0))
    assert_mc(res, bs_price(100, 105, 1.0, 0.03, 0.2, 0.01))


def test_qmc_beats_pseudo():
    m = BlackScholes(spot=100, vol=0.2, rate=0.03)
    p = EuropeanOption(100, 1.0)
    a = MonteCarloEngine(n_paths=2**16, method="pseudo").price(m, p)
    b = MonteCarloEngine(n_paths=2**16, method="sobol").price(m, p)
    assert b.stderr < a.stderr / 10


def test_multi_asset_moments():
    corr = np.array([[1, 0.6], [0.6, 1]])
    m = BlackScholes(
        spot=np.array([100.0, 50.0]),
        vol=np.array([0.2, 0.3]),
        rate=0.02,
        div=np.array([0.0, 0.01]),
        corr=corr,
    )
    p = MonteCarloEngine(n_paths=200_000, seed=3).simulate(m, [1.0])
    s = p.spot[:, -1]
    assert np.allclose(s.mean(axis=0), [100 * np.exp(0.02), 50 * np.exp(0.01)], rtol=3e-3)
    assert abs(np.corrcoef(np.log(s).T)[0, 1] - 0.6) < 0.01


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_heston_qe_unbiased_with_coarse_steps(K):
    m = Heston(
        spot=100, v0=0.04, kappa=1.5, theta=0.04, xi=0.8, rho=-0.7, rate=0.03, div=0.01, scheme="qe", dt=1 / 8
    )
    res = MonteCarloEngine(n_paths=100_000, antithetic=True, seed=7).price(m, EuropeanOption(K, 2.0))
    assert_mc(res, heston_price(100, K, 2.0, 0.03, 0.01, m.params)[0])


def test_heston_euler_converges():
    m = Heston(
        spot=100, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, rate=0.03, scheme="euler", dt=1 / 100
    )
    res = MonteCarloEngine(n_paths=100_000, antithetic=True, seed=2).price(m, EuropeanOption(100, 1.0))
    assert_mc(res, heston_price(100, 100, 1.0, 0.03, 0.0, m.params)[0], abs_tol=0.03)


def test_bates_and_merton():
    b = Heston(
        spot=100,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.5,
        rho=-0.7,
        rate=0.03,
        lam=0.5,
        mu_j=-0.1,
        sigma_j=0.1,
        dt=1 / 16,
    )
    res = MonteCarloEngine(n_paths=100_000, seed=4).price(b, EuropeanOption(90, 1.0, "put"))
    assert_mc(res, heston_price(100, 90, 1.0, 0.03, 0.0, b.params, "put")[0])
    mj = MertonJumpDiffusion(spot=100, vol=0.2, lam=1.0, mu_j=-0.1, sigma_j=0.15, rate=0.05)
    res = MonteCarloEngine(n_paths=200_000, seed=5).price(mj, EuropeanOption(100, 1.0))
    assert_mc(res, merton_price(100, 100, 1, 0.05, 0.2, 1.0, -0.1, 0.15))


def test_local_vol_reprices_ssvi_smile():
    surf = SSVISurface(sigma0=0.18, sigma_inf=0.22, lam=1.0, rho=-0.6, eta=1.2, gamma=0.4)
    S0, r, q, T = 100.0, 0.03, 0.01, 1.0
    lv = LocalVol(spot=S0, surface=surf, rate=r, div=q, dt=1 / 100)
    eng = MonteCarloEngine(n_paths=100_000, antithetic=True, seed=11)
    F = S0 * np.exp((r - q) * T)
    for K in (80.0, 100.0, 120.0):
        ot = "call" if K >= F else "put"
        res = eng.price(lv, EuropeanOption(K, T, ot))
        iv = implied_vol(res.price, S0, K, T, r, q, ot)
        assert abs(iv - surf.implied_vol(np.log(K / F), T)) < 0.003  # < 30 bp


def test_ssvi_local_variance_derivatives():
    surf = SSVISurface(rho=-0.5, eta=1.0)
    y, T = np.array([-0.3, 0.0, 0.2]), 0.7
    h = 1e-5
    fd_T = (surf.total_variance(y, T + h) - surf.total_variance(y, T - h)) / (2 * h)
    fd_y = (surf.total_variance(y + h, T) - surf.total_variance(y - h, T)) / (2 * h)
    assert np.allclose(surf.w_T(y, T), fd_T, atol=1e-7)
    assert np.allclose(surf.w_y(y, T), fd_y, atol=1e-7)
    assert np.all(surf.local_variance(y, T) > 0)


def test_rough_bergomi_structure():
    rb = RoughBergomi(spot=100, xi0=0.04, eta=1.9, hurst=0.1, rho=-0.9)
    g = rb.build_grid([1.0])
    z = np.random.default_rng(1).standard_normal((20_000, g.n_steps, 3))
    Y, _ = rb.volterra(g.dt[0], g.n_steps, z)
    for i in (5, 50, 252):  # Var(Y_t) = t^{2H}
        assert abs(Y[:, i].var() / g.times[i] ** 0.2 - 1) < 0.05
    p = rb.simulate(g, z)
    assert abs(p.variance[:, -1].mean() / 0.04 - 1) < 0.03  # E[v_t] = ξ0
    res = MonteCarloEngine(n_paths=40_000, antithetic=True, seed=3).price(rb, EuropeanOption(0.0 + 1e-9, 1.0))
    assert_mc(res, 100.0)  # martingale
    # skew ATM négatif et plus raide à court terme
    eng = MonteCarloEngine(n_paths=40_000, antithetic=True, seed=4)
    skews = []
    for T in (0.1, 1.0):
        v = [
            implied_vol(
                eng.price(rb, EuropeanOption(K, T, "call" if K >= 100 else "put")).price,
                100,
                K,
                T,
                0,
                0,
                "call" if K >= 100 else "put",
            )
            for K in (95.0, 105.0)
        ]
        skews.append((v[1] - v[0]) / np.log(105 / 95))
    assert skews[0] < skews[1] < 0


def test_sabr_vs_hagan():
    F, T = 0.03, 1.0
    m = SABR(spot=F, alpha=0.2 * F**0.5, beta=0.5, rho=-0.3, nu=0.4, dt=1 / 100)
    eng = MonteCarloEngine(n_paths=100_000, antithetic=True, seed=6)
    for K in (0.025, 0.03, 0.035):
        ot = "call" if K >= F else "put"
        iv = black_implied_vol(eng.price(m, EuropeanOption(K, T, ot)).price, F, K, T, 1.0, ot)
        assert abs(iv - sabr_hagan_vol(F, K, T, m.alpha, 0.5, -0.3, 0.4)) < 0.004
