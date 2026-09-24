import numpy as np
import pytest

from mcfin import EuropeanOption, LocalStochasticVol, MonteCarloEngine, SSVISurface, Cliquet, LocalVol
from mcfin.analytics import implied_vol


@pytest.mark.slow
def test_lsv_particle_calibration_reprices_smile():
    surf = SSVISurface(sigma0=0.18, sigma_inf=0.22, lam=1.0, rho=-0.6, eta=1.2, gamma=0.4)
    S0, r, q = 100.0, 0.03, 0.01
    lsv = LocalStochasticVol(spot=S0, surface=surf, rate=r, div=q, v0=0.04, kappa=2.0,
                             theta=0.04, xi=0.3, rho=-0.6, dt=1 / 100).calibrate(1.0, 50_000)
    eng = MonteCarloEngine(n_paths=100_000, antithetic=True, seed=5)
    for T in (0.5, 1.0):
        F = S0 * np.exp((r - q) * T)
        for K in (80.0, 100.0, 120.0):
            ot = "call" if K >= F else "put"
            iv = implied_vol(eng.price(lsv, EuropeanOption(K, T, ot)).price, S0, K, T, r, q, ot)
            assert abs(iv - surf.implied_vol(np.log(K / F), T)) < 0.004   # < 40 bp
    # levier ~ plat quand la vol-stoch porte déjà le smile : valeurs positives et finies
    assert np.all(np.isfinite(lsv.leverage.values)) and np.all(lsv.leverage.values > 0)
