import numpy as np
import pytest

from mcfin.core.rng import BrownianBridge, GaussianGenerator, PCAConstruction, poisson_inverse
from mcfin.core.timegrid import TimeGrid
from mcfin.market.curves import FlatCurve, InterpolatedCurve, NelsonSiegelSvensson


@pytest.mark.parametrize("cls", [BrownianBridge, PCAConstruction])
def test_constructions_preserve_law(cls, rng):
    """Pont brownien et ACP : incréments normalisés i.i.d. N(0, 1)."""
    times = np.array([0.1, 0.25, 0.3, 0.7, 1.0, 1.6, 2.0])
    z = rng.standard_normal((200_000, times.size, 1))
    out = cls(times).transform(z)[:, :, 0]
    assert np.allclose(np.cov(out.T), np.eye(times.size), atol=0.02)
    assert np.allclose(out.mean(axis=0), 0.0, atol=0.01)


def test_bridge_terminal_uses_first_coordinate():
    times = np.linspace(0.25, 1.0, 4)
    z = np.zeros((1, 4, 1))
    z[0, 0, 0] = 1.0
    dw = BrownianBridge(times).transform(z)[0, :, 0] * np.sqrt(0.25)
    assert np.isclose(dw.sum(), 1.0)  # W(T) = sqrt(T) z_0


def test_antithetic_and_moment_matching():
    g = GaussianGenerator("pseudo", 0, antithetic=True, moment_matching=True)
    z = g.normals(1000, np.array([0.0, 0.5, 1.0]), 2)
    assert np.allclose(z[:500], -z[500:])
    assert np.allclose(z.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(z.std(axis=0), 1, atol=1e-12)


def test_sobol_dimension_is_checked():
    g = GaussianGenerator("sobol", 0)
    g.normals(64, np.linspace(0, 1, 5), 2)
    with pytest.raises(ValueError):
        g.normals(64, np.linspace(0, 1, 6), 2)


def test_poisson_inverse_matches_scipy(rng):
    from scipy.stats import poisson

    u = rng.random(20_000)
    for mu in (0.01, 0.7, 5.0):
        assert np.array_equal(poisson_inverse(u, mu), poisson.ppf(u, mu))


def test_time_grid():
    g = TimeGrid.build([0.5, 1.0, 1.0, 0.25], max_dt=0.1)
    assert np.allclose(g.obs_times, [0.25, 0.5, 1.0])
    assert g.dt.max() <= 0.1 + 1e-12 and g.times[0] == 0
    u = TimeGrid.uniform([0.5, 1.0], 1 / 252)
    assert np.allclose(u.dt, u.dt[0]) and np.allclose(u.obs_times, [0.5, 1.0])


def test_curves():
    flat = FlatCurve(0.03)
    assert np.isclose(flat.df(2.0), np.exp(-0.06))
    interp = InterpolatedCurve([1, 2, 5], [0.02, 0.025, 0.03])
    assert np.isclose(interp.zero_rate(2.0), 0.025)
    nss = NelsonSiegelSvensson(0.03, -0.01, 0.01, 0.005)
    t = np.array([0.5, 3.0, 10.0])
    # forward instantané analytique = différences finies
    fd = -(np.log(nss.df(t + 1e-5)) - np.log(nss.df(t - 1e-5))) / 2e-5
    assert np.allclose(nss.inst_forward(t), fd, atol=1e-7)
    assert np.isclose(flat.shift(0.01).df(1.0), np.exp(-0.04))
