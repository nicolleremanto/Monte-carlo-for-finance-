import numpy as np
import pytest


def assert_mc(res, ref, n_sigma=4.0, abs_tol=0.0):
    """Le prix MC doit être à moins de n_sigma erreurs standard (+ tolérance
    absolue pour les biais de discrétisation connus) de la référence."""
    err = abs(res.price - ref)
    bound = n_sigma * res.stderr + abs_tol
    assert err <= bound, (
        f"MC={res.price:.6f} ± {res.stderr:.6f}, ref={ref:.6f}, écart={err:.2e} > {bound:.2e}"
    )


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
